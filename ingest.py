import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import TokenTextSplitter
from config import Config
from dotenv import load_dotenv

load_dotenv()

MANIFEST_PATH = "ingest_manifest.json"


def load_manifest() -> Dict[str, dict]:
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(m: Dict[str, dict]) -> None:
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def _transcript_preview(transcript: str, max_chars: int = 2400) -> str:
    """Cheap 'talk-level' semantic signal without creating a second index/namespace.
    We sample beginning/middle/end so themes that appear later can still be represented.
    """
    t = (transcript or "").strip()
    if not t:
        return ""
    if len(t) <= max_chars:
        return t

    # 3 slices
    third = max_chars // 3
    a = t[:third]
    mid_start = max(0, (len(t) // 2) - (third // 2))
    b = t[mid_start : mid_start + third]
    c = t[-third:]
    return a + "\n...\n" + b + "\n...\n" + c


def build_metadata_text(row: Dict[str, str]) -> str:
    """Lean metadata doc (1 per talk).
    Avoid embedding every column; keep only what helps retrieval + attribution.
    """
    transcript = (row.get("transcript") or "").strip()
    preview = _transcript_preview(transcript, max_chars=2400)

    fields = [
        ("talk_id", row.get("talk_id", "")),
        ("title", row.get("title", "")),
        ("speaker_1", row.get("speaker_1", "")),
        ("all_speakers", row.get("all_speakers", "")),
        ("occupations", row.get("occupations", "")),
        ("topics", row.get("topics", "")),
        ("recorded_date", row.get("recorded_date", "")),
        ("published_date", row.get("published_date", "")),
        ("event", row.get("event", "")),
        ("duration", row.get("duration", "")),
        ("views", row.get("views", "")),
        ("url", row.get("url", "")),
        ("description", row.get("description", "")),
    ]
    lines = [f"{k}: {v}".strip() for k, v in fields if (v or "").strip()]

    if preview:
        lines.append("transcript_preview:")
        lines.append(preview)

    return "TED Talk Overview\n" + "\n".join(lines)


def get_embeddings() -> OpenAIEmbeddings:
    key =   os.getenv("API_KEY_MANAGEMENT")
    url =   os.getenv("BASE_URL")
    model = os.getenv("EMBEDDING_MODEL")


    if not (key and url and model):
        raise RuntimeError("Missing embedding configuration (API_KEY_MANAGEMENT/BASE_URL/EMBEDDING_MODEL)")
    return OpenAIEmbeddings(api_key=key, base_url=url, model=model)


def get_vector_store() -> PineconeVectorStore:
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    cloud = os.getenv("PINECONE_CLOUD", "aws")
    region = os.getenv("PINECONE_REGION", "us-east-1")
    namespace = os.getenv("PINECONE_NAMESPACE", "default")
    dimension = int(os.getenv("PINECONE_DIMENSION", "1536"))
    metric = os.getenv("PINECONE_METRIC", "cosine")

    if not api_key or not index_name:
        raise RuntimeError("Missing PINECONE_API_KEY or PINECONE_INDEX_NAME")

    pc = Pinecone(api_key=api_key)
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )

    index = pc.Index(index_name)
    return PineconeVectorStore(index=index, embedding=get_embeddings(), namespace=namespace)


def iter_rows(csv_path: str, limit: Optional[int]) -> List[Dict[str, str]]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            rows.append(row)
            if limit is not None and (i + 1) >= limit:
                break
    return rows


def make_docs_for_talk(row: Dict[str, str], cfg: Config) -> Tuple[List[Document], List[str], str]:
    talk_id = (row.get("talk_id") or "").strip()
    title = (row.get("title") or "").strip()
    speaker_1 = (row.get("speaker_1") or "").strip()
    url = (row.get("url") or "").strip()

    transcript = (row.get("transcript") or "").strip()
    meta_text = build_metadata_text(row)

    combined_hash = sha1_text(meta_text + "\n\n" + transcript)
    docs: List[Document] = []
    ids: List[str] = []

    # 1) overview/meta doc (1 per talk)
    docs.append(
        Document(
            page_content=meta_text,
            metadata={
                "talk_id": talk_id,
                "title": title,
                "speaker_1": speaker_1,
                "url": url,
                "chunk_type": "meta",
                "chunk_index": -1,
            },
        )
    )
    ids.append(f"{talk_id}::meta")

    # 2) transcript chunks (evidence)
    if transcript:
        splitter = TokenTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=int(cfg.chunk_size * cfg.overlap_ratio),
        )
        chunks = splitter.split_text(transcript)

        for idx, chunk in enumerate(chunks):
            chunk_text = f'Talk: "{title}" | Speaker: {speaker_1}\n\n{chunk}'
            docs.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "talk_id": talk_id,
                        "title": title,
                        "speaker_1": speaker_1,
                        "url": url,
                        "chunk_type": "transcript",
                        "chunk_index": idx,
                    },
                )
            )
            ids.append(f"{talk_id}::t{idx}")

    return docs, ids, combined_hash


def main() -> None:
    cfg = Config()
    manifest = load_manifest()
    vs = get_vector_store()

    rows = iter_rows(cfg.csv_path, None)

    batch_docs: List[Document] = []
    batch_ids: List[str] = []
    updated = 0
    skipped = 0

    for row in rows:
        talk_id = (row.get("talk_id") or "").strip()
        if not talk_id:
            continue

        docs, ids, content_hash = make_docs_for_talk(row, cfg)

        prior = manifest.get(talk_id)
        if prior and prior.get("hash") == content_hash and \
           prior.get("chunk_size") == cfg.chunk_size and \
           prior.get("overlap_ratio") == cfg.overlap_ratio:
            skipped += 1
            continue

        batch_docs.extend(docs)
        batch_ids.extend(ids)

        manifest[talk_id] = {
            "hash": content_hash,
            "chunk_size": cfg.chunk_size,
            "overlap_ratio": cfg.overlap_ratio,
            "num_vectors": len(ids),
        }
        updated += 1

        if len(batch_docs) >= cfg.batch_size:
            vs.add_documents(batch_docs, ids=batch_ids)
            batch_docs.clear()
            batch_ids.clear()
            save_manifest(manifest)
            print(f"Upserted batch. updated={updated} skipped={skipped}")

    if batch_docs:
        vs.add_documents(batch_docs, ids=batch_ids)
        save_manifest(manifest)

    print(f"Done. updated={updated} skipped={skipped}. Manifest saved to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
