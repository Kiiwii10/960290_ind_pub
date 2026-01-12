import os
from functools import lru_cache
from typing import Any, Dict, List, Tuple, Set, Optional
import re


from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from config import Config
from dotenv import load_dotenv

load_dotenv()

config = Config()


class PromptRequest(BaseModel):
    question: str


BASE_DIR = os.path.dirname(__file__)
FRONTEND_DIST = os.path.join(BASE_DIR, "frontend", "dist")


app = FastAPI()

if os.path.isdir(FRONTEND_DIST):
    app.mount("/app", StaticFiles(directory=FRONTEND_DIST), name="frontend_assets")


# --- Overlap-aware prompt context rendering (does NOT change API `context`) ---
try:
    import tiktoken  # type: ignore
    _ENC = tiktoken.get_encoding("gpt2")

    def _encode_tokens(s: str) -> List[int]:
        return _ENC.encode(s)

    def _decode_tokens(toks: List[int]) -> str:
        return _ENC.decode(toks)

except Exception:
    # Fallback: whitespace “tokens” (still reduces redundancy; less precise)
    def _encode_tokens(s: str) -> List[str]:
        return s.split()

    def _decode_tokens(toks: List[str]) -> str:
        return " ".join(toks)


def retrieve_context(vs, question: str) -> List[Tuple[Any, float]]:
    """
    TRUE strict top_k semantics:
    - exactly one vector search call
    - return results in the order Pinecone returns them
    """
    top_k = int(config.top_k)
    if top_k <= 0:
        return []
    return vs.similarity_search_with_score(question, k=top_k)


_TRANSCRIPT_HEADER_RE = re.compile(r'^Talk:\s*".*?"\s*\|\s*Speaker:.*?\n\n', re.DOTALL)

def _strip_transcript_header(text: str) -> str:
    # We already show title/speaker in the context item header line, so strip the repeated body header.
    return re.sub(_TRANSCRIPT_HEADER_RE, "", text, count=1)


def _drop_leading_overlap_tokens(text: str, n_tokens: int) -> str:
    toks = _encode_tokens(text)
    if n_tokens <= 0 or len(toks) <= n_tokens:
        return text
    trimmed = _decode_tokens(toks[n_tokens:])
    return trimmed.lstrip()


def build_user_prompt(question: str, selected: List[Tuple[Any, float]]) -> str:
    """
    Keep the output 'context' unchanged (raw doc.page_content),
    but reduce redundancy inside the model prompt:
      - strip repeated Talk/Speaker header inside transcript chunks
      - if two consecutive chunks from same talk are adjacent (chunk_index + 1),
        drop the known overlap from the *second* chunk
      - if the same (talk_id, type, chunk_index) repeats in selected,
        keep the header but omit repeated body (point back to first occurrence)
    """
    overlap_tokens = int(config.chunk_size * config.overlap_ratio)

    blocks: List[str] = []
    # Track first occurrence so we can collapse duplicates in the *prompt* only
    first_seen: Dict[Tuple[str, str, str], int] = {}
    # Track previous chunk per talk in prompt order (to trim overlap only when consecutive in prompt)
    prev_chunk_index_by_talk: Dict[str, int] = {}

    for i, (doc, score) in enumerate(selected, start=1):
        md = doc.metadata or {}
        talk_id = str(md.get("talk_id", ""))
        title = str(md.get("title", ""))
        chunk_type = str(md.get("chunk_type", ""))
        chunk_index_raw = str(md.get("chunk_index", ""))
        url = str(md.get("url", ""))

        header = (
            f"({i}) talk_id={talk_id} | title={title} | type={chunk_type} | "
            f"chunk_index={chunk_index_raw} | score={score:.4f} | url={url}"
        )

        key = (talk_id, chunk_type, chunk_index_raw)

        if key in first_seen:
            # IMPORTANT: we are not hiding retrieval — we show it happened,
            # but we avoid feeding the same text twice to the model.
            first_i = first_seen[key]
            body_for_prompt = f"(duplicate of context item {first_i}; body omitted to reduce redundancy)"
            blocks.append(header + "\n" + body_for_prompt)
            continue

        first_seen[key] = i

        body_for_prompt = doc.page_content

        if chunk_type == "transcript":
            # 1) remove repeated Talk/Speaker line inside the chunk
            body_for_prompt = _strip_transcript_header(body_for_prompt)

            # 2) trim overlap only if the previous chunk in the prompt is from the same talk
            #    and is exactly chunk_index-1 (i.e., adjacent)
            try:
                ci = int(float(chunk_index_raw))
            except Exception:
                ci = None

            if ci is not None:
                prev_ci = prev_chunk_index_by_talk.get(talk_id)
                if prev_ci is not None and ci == prev_ci + 1:
                    body_for_prompt = _drop_leading_overlap_tokens(body_for_prompt, overlap_tokens)

                prev_chunk_index_by_talk[talk_id] = ci

        blocks.append(header + "\n" + body_for_prompt)

    context_text = "\n\n---\n\n".join(blocks) if blocks else "(no context retrieved)"

    return (
        f"Question:\n{question}\n\n"
        "TED dataset context (metadata/transcript passages):\n"
        f"{context_text}\n\n"
        "Answer using ONLY the context above and follow the citation rules."
    )

# def build_user_prompt(question: str, selected: List[Tuple[Any, float]]) -> str:
#     blocks = []
#     for i, (doc, score) in enumerate(selected, start=1):
#         md = doc.metadata or {}
#         talk_id = str(md.get("talk_id", ""))
#         title = str(md.get("title", ""))
#         chunk_type = str(md.get("chunk_type", ""))
#         chunk_index = str(md.get("chunk_index", ""))

#         header = f"[{i}] talk_id={talk_id} | title={title} | type={chunk_type} | chunk_index={chunk_index} | score={score:.4f}"
#         blocks.append(header + "\n" + doc.page_content)

#     context_text = "\n\n---\n\n".join(blocks) if blocks else "(no context retrieved)"

#     return (
#         f"Question:\n{question}\n\n"
#         "TED dataset context (metadata/transcript passages):\n"
#         f"{context_text}\n\n"
#         "Answer using ONLY the context above and follow the citation rules."
#     )




@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    key =   os.getenv("API_KEY_MANAGEMENT")
    url =   os.getenv("BASE_URL")
    model = os.getenv("EMBEDDING_MODEL")

    if not (key and url and model):
        raise RuntimeError("Missing embedding configuration (API_KEY_MANAGEMENT/BASE_URL/EMBEDDING_MODEL)")
    return OpenAIEmbeddings(api_key=key, base_url=url, model=model)


@lru_cache(maxsize=1)
def get_vector_store() -> PineconeVectorStore:
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    cloud = os.getenv("PINECONE_CLOUD", "aws")
    region = os.getenv("PINECONE_REGION", "us-east-1")
    namespace = os.getenv("PINECONE_NAMESPACE", "default")

    if not api_key or not index_name:
        raise RuntimeError("Missing PINECONE_API_KEY or PINECONE_INDEX_NAME")

    pc = Pinecone(api_key=api_key)
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=config.pc_dimension,
            metric=config.pc_metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )

    index = pc.Index(index_name)
    return PineconeVectorStore(index=index, embedding=get_embeddings(), namespace=namespace)


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    key =      os.getenv("API_KEY_MANAGEMENT")
    endpoint = os.getenv("BASE_URL")
    model =    os.getenv("CHAT_MODEL")
    if not endpoint or not key:
        raise RuntimeError("Missing API_KEY_MANAGEMENT or BASE_URL")

    return ChatOpenAI(base_url=endpoint, api_key=key, model=model, temperature=1)


def _dedupe_keep_order(items: List[Tuple[Any, float]]) -> List[Tuple[Any, float]]:
    seen: Set[tuple] = set()
    out: List[Tuple[Any, float]] = []
    for doc, score in items:
        md = doc.metadata or {}
        key = (str(md.get("talk_id", "")), str(md.get("chunk_type", "")), str(md.get("chunk_index", "")))
        if key in seen:
            continue
        seen.add(key)
        out.append((doc, score))
    return out


def retrieve_context(vs: PineconeVectorStore, question: str) -> List[Tuple[Any, float]]:
    """Strict top_k semantics:
    - We retrieve at most config.top_k total chunks from Pinecone (counting meta + transcript).
    - No oversampling + post-filtering that changes what top_k means.
    - We still enforce diversity by first retrieving talk-level meta docs (1 per talk),
      then fetching up to chunks_per_talk transcript chunks per selected talk until budget is used.
    """

    top_k = int(config.top_k)
    if top_k <= 0:
        return []

    selected: List[Tuple[Any, float]] = []

    # 1) Talk-level selection (meta docs). One meta doc per talk by construction.
    meta_k = min(3, top_k)  # assignment only needs up to 3 distinct talks in lists
    meta = vs.similarity_search_with_score(question, k=meta_k, filter={"chunk_type": "meta"})
    meta = _dedupe_keep_order(meta)
    selected.extend(meta[:meta_k])

    remaining = top_k - len(selected)
    if remaining <= 0:
        return selected[:top_k]

    # 2) Evidence: fetch transcript chunks per chosen talk_id
    for doc, _score in meta:
        if remaining <= 0:
            break
        talk_id = str((doc.metadata or {}).get("talk_id", ""))
        if not talk_id:
            continue

        k_for_talk = min(int(config.chunks_per_talk), remaining)
        tr = vs.similarity_search_with_score(
            question,
            k=k_for_talk,
            filter={"chunk_type": "transcript", "talk_id": talk_id},
        )
        tr = _dedupe_keep_order(tr)
        selected.extend(tr[:k_for_talk])
        remaining = top_k - len(selected)

    # 3) If still remaining, fill with best transcript chunks globally (still within top_k)
    if remaining > 0:
        fill = vs.similarity_search_with_score(question, k=remaining, filter={"chunk_type": "transcript"})
        fill = _dedupe_keep_order(fill)
        selected.extend(fill[:remaining])

    return selected[:top_k]


@app.get("/api/stats")
def stats() -> Dict[str, Any]:
    return {"chunk_size": config.chunk_size, "overlap_ratio": config.overlap_ratio, "top_k": config.top_k}


@app.post("/api/prompt")
def prompt(req: PromptRequest) -> Dict[str, Any]:
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    vs = get_vector_store()
    llm = get_llm()

    selected = retrieve_context(vs, question)

    system_prompt = config.system_prompt
    user_prompt = build_user_prompt(question, selected)

    if not selected:
        return {
            "response": config.unknown_answer,
            "context": [],
            "Augmented_prompt": {"System": system_prompt, "User": user_prompt},
        }

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    ai_msg = llm.invoke(messages)
    response_text = ai_msg.content if hasattr(ai_msg, "content") else str(ai_msg)

    context_items = []
    for doc, score in selected:
        md = doc.metadata or {}
        context_items.append(
            {
                "talk_id": str(md.get("talk_id", "")),
                "title": str(md.get("title", "")),
                "chunk": doc.page_content,
                "score": float(score),
            }
        )

    print("----LLM Response----\n")
    print("responese_text:\n", response_text, "\n")
    print("context_items:\n", context_items, "\n")
    print("agumented_prompt:\n", {"System": system_prompt, "User": user_prompt}, "\n")
    print("----end LLM Response----\n")

    return {
        "response": response_text,
        "context": context_items,
        "Augmented_prompt": {"System": system_prompt, "User": user_prompt},
    }


@app.get("/", response_class=FileResponse)
def serve_frontend() -> FileResponse:
    index_path = os.path.join(FRONTEND_DIST, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=500, detail="Frontend build is missing")
    return FileResponse(index_path)
