import argparse
import json
import re
import sys
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import os
from dotenv import load_dotenv

load_dotenv()


UNKNOWN_ANSWERS = {
    "I don’t know based on the provided TED data.",
    "I don't know based on the provided TED data.",
}


def http_post_json(url: str, payload: dict, timeout: int = 60) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def normalize_line(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^\s*[-*]\s*", "", s)
    s = re.sub(r"^\s*\d+[\).\s]+", "", s)
    return s.strip()


def extract_speaker_candidates_from_chunk(chunk: str):
    candidates = set()

    m = re.search(r"speaker_1:\s*(.+)", chunk, re.IGNORECASE)
    if m:
        candidates.add(m.group(1).strip())

    m2 = re.search(r"Speaker:\s*([^\n|]+)", chunk, re.IGNORECASE)
    if m2:
        candidates.add(m2.group(1).strip())

    m3 = re.search(r"all_speakers:\s*(.+)", chunk, re.IGNORECASE)
    if m3:
        raw = m3.group(1).strip()
        for part in re.split(r"[;,]", raw):
            p = part.strip()
            if p:
                candidates.add(p)

    return [c for c in candidates if len(c) >= 2]


def assert_common_response_shape(out: dict):
    for k in ["response", "context", "Augmented_prompt"]:
        if k not in out:
            raise AssertionError(f"Missing key '{k}' in response JSON")
    if not isinstance(out["context"], list):
        raise AssertionError("context must be a list")
    if "System" not in out["Augmented_prompt"] or "User" not in out["Augmented_prompt"]:
        raise AssertionError("Augmented_prompt must contain System and User")


_CITATION_RE = re.compile(r"\[\s*([^\[\]:]+?)\s*:\s*([^\[\]]+?)\s*\]")


def extract_citations(text: str):
    """Return list of (talk_id, title) tuples cited in the response."""
    return [(m.group(1).strip(), m.group(2).strip()) for m in _CITATION_RE.finditer(text)]


def _normalize_title(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()


def assert_valid_citations(out: dict):
    resp = out["response"]
    cites = extract_citations(resp)

    if not cites:
        raise AssertionError("No citations found in response. Expected citations like [<talk-id>:<title>].")

    context = out["context"]
    lookup = {}
    for ctx in context:
        talk_id = str(ctx.get("talk_id", "")).strip()
        title = str(ctx.get("title", "")).strip()
        if not (talk_id and title):
            continue
        lookup.setdefault(talk_id, set()).add(_normalize_title(title))

    invalid = []
    for talk_id, cited_title in cites:
        normalized_title = _normalize_title(cited_title)
        if not talk_id or not normalized_title:
            invalid.append(f"[{talk_id}:{cited_title}]")
            continue
        titles = lookup.get(talk_id)
        if not titles or normalized_title not in titles:
            invalid.append(f"[{talk_id}:{cited_title}]")

    if invalid:
        raise AssertionError("Invalid citations (missing talk_id/title match in context): " + ", ".join(invalid))


def assert_dataset_grounding(out: dict):
    titles = [c.get("title", "") for c in out["context"] if c.get("title")]
    if not titles:
        raise AssertionError("No titles in context; cannot verify grounding")
    resp = out["response"]
    if not any(t in resp for t in titles):
        raise AssertionError("Response does not contain any title from retrieved context (possible hallucination)")


def test_goal1(out: dict):
    assert_valid_citations(out)
    assert_dataset_grounding(out)

    resp = out["response"]
    for ctx in out["context"]:
        title = ctx.get("title", "")
        chunk = ctx.get("chunk", "")
        if title and title in resp:
            speakers = extract_speaker_candidates_from_chunk(chunk)
            if any(sp in resp for sp in speakers):
                return
    raise AssertionError("Goal1: Could not find (title AND matching speaker) supported by retrieved chunks")


def test_goal2(out: dict):
    assert_valid_citations(out)

    titles = [c.get("title", "") for c in out["context"] if c.get("title")]
    resp_lines = [normalize_line(x) for x in out["response"].splitlines()]
    resp_lines = [x for x in resp_lines if x]

    if len(resp_lines) != 3:
        raise AssertionError(f"Goal2: Expected exactly 3 non-empty lines, got {len(resp_lines)}")

    matched_titles = []
    for line in resp_lines:
        # require each line to have a citation
        if not _CITATION_RE.search(line):
            raise AssertionError("Goal2: Each title line must include a citation like [<talk-id>:<title>].")

        match = next((t for t in titles if t and t in line), None)
        if not match:
            raise AssertionError("Goal2: A returned line does not contain any title from context")
        matched_titles.append(match)

    if len(set(matched_titles)) != 3:
        raise AssertionError("Goal2: Titles are not 3 distinct talks (duplicates detected)")


def test_goal3(out: dict):
    assert_valid_citations(out)
    assert_dataset_grounding(out)

    resp = out["response"]
    if "Title:" not in resp or "Key idea:" not in resp:
        raise AssertionError("Goal3: Expected 'Title:' and 'Key idea:' sections")

    m = re.search(r"Key idea:\s*(.+)", resp, re.IGNORECASE | re.DOTALL)
    if not m:
        raise AssertionError("Goal3: Could not parse Key idea section")
    key_idea = m.group(1).strip()
    if len(key_idea.split()) < 25:
        raise AssertionError("Goal3: Key idea summary too short (<25 words)")


def test_goal4(out: dict):
    assert_valid_citations(out)
    assert_dataset_grounding(out)

    resp = out["response"]
    if "Recommendation:" not in resp or "Why:" not in resp:
        raise AssertionError("Goal4: Expected 'Recommendation:' and 'Why:' sections")

    m = re.search(r"Why:\s*(.+)", resp, re.IGNORECASE | re.DOTALL)
    if not m:
        raise AssertionError("Goal4: Could not parse Why section")
    why = m.group(1).strip()
    if len(why.split()) < 30:
        raise AssertionError("Goal4: Justification too short (<30 words)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=None, help="Base URL for the API (default from questions.json or http://localhost:8000)")
    ap.add_argument("--questions", default="tests/questions.json")
    args = ap.parse_args()

    with open(args.questions, "r", encoding="utf-8") as f:
        spec = json.load(f)

    base_url = os.getenv("BASE_URL_TEST_RUN", "http://localhost:8000")
    endpoint = base_url.rstrip("/") + "/api/prompt"

    failures = 0

    for case in spec["cases"]:
        cid = case["id"]
        q = case["question"]
        print(f"\n== Running: {cid} ==")
        try:
            out = http_post_json(endpoint, {"question": q})
            assert_common_response_shape(out)

            if out.get("response", "").strip() in UNKNOWN_ANSWERS:
                raise AssertionError("Model responded with 'I don’t know…'")

            if cid == "goal1_precise_fact":
                test_goal1(out)
            elif cid == "goal2_list_3":
                test_goal2(out)
            elif cid == "goal3_summary":
                test_goal3(out)
            elif cid == "goal4_recommend":
                test_goal4(out)
            else:
                raise AssertionError(f"Unknown test id: {cid}")

            print("PASS")

        except (HTTPError, URLError) as e:
            failures += 1
            print(f"FAIL (HTTP): {e}")
        except Exception as e:
            failures += 1
            print(f"FAIL: {e}")

    print("\n====================")
    if failures:
        print(f"Tests failed: {failures}")
        sys.exit(1)
    print("All tests passed!")
    sys.exit(0)


if __name__ == "__main__":
    main()
