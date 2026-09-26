"""`python -m koottam prepare`: download the datasets and write the question files.

Two tasks, both Apache-2.0 licensed so the training set built from them is ours to keep:

- cyber     tihanyin/CyberMetric: security MCQs, four options.
            TEST  = CyberMetric-500 (the published benchmark subset, so our numbers are
                    comparable with the paper's).
            TRAIN = CyberMetric-10000 minus every question also in the 500.
- injection deepset/prompt-injections: user inputs labelled injection / not, used as a
            two-option question. Its own train/test split is kept.

The rule the whole project rests on: no test question ever appears in training. The
dedupe compares normalised text, not ids, because the CyberMetric files are nested
subsets of each other with no shared ids.
"""

import hashlib
import io
import json
import re
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

from koottam.config import DATA
from koottam.schema import Question

HF = "https://huggingface.co/datasets"
CYBERMETRIC = HF + "/tihanyin/CyberMetric/resolve/main/CyberMetric-{n}-v1.json"
INJECTION = HF + "/deepset/prompt-injections/resolve/main/data/{split}-00000-of-00001-{h}.parquet"
INJECTION_HASH = {"train": "9564e8b05b4757ab", "test": "701d16158af87368"}

INJECTION_OPTIONS = {
    "A": "Safe: an ordinary request or statement",
    "B": "Prompt injection: tries to override or hijack an AI assistant's instructions",
}


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def qid(task: str, text: str) -> str:
    return f"{task}-{hashlib.sha1(normalise(text).encode()).hexdigest()[:12]}"


def fetch(url: str, cache: Path) -> bytes:
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {url.rsplit('/', 1)[-1]}")
        with urllib.request.urlopen(url, timeout=120) as r:
            cache.write_bytes(r.read())
    return cache.read_bytes()


def cybermetric(n: int) -> list[Question]:
    raw = json.loads(fetch(CYBERMETRIC.format(n=n), DATA / "raw" / f"cybermetric-{n}.json"))
    out = []
    for item in raw["questions"]:
        q = Question(
            id=qid("cyber", item["question"]),
            task="cyber",
            source=f"CyberMetric-{n}",
            question=item["question"].strip(),
            options={k: str(v).strip() for k, v in item["answers"].items()},
            answer=item["solution"].strip().upper(),
        )
        if q.answer in q.options:  # skip the odd malformed row rather than train on it
            out.append(q)
    return out


def injection(split: str) -> list[Question]:
    url = INJECTION.format(split=split, h=INJECTION_HASH[split])
    table = pq.read_table(io.BytesIO(fetch(url, DATA / "raw" / f"injection-{split}.parquet")))
    out = []
    for row in table.to_pylist():
        text = str(row["text"]).strip()
        out.append(
            Question(
                id=qid("injection", text),
                task="injection",
                source=f"deepset/prompt-injections:{split}",
                question=(
                    "Classify the user input between the markers.\n\n"
                    f"<<<\n{text}\n>>>"
                ),
                options=INJECTION_OPTIONS,
                answer="B" if int(row["label"]) == 1 else "A",
            )
        )
    return out


def dedupe(questions: list[Question]) -> list[Question]:
    seen: dict[str, Question] = {}
    for q in questions:
        seen.setdefault(q.id, q)
    return list(seen.values())


def near_duplicate(a: set[str], b: set[str], threshold: float = 0.8) -> bool:
    """Word-overlap (Jaccard) test. Catches the same question with a typo or small
    rewording, e.g. 'wwhat is a common first line of defense…', which exact matching
    misses. Found by checking prepare's counts on 2026-09-26."""
    return bool(a | b) and len(a & b) / len(a | b) >= threshold


def split() -> tuple[list[Question], list[Question]]:
    test = dedupe(cybermetric(500) + injection("test"))
    test_ids = {q.id for q in test}
    test_words = [set(normalise(q.question).split()) for q in test]
    train = []
    for q in dedupe(cybermetric(10000) + injection("train")):
        words = set(normalise(q.question).split())
        if q.id not in test_ids and not any(near_duplicate(words, t) for t in test_words):
            train.append(q)
    return train, test


def write(path: Path, questions: list[Question]) -> None:
    path.write_text("".join(q.model_dump_json() + "\n" for q in questions), encoding="utf-8")


def read(name: str) -> list[Question]:
    path = DATA / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} missing: run `python -m koottam prepare` first")
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Question.model_validate_json(line) for line in lines if line.strip()]


def prepare() -> None:
    train, test = split()
    DATA.mkdir(exist_ok=True)
    write(DATA / "train.jsonl", train)
    write(DATA / "test.jsonl", test)
    for name, qs in (("train", train), ("test", test)):
        counts = {t: sum(q.task == t for q in qs) for t in ("cyber", "injection")}
        print(f"{name}: {len(qs)} questions {counts}")
    overlap = {q.id for q in train} & {q.id for q in test}
    print(f"train/test overlap: {len(overlap)} (must be 0)")
