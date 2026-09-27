"""Offline tests: no network, no model. They pin the logic every result depends on."""

from pathlib import Path

import pytest

from koottam.client import RateLimiter
from koottam.council import vote
from koottam.data import dedupe, near_duplicate, normalise, qid
from koottam.prompts import parse_choice, strip_answer_line
from koottam.schema import Answer, Question
from koottam.store import AnswerStore

ABCD = {"A", "B", "C", "D"}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Firewalls filter traffic.\nANSWER: C", "C"),
        ("reasoning...\n**ANSWER: b**", "B"),
        ("ANSWER: (D)", "D"),
        ("I will give ANSWER: <letter> at the end.\nANSWER: A", "A"),  # last one wins
        ("A", "A"),
        ("ANSWER: E", None),  # not an offered option
        ("I am not sure.", None),
        ("", None),
    ],
)
def test_parse_choice(text: str, expected: str | None) -> None:
    assert parse_choice(text, ABCD) == expected


def test_strip_answer_line() -> None:
    assert strip_answer_line("Because X.\nSo Y.\n\nANSWER: B\n") == "Because X.\nSo Y."


def test_vote_plurality_wins() -> None:
    assert vote({"a": "B", "b": "B", "c": "A", "d": None}) == ("B", 2)


def test_vote_tie_has_no_winner() -> None:
    assert vote({"a": "A", "b": "B", "c": "A", "d": "B"}) == (None, 2)


def test_vote_nobody_answered() -> None:
    assert vote({"a": None, "b": None}) == (None, 0)


def test_normalised_ids_catch_near_duplicates() -> None:
    # The CyberMetric files overlap with different whitespace/punctuation.
    assert normalise("What is  XSS?") == normalise("what is XSS")
    assert qid("cyber", "What is XSS?") == qid("cyber", " what is xss ")


def test_near_duplicate_catches_typos_not_different_questions() -> None:
    def words(s: str) -> set[str]:
        return set(normalise(s).split())

    base = "what is a common first line of defense in cybersecurity to prevent attacks"
    assert near_duplicate(words(base), words("w" + base))
    assert not near_duplicate(words(base), words("what does a firewall log record"))


def _q(text: str) -> Question:
    return Question(id=qid("cyber", text), task="cyber", source="t", question=text,
                    options={"A": "x", "B": "y"}, answer="A")


def test_dedupe_keeps_first() -> None:
    assert len(dedupe([_q("Same?"), _q("same"), _q("Other?")])) == 2


def test_store_caches_answers_but_not_errors(tmp_path: Path) -> None:
    store = AnswerStore("m", "model-1", root=tmp_path)
    store.put(Answer(member="m", model="model-1", question_id="q1", choice="A", text="ANSWER: A"))
    store.put(Answer(member="m", model="model-1", question_id="q2", choice=None, text="",
                     error="HTTPStatusError: 401"))
    reloaded = AnswerStore("m", "model-1", root=tmp_path)
    assert reloaded.get("q1") is not None
    assert reloaded.get("q2") is None  # failures are retried, not remembered


def test_store_ignores_answers_from_a_different_model(tmp_path: Path) -> None:
    AnswerStore("m", "old", root=tmp_path).put(
        Answer(member="m", model="old", question_id="q1", choice="A", text="ANSWER: A"))
    assert AnswerStore("m", "new", root=tmp_path).get("q1") is None


async def test_rate_limiter_spaces_requests() -> None:
    import time

    limiter = RateLimiter(rpm=600)  # one start every 0.1 s
    t0 = time.monotonic()
    for _ in range(4):
        await limiter.wait()
    assert time.monotonic() - t0 >= 0.29


def test_placeholder_key_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    from koottam.config import ModelConfig

    m = ModelConfig(name="x", provider="p", base_url="u", model="m", api_key_env="K")
    monkeypatch.setenv("K", "replace-me")
    assert m.api_key is None  # a placeholder must read as "no key", not cause a 401
    monkeypatch.setenv("K", "real")
    assert m.api_key == "real"


def test_strip_thoughts_removes_private_reasoning() -> None:
    from koottam.client import strip_thoughts

    raw = "<thought>* Options: A...\nANSWER: <letter></thought>WPA2 encrypts traffic.\n\nANSWER: B"
    assert strip_thoughts(raw) == "WPA2 encrypts traffic.\n\nANSWER: B"
    assert strip_thoughts("<think>hmm</think>Plain.") == "Plain."
    assert strip_thoughts("No tags here.") == "No tags here."


def test_bigger_build_extends_smaller_one() -> None:
    from koottam.build import train_order

    pool = [_q(f"question number {i}") for i in range(10150)]
    small = {q.id for q in train_order(pool)[:500]}
    big = {q.id for q in train_order(pool)[:3000]}
    assert small <= big  # cached answers from the small run are all reused
