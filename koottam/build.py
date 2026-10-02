"""`python -m koottam build`: turn the council's answers on TRAIN questions into a
fine-tuning set, data/sft.jsonl.

An example is kept only when
  (a) at least `min_agree` members voted for the same answer, AND
  (b) that answer matches the dataset's answer key.

(b) is a luxury: real-world data has no key, and (a) alone would have to do. So the build
also reports how often (a) passed but (b) failed, which measures how much you could
trust council agreement on its own.

The training target is the reasoning of the first agreeing member in council order
(strongest first), with its ANSWER line rewritten into one canonical form. The student
learns to reason the council's way, not just to emit a letter.

The filter is not neutral about answers. On injection, the council agrees easily that a
text is safe and less often that it is an attack, so the first build kept 78 of 78 safe
texts but 37 of 50 attacks, and the student learnt to lean "safe". Two options fix that:
--all-injection asks about every injection question in the pool, not only those within
--limit, and --balance then keeps as many examples of each answer as of the rarest one.
"""

import json
import random
from collections import Counter
from typing import Any

import httpx

from koottam import data
from koottam.config import DATA, Config
from koottam.council import Council
from koottam.prompts import INSTRUCTION, SYSTEM, render_question, strip_answer_line
from koottam.schema import Question

SFT = DATA / "sft.jsonl"


def train_order(pool: list[Question]) -> list[Question]:
    """One fixed shuffle of the train pool. `build --limit N` takes its first N, so a later,
    bigger build is a superset of an earlier one and reuses every cached answer.
    (random.sample does not guarantee that: it switches algorithm with the sample size.)"""
    order = list(pool)
    random.Random(0).shuffle(order)
    return order


def answer_of(row: dict[str, Any]) -> str:
    """The letter a training example teaches: its target always ends 'ANSWER: <letter>'."""
    return str(row["messages"][-1]["content"]).rsplit("ANSWER: ", 1)[1]


def balance(rows: list[dict[str, Any]], task: str) -> list[dict[str, Any]]:
    """Every answer of `task` keeps as many examples as its rarest answer has; the latest
    examples of the commoner answers are dropped. Other tasks' rows pass through."""
    counts = Counter(answer_of(r) for r in rows if r["task"] == task)
    if not counts:
        return rows
    cap = min(counts.values())
    seen: Counter[str] = Counter()
    kept = []
    for r in rows:
        if r["task"] == task:
            seen[answer_of(r)] += 1
            if seen[answer_of(r)] > cap:
                continue
        kept.append(r)
    return kept


async def build(config: Config, limit: int | None, all_injection: bool = False,
                balanced: bool = False) -> None:
    pool = data.read("train")
    order = train_order(pool)
    questions = order[:limit] if limit else pool
    if limit and all_injection:
        questions += [q for q in order[limit:] if q.task == "injection"]
    print(f"building from {len(questions)} of {len(pool)} train questions")

    async with httpx.AsyncClient(timeout=180) as http:
        council = Council(config, http)
        # No tie-breaks: a tie has at most half the votes, below min_agree, so the
        # aggregator's call could never produce a kept example. Saves its daily quota.
        verdicts = await council.decide(questions, use_aggregator=False)
        answers = {m.cfg.name: m.store for m in council.members}

    by_id = {q.id: q for q in questions}
    stats: Counter[str] = Counter()
    rows = []
    for v in verdicts:
        q = by_id[v.question_id]
        agreed = v.choice is not None and v.agree >= config.min_agree
        correct = v.choice == q.answer
        stats[f"{q.task}:total"] += 1
        stats[f"{q.task}:{q.answer}:total"] += 1
        if agreed and not correct:
            stats[f"{q.task}:agreed_but_wrong"] += 1
        if not (agreed and correct):
            continue
        reasoning = next(
            (
                strip_answer_line(a.text)
                for name in config.members
                if (a := answers[name].get(q.id)) and a.choice == v.choice
            ),
            "",
        )
        if len(reasoning) < 20:  # a bare letter teaches nothing
            stats[f"{q.task}:no_reasoning"] += 1
            continue
        rows.append({
            "id": q.id,
            "task": q.task,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"{render_question(q)}\n\n{INSTRUCTION}"},
                {"role": "assistant", "content": f"{reasoning}\nANSWER: {v.choice}"},
            ],
        })
        stats[f"{q.task}:kept"] += 1
        stats[f"{q.task}:{q.answer}:kept"] += 1

    # Two-option tasks: show how evenly the filter kept each answer, and even it out.
    two_way = sorted({q.task for q in questions if len(q.options) == 2})
    for task in two_way:
        options = sorted({q.answer for q in questions if q.task == task})
        print(f"  {task} kept per true answer: " + "  ".join(
            f"{a} {stats[f'{task}:{a}:kept']}/{stats[f'{task}:{a}:total']}" for a in options))
        if balanced:
            before = len(rows)
            rows = balance(rows, task)
            print(f"  {task} balanced: dropped the latest {before - len(rows)} examples of the"
                  f" commoner answer, leaving {sum(r['task'] == task for r in rows)}")

    SFT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                   encoding="utf-8")
    print(f"wrote {len(rows)} examples to {SFT}")
    for task in ("cyber", "injection"):
        t = stats[f"{task}:total"]
        if t:
            print(f"  {task:<10} kept {stats[f'{task}:kept']}/{t}  "
                  f"agreed-but-wrong {stats[f'{task}:agreed_but_wrong']}  "
                  f"no-reasoning {stats[f'{task}:no_reasoning']}")
