"""`python -m koottam eval`: score a model, or the whole council, on the held-out test set.

Every run appends rows to results.csv, one per task. The two questions the project asks
are answered from that file:

1. Is the council more accurate than its best single member?
2. Is the fine-tuned student more accurate than the base student?
"""

import csv
import random
from datetime import UTC, datetime

import httpx

from koottam import data
from koottam.config import ROOT, Config
from koottam.council import Council, Member, ask_all
from koottam.schema import Question

RESULTS = ROOT / "results.csv"
FIELDS = ["when", "system", "task", "n", "correct", "accuracy", "no_answer", "errors", "note"]


def sample(questions: list[Question], limit: int | None) -> list[Question]:
    """A fixed random subset for cheap smoke runs. Same seed, same subset, so every model
    is scored on exactly the same questions."""
    if not limit or limit >= len(questions):
        return questions
    return random.Random(0).sample(questions, limit)


def record(
    system: str, questions: list[Question], choices: dict[str, str | None],
    errors: int = 0, note: str = "",
) -> None:
    new = not RESULTS.exists()
    with RESULTS.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for task in ("cyber", "injection", "all"):
            qs = [q for q in questions if task == "all" or q.task == task]
            if not qs:
                continue
            correct = sum(choices.get(q.id) == q.answer for q in qs)
            row = {
                "when": datetime.now(UTC).strftime("%Y-%m-%d %H:%M"),
                "system": system,
                "task": task,
                "n": len(qs),
                "correct": correct,
                "accuracy": f"{correct / len(qs):.3f}",
                "no_answer": sum(choices.get(q.id) is None for q in qs),
                "errors": errors if task == "all" else "",
                "note": note,
            }
            w.writerow(row)
            print(f"  {system:<18} {task:<10} {row['accuracy']}  ({correct}/{len(qs)})")


async def eval_model(config: Config, name: str, limit: int | None) -> None:
    questions = sample(data.read("test"), limit)
    async with httpx.AsyncClient(timeout=180) as http:
        member = Member(config.model(name), http)
        answers = await ask_all(member, questions)
    errs = [a.error for a in answers.values() if a.error]
    if errs:
        print(f"  {len(errs)} calls failed, e.g. {errs[0]}\n  (failures are not cached; rerun)")
    record(name, questions, {k: a.choice for k, a in answers.items()}, len(errs),
           note=f"limit={limit}" if limit else "")


async def eval_council(config: Config, limit: int | None) -> None:
    questions = sample(data.read("test"), limit)
    async with httpx.AsyncClient(timeout=180) as http:
        verdicts = await Council(config, http).decide(questions)
    by_agg = sum(v.decided_by == "aggregator" for v in verdicts)
    note = f"members={'+'.join(config.members)} aggregator_decided={by_agg}"
    if limit:
        note += f" limit={limit}"
    record("council", questions, {v.question_id: v.choice for v in verdicts}, note=note)
    print(f"  aggregator decided {by_agg}/{len(verdicts)} ties")
