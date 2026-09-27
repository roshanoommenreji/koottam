"""`python -m koottam weights`: learn how much each member's vote should count.

Equal votes lost to the best single member on the test set (92.7% vs 94.3%): two members
that often shared the same wrong answer outvoted the most accurate one. The fix is the
classic optimal rule for combining independent voters (Nitzan & Paroush, 1982):

    weight = log( p * (k - 1) / (1 - p) )

p is the member's accuracy and k the number of options. It is the log-odds of the member
being right versus picking one particular wrong option, so a member at chance (p = 1/k)
gets weight 0, and each halving of the error rate adds a constant amount of weight.

The accuracies are measured on TRAIN questions the members have already answered (cached
by `build`), per task, because members differ a lot between tasks (gpt-oss: 93.6% on
security but 75.9% on injection). Measuring them on the test set would be fitting the
answer key; the test set is only used afterwards, to see whether the weights help.
"""

import json
import math

from koottam import data
from koottam.build import train_order
from koottam.config import DATA, Config
from koottam.store import AnswerStore

WEIGHTS = DATA / "weights.json"


def weight(correct: int, total: int, k: int) -> float:
    # Laplace smoothing: (c+1)/(n+2) keeps p off 0 and 1 (log would blow up) and pulls
    # small samples towards the middle instead of trusting a lucky 20/20.
    p = (correct + 1) / (total + 2)
    return max(0.0, math.log(p * (k - 1) / (1 - p)))


def learn(config: Config, limit: int) -> dict[str, dict[str, float]]:
    questions = train_order(data.read("train"))[:limit]
    stores = {m: AnswerStore(m, config.teachers[m].fingerprint) for m in config.members}
    # Only questions every member has answered, so all members are judged on the same set.
    answered = [q for q in questions if all(s.get(q.id) for s in stores.values())]
    print(f"learning from {len(answered)} train questions every member has answered")

    result: dict[str, dict[str, float]] = {}
    for task in ("cyber", "injection"):
        qs = [q for q in answered if q.task == task]
        if not qs:
            continue
        k = len(qs[0].options)
        result[task] = {}
        for m, store in stores.items():
            correct = 0
            for q in qs:
                a = store.get(q.id)
                correct += bool(a and a.choice == q.answer)
            result[task][m] = round(weight(correct, len(qs), k), 3)
            print(f"  {task:<10} {m:<12} {correct}/{len(qs)} = {correct / len(qs):.3f}"
                  f"  -> weight {result[task][m]}")
    WEIGHTS.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {WEIGHTS}")
    return result


def load() -> dict[str, dict[str, float]]:
    if not WEIGHTS.exists():
        raise SystemExit(f"{WEIGHTS} missing: run `python -m koottam weights` first")
    loaded: dict[str, dict[str, float]] = json.loads(WEIGHTS.read_text(encoding="utf-8"))
    return loaded
