"""`python -m koottam compare A B`: is B really better than A, or is the gap luck?

Both systems answered the same 616 test questions, so the comparison is paired. Questions
both got right, or both got wrong, say nothing about which is better. Only the flips do:
questions B got right and A missed ("gained"), and the reverse ("lost"). If B were no
better than A, each flip would be a coin toss, so McNemar's exact test asks how likely a
split at least this lopsided would be by chance.

It reads cached answers only and makes no calls.
"""

import math
import random

from koottam import data
from koottam.config import Config
from koottam.schema import Question
from koottam.store import AnswerStore


def mcnemar(gained: int, lost: int) -> float:
    """Two-sided exact McNemar p-value: the chance of a split at least this uneven if every
    flip were a fair coin toss."""
    n = gained + lost
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(gained, lost) + 1))
    outcomes: int = 2**n  # the equally likely ways n coin tosses can fall
    return min(1.0, 2 * tail / outcomes)


def bootstrap(diffs: list[int], rounds: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """95% interval for the mean per-question difference (B right minus A right), from
    resampling the questions with replacement."""
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(rounds))
    return means[int(0.025 * rounds) - 1], means[int(0.975 * rounds) - 1]


def correct(store: AnswerStore, q: Question) -> bool:
    ans = store.get(q.id)
    return ans is not None and ans.choice == q.answer


def compare(config: Config, a: str, b: str) -> None:
    questions = data.read("test")
    sa, sb = (AnswerStore(n, config.model(n).fingerprint) for n in (a, b))
    for name, store in ((a, sa), (b, sb)):
        if len(store) < len(questions):
            print(f"  {name} has answered {len(store)}/{len(questions)}; run eval first")
            return

    print(f"  {'task':<10} {'n':>4} {a:>16} {b:>16}  gained  lost  p (McNemar)")
    for task in ("cyber", "injection", "all"):
        qs = [q for q in questions if task == "all" or q.task == task]
        ra = sum(correct(sa, q) for q in qs)
        rb = sum(correct(sb, q) for q in qs)
        gained = sum(correct(sb, q) and not correct(sa, q) for q in qs)
        lost = sum(correct(sa, q) and not correct(sb, q) for q in qs)
        print(f"  {task:<10} {len(qs):>4} {ra / len(qs):>16.3f} {rb / len(qs):>16.3f}"
              f"  {gained:>6}  {lost:>4}  {mcnemar(gained, lost):.4f}")

    diffs = [correct(sb, q) - correct(sa, q) for q in questions]
    lo, hi = bootstrap(diffs)
    print(f"\n  {b} minus {a}: {100 * sum(diffs) / len(diffs):+.1f} points"
          f" (95% interval {100 * lo:+.1f} to {100 * hi:+.1f})")

    # On a two-option task, accuracy per true answer shows a lean towards one answer.
    for task in sorted({q.task for q in questions}):
        qs = [q for q in questions if q.task == task]
        if len(qs[0].options) != 2:
            continue
        print(f"\n  {task}, right per true answer:")
        for label, text in qs[0].options.items():
            ql = [q for q in qs if q.answer == label]
            print(f"    {label} {text.split(':')[0]:<18} n={len(ql):<4}"
                  f" {a} {sum(correct(sa, q) for q in ql)}/{len(ql)}"
                  f"   {b} {sum(correct(sb, q) for q in ql)}/{len(ql)}")

    # A distilled student should do best where the council was sure, and copy its mistakes
    # where the council was wrong.
    members = [AnswerStore(m, config.model(m).fingerprint) for m in config.members]
    full = len(members)
    print("\n  by how many council members got the question right:")
    for task in ("cyber", "injection"):
        for label, fewest, most in ((f"all {full}", full, full),
                                    (f"{full - 1} of {full}", full - 1, full - 1),
                                    (f"{full - 2} or fewer", 0, full - 2)):
            qs = [q for q in questions
                  if q.task == task and fewest <= sum(correct(m, q) for m in members) <= most]
            if qs:
                print(f"    {task:<10} {label:<11} n={len(qs):<4}"
                      f" {a} {sum(correct(sa, q) for q in qs) / len(qs):.3f}"
                      f"   {b} {sum(correct(sb, q) for q in qs) / len(qs):.3f}")
