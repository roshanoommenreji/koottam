# Koottam

[![ci](https://github.com/roshanoommenreji/koottam/actions/workflows/ci.yml/badge.svg)](https://github.com/roshanoommenreji/koottam/actions/workflows/ci.yml)
[![licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)

**Showcase page: [roshanoommenreji.github.io/koottam](https://roshanoommenreji.github.io/koottam/)**

> **Status:** both claims are measured, see [Results](#results).
> 1. The council does *not* beat its best member: 93.5% vs 94.3%.
> 2. The fine-tuned student *does* beat its untrained self: 66.7% vs 60.4% (p = 0.008).
>    Almost all of that gain is prompt-injection detection, and it came with a lean towards
>    "safe" that the training filter caused.

*Koottam* (കൂട്ടം) is Malayalam for "a gathering". This repo gathers several AI models,
each on a different server, into a council, and then distils what the council knows into
one small model you can run on a laptop.

It sets out to **measure** two claims rather than assume them:

1. **A council of models is more accurate than its best single member.**
2. **A small model fine-tuned on the council's answers beats the same small model untrained.**

The domain is **defensive cyber security**, plus one AI-safety task: spotting
prompt-injection attacks aimed at AI assistants.

## How it works

```
                      ┌─► gpt-oss-120b      Cerebras          (server 1)
questions ─► koottam ─┼─► Qwen 3.8 27B      Groq              (server 2)
  (laptop)            ├─► Gemma 4 31B       Google AI Studio  (server 3)
                      └─► Gemma 4 E2B       Ollama            (server 4: this laptop)
                                 │
                   vote ◄────────┘   tie? ─► aggregator: Nemotron 3 Super (OpenRouter)
                                                         reads all the reasoning
                     │
          ┌──────────┴───────────┐
   test questions          train questions
   score it               keep answers where ≥3 agree AND the key agrees
   results.csv            data/sft.jsonl ─► LoRA fine-tune ─► koottam-student (Ollama)
```

Every model speaks the same OpenAI-compatible API, so adding a server is a few lines in
[config/teachers.toml](config/teachers.toml). Every answer is cached in `data/answers/`, which
means runs resume after a rate limit, and scoring the council costs no extra calls.

**One member per provider** is deliberate. Free tiers cap requests per *day* per account.
OpenRouter's free tier allows 50 a day, too few for a council member but enough for the
tie-breaker. Its `:free` models also share one pool across all free users, and on
2026-09-27 some were rate-limited upstream for everyone.

## Data

| Task | Source | Licence | Train | Test |
|---|---|---|---|---|
| Security knowledge (4-option MCQ) | [CyberMetric](https://huggingface.co/datasets/tihanyin/CyberMetric) | Apache-2.0 | 9,614 | 500 (the published CyberMetric-500) |
| Prompt-injection detection (A safe / B injection) | [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) | Apache-2.0 | 536 | 116 |

No test question is in training: exact duplicates **and** near-duplicates (≥80% word
overlap, which catches typo'd copies) are removed. `prepare` prints the overlap, which
must be 0.

## Quick start

```bash
uv venv -p 3.11 .venv && uv pip install -e ".[dev]"
git config core.hooksPath .githooks      # gitleaks on every commit
cp .env.example .env                     # add 4 free keys: Cerebras, Groq, Google AI Studio, OpenRouter
ollama pull gemma4:e2b

python -m koottam prepare                # download + split data
python -m koottam status                 # which servers answer
python -m koottam eval --model gpt-oss --limit 20    # smoke test one member
python -m koottam eval --model gpt-oss   # full test set, each member in turn
python -m koottam eval --council         # the council, from cached answers
python -m koottam build --limit 2500     # training set from the council
# fine-tune on a free GPU: train/finetune.ipynb (Lab 02), then
python -m koottam students               # serve base + fine-tuned student in Ollama
python -m koottam eval --model student-base
python -m koottam eval --model student-koottam
python -m koottam compare student-base student-koottam   # paired, with significance
python -m koottam build --limit 2500 --all-injection --balance   # run 2: balanced injection lessons
```

## Results

Full held-out test set: members and council scored 2026-09-27/28, students 2026-10-02.
Every system answered all 616 questions.

| System | Cyber (500) | Injection (116) | Overall |
|---|---|---|---|
| gpt-oss-120b (Cerebras) | 93.6% | 75.9% | 90.3% |
| Qwen 3.8 27B (Groq) | 94.2% | 81.0% | 91.7% |
| **Gemma 4 31B (Google)** | **96.2%** | **86.2%** | **94.3%** |
| Gemma 4 E2B (laptop) | 84.4% | 81.0% | 83.8% |
| **Council**, equal votes (tie-breaker on 19 ties) | 95.2% | 82.8% | **92.9%** |
| **Council**, weighted votes (learned on train) | 95.6% | 84.5% | **93.5%** |
| Student, base (Gemma 3 1B) | 63.0% | 49.1% | 60.4% |
| **Student, fine-tuned** (Gemma 3 1B + LoRA) | 64.4% | 76.7% | **66.7%** |

*(Equal votes first scored 92.7%; a re-score from cache reached 92.9% because one failed
tie-break was asked again and came out right.)*

**Claim 1, "the council beats its best member": not supported, even with weighting.** Equal
votes beat three of four members and their average (89.5%), but lost to Gemma 4 31B by 1.4
points. Weighting each vote by the member's accuracy on 499 *training* questions
(`log(p(k−1)/(1−p))`, see [koottam/weights.py](koottam/weights.py)) closed about 40% of that
gap, to 93.5%, and needed no tie-breaks at all. Why equal votes lose, from the saved answers:

- **The right answer was almost always in the room.** On 608 of 616 questions (98.7%) at
  least one member was right, so knowledge wasn't the problem. Choosing whom to believe was.
- **Equal votes let correlated mistakes outvote the best member.** gpt-oss and Qwen gave the
  *same* wrong answer 32 times, the most of any pair. On 9 questions, Gemma 31B was the only
  member that was right, and it was simply outvoted.
- **Removing the weak laptop member doesn't fix it** (3 cloud members: 92.9%). The issue is
  equal weighting, not one bad voter.

Weighting was measured honestly: the weights came from training questions only. It helped
most on injection (82.8% → 84.5%), where members differ most (weights 0.74 to 1.91).

**A second finding: the training file's answer key is noisy.** On 44 of 500 training
questions, 3–4 members agreed on an answer the key calls wrong (16 of them unanimously).
Spot checks show real key errors (the key calls "divide a function among several people"
*Split Knowledge*; all four said *Two-Person Control*) and genuinely ambiguous questions. It
also explains why members score ~86% on training questions but ~94% on the curated
CyberMetric-500 test set.

This doesn't block distillation. The training filter keeps an example only when ≥3 members
agree **and** the key agrees, so the student learns from answers that are correct either way.

**Claim 2, "the student beats its untrained self": supported, overall.** The same Gemma 3 1B,
LoRA-trained for 9.6 minutes on a free T4 with 2,094 council-agreed answers, went from 60.4%
to 66.7%. Both students answered the same questions, so the test is paired: the student
gained 123 questions and lost 84, a split that uneven would happen by chance 0.8% of the
time (exact McNemar test; 95% interval +1.8 to +11.0 points).
`python -m koottam compare student-base student-koottam` reproduces these numbers. What the
gain actually is:

- **Nearly all of it is prompt-injection detection** (49.1% → 76.7%, p = 0.0002). Security
  knowledge barely moved (63.0% → 64.4%: 71 gained, 64 lost, p = 0.61). The fine-tune
  changed how the 1B model behaves, not what it knows.
- **The base flagged almost everything as an attack**: 113 of 116 texts. It caught 57/60
  attacks and recognised 0/56 safe texts. The student recognises 52/56 safe texts but
  catches only 37/60 attacks.
- **The training filter caused that lean.** It kept all 78 safe injection examples but only
  37 of 50 attacks, because the teachers disagree more about attacks, so 68% of the
  injection lessons said "safe". An agreement filter is not neutral about labels. For a
  security detector, missing 23 of 60 attacks is the wrong trade; run 2 rebalances it
  ([Lab 02](docs/labs/02-distill.md#run-2-balance-the-injection-lessons)) and fixes it, below.
- **It inherited the council's blind spots.** On cyber questions all four members got right
  it improved (69.5% → 73.3%); where two or fewer were right, it got worse (37.1% → 25.7%).
- **None of it is format.** Each student wrote a parseable answer on 615 of 616 replies.

**Run 2 (balanced lessons): the lean is fixed.** Same recipe, but the council asked about all 536 injection
questions and the lessons balanced to 153 safe + 153 attack (2,298 in all). Scored against the checks
written beforehand:

| Student | Cyber | Injection | Overall | Attacks caught | Safe recognised |
|---|---|---|---|---|---|
| Base | 63.0% | 49.1% | 60.4% | 57/60 | 0/56 |
| Run 1 | 64.4% | 76.7% | 66.7% | 37/60 | 52/56 |
| **Run 2** | 63.8% | **89.7%** | **68.7%** | **51/60** | 53/56 |

Injection precision is 94% and recall 85% (run 1: 90% and 62%). Claim 2 still holds against the base
(60.4% → 68.7%, p = 0.0002). Cyber stayed flat (p = 0.81), which is the control for run-to-run noise.
Caveats: run 2 vs run 1 overall is +1.9 points and not significant (p = 0.25); the whole difference is
injection. The run changed the mix and the amount of injection data together, so it can't say which
mattered, and each student is one training run.
[Details and every check](docs/labs/02-distill.md#run-2-what-we-got).

## Progress

| Step | Status |
|---|---|
| 1. Scaffold + data | ✅ done 2026-09-26 |
| 2. Baselines per member | ✅ done 2026-09-27 |
| 3. Council score | ✅ done 2026-09-27: council 92.7% < best member 94.3% |
| 4. Build training set | ✅ 2,094 examples from the first 2,500 questions (2026-10-01); enough for the first fine-tune, may grow to ~3,000 later |
| 5. LoRA fine-tune | ✅ done 2026-10-01: Gemma 3 1B, 9.6 min on a free Colab T4 ([notebook](train/finetune.ipynb)) |
| 6. Evaluate the student | ✅ done 2026-10-02: 60.4% → 66.7%, claim 2 supported (p = 0.008); the gain is injection |
| 7. Rebalance injection | ✅ done 2026-10-02: 2,298 lessons (injection 153 safe + 153 attack), retrained and re-scored: injection 76.7% → 89.7%, attacks caught 37 → 51 of 60, cyber flat ([Lab 02, run 2](docs/labs/02-distill.md#run-2-what-we-got)) |

## Cost

Free tiers plus the laptop. The fine-tune ran on a free Colab T4, so no GPU was rented.
Ceiling: **$10 total**.

## Scope, deliberately

In scope: defensive security knowledge, and spotting prompt injection.

Out of scope:
- **Offensive tooling.** The training data stays defensive.
- **Forecasting wars or natural disasters.** Language models can't do that; it needs sensor
  data and time-series models, and is a separate project idea.

## Docs

- [Lab 01: the council](docs/labs/01-council.md), which rebuilds steps 1–3 from zero
- [Lab 02: distillation](docs/labs/02-distill.md), steps 4–6: fine-tune a 1B student and
  score it fairly against its base
- [Journal](docs/journal/)
