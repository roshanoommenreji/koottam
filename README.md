# Koottam

[![ci](https://github.com/roshanoommenreji/koottam/actions/workflows/ci.yml/badge.svg)](https://github.com/roshanoommenreji/koottam/actions/workflows/ci.yml)
[![licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)

**Showcase page: [roshanoommenreji.github.io/koottam](https://roshanoommenreji.github.io/koottam/)**

> **Status:** claim 1 is measured (the council does *not* beat its best member: 93.5% vs
> 94.3%, see [Results](#results)). Claim 2 is in progress: the training set is being built
> and the fine-tune is next.

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
python -m koottam build --limit 3000     # training set from the council
```

## Results

Full held-out test set, 2026-09-27. Every member answered all 616 questions.

| System | Cyber (500) | Injection (116) | Overall |
|---|---|---|---|
| gpt-oss-120b (Cerebras) | 93.6% | 75.9% | 90.3% |
| Qwen 3.8 27B (Groq) | 94.2% | 81.0% | 91.7% |
| **Gemma 4 31B (Google)** | **96.2%** | **86.2%** | **94.3%** |
| Gemma 4 E2B (laptop) | 84.4% | 81.0% | 83.8% |
| **Council**, equal votes (tie-breaker on 19 ties) | 95.2% | 82.8% | **92.9%** |
| **Council**, weighted votes (learned on train) | 95.6% | 84.5% | **93.5%** |
| Student, base | – | – | – |
| **Student, fine-tuned** | – | – | – |

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

## Progress

| Step | Status |
|---|---|
| 1. Scaffold + data | ✅ done 2026-09-26 |
| 2. Baselines per member | ✅ done 2026-09-27 |
| 3. Council score | ✅ done 2026-09-27: council 92.7% < best member 94.3% |
| 4. Build training set | ✅ 2,094 examples from the first 2,500 questions (2026-10-01); enough for the first fine-tune, may grow to ~3,000 later |
| 5. LoRA fine-tune | ☐ notebook not written; student model not chosen |
| 6. Evaluate the student | ☐ |

## Cost

Free tiers plus the laptop. The only possible spend is one rented-GPU fine-tune (~$1–3) if
the free Kaggle/Colab GPUs aren't enough. Ceiling: **$10 total**.

## Scope, deliberately

In scope: defensive security knowledge, and spotting prompt injection.

Out of scope:
- **Offensive tooling.** The training data stays defensive.
- **Forecasting wars or natural disasters.** Language models can't do that; it needs sensor
  data and time-series models, and is a separate project idea.

## Docs

- [Lab 01: the council](docs/labs/01-council.md), which rebuilds steps 1–3 from zero
- [Journal](docs/journal/)
