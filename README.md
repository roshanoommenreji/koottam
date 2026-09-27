# Koottam

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
questions ─► koottam ─┼─► Llama 3.3 70B     Groq              (server 2)
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

*Not measured yet.* Rows land in `results.csv` (gitignored) and are copied here once a
full run finishes.

| System | Cyber (500) | Injection (116) |
|---|---|---|
| gpt-oss-120b | – | – |
| Llama 3.3 70B | – | – |
| Gemma 4 31B | – | – |
| Gemma 4 E2B (local) | – | – |
| **Council** | – | – |
| Student, base | – | – |
| **Student, fine-tuned** | – | – |

## Progress

| Step | Status |
|---|---|
| 1. Scaffold + data | ✅ done 2026-09-26 |
| 2. Baselines per member | ⏳ needs Cerebras, Groq and Google keys; local member and tie-breaker answering |
| 3. Council score | ⏳ code done, needs step 2 |
| 4. Build training set | ⏳ code done, needs keys |
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
