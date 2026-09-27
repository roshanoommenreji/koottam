# Lab 01: A council of models on different servers

**Goal:** get four models on four servers answering the same security questions, combine
their votes, and measure whether the council is more accurate than its best member.

**Time:** ~1 hour hands-on, then several hours of unattended runs (free-tier rate limits).
**Cost:** $0.

## Concepts first

**Why would a council be smarter than its members?** Only if members make *different*
mistakes. If three models all share a misconception, a vote just repeats it three times.
That is why the council deliberately mixes model families (OpenAI's gpt-oss, Alibaba's
Qwen, Google's Gemma) and why the tie-breaker (NVIDIA's Nemotron) is a family that isn't
on the council at all.

**Why "different servers"?** Nothing about voting needs it, but it is how you would build
this for real. No single provider hosts every model, and spreading load across providers
means one provider's rate limit or outage doesn't stop the run. The cost is that you must
handle each server's own limits, which is most of `client.py`.

**Lesson learned: free tiers limit per day, per account.** The first design put three
roles on OpenRouter. Its key endpoint (`GET /api/v1/key`) showed the free tier allows **50
requests a day**, which would have made scoring the test set alone a 37-day job. On top of
that, its `:free` models share one pool across every free user, and on 2026-09-27 the free
Qwen and Gemma pools were rate-limited for everyone. The fix was one member per provider,
with OpenRouter only breaking ties. Always read a free tier's real limits from its API
before sizing a job.

**Why multiple-choice?** Because it can be scored automatically. "Is the council better?"
is only answerable if "better" is a number.

## 1. Set up

Prerequisites: Python 3.11, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com),
gitleaks, git.

```bash
git clone <this repo> koottam && cd koottam
uv venv -p 3.11 .venv
uv pip install -e ".[dev]"
git config core.hooksPath .githooks
python -m pytest -q                      # 19 offline tests, no network needed
```

API keys. All are free tiers, but **Cerebras needs a card on file** (checked 2026-09-27):
API access stays off until you add a payment method, which then unlocks $5 of signup
credit valid for a month. On its Billing → Pay as you go page, confirm **"Auto-recharge is
off"**. Then requests simply stop at $0 and the card is never charged. Groq and Google AI
Studio need no card; on Google, create the key in a **new project**, because Google's
limits are per project and an existing project's other apps would share them.

| Key | Where | Used for |
|---|---|---|
| `CEREBRAS_API_KEY` | https://cloud.cerebras.ai | gpt-oss-120b |
| `GROQ_API_KEY` | https://console.groq.com/keys | Qwen 3.8 27B |
| `GOOGLE_API_KEY` | https://aistudio.google.com/apikey | Gemma 4 31B |
| `OPENROUTER_API_KEY` | https://openrouter.ai/settings/keys | the tie-breaker |

On OpenRouter, the "Key limit $100" shown next to a new key is a *spending cap*, not a
charge. With zero credits bought nothing can be billed, and `:free` models cost $0.

```bash
cp .env.example .env    # keys go in .env ONLY: .env.example is committed
ollama pull gemma4:e2b  # the local member, ~7 GB
```

## 2. Prepare the data

```bash
python -m koottam prepare
```

Expected output:

```
train: 10150 questions {'cyber': 9614, 'injection': 536}
test: 616 questions {'cyber': 500, 'injection': 116}
train/test overlap: 0 (must be 0)
```

**Lesson learned here:** the first version only removed *exact* duplicates, and one test
question survived in the training pool as a typo'd copy ("wwhat is a common first line of
defense…"). A model trained on it would score higher on that test question for the wrong
reason. `data.py` now also drops any training question with ≥80% word overlap with a test
question. **Always check your split's counts against what you expect.** That is how this was
caught.

## 3. Check every server

```bash
python -m koottam status
```

Every line should say `ok`. `no key` means a `.env` entry is missing. A `404` on a remote
model means the free catalogue changed: find the current id in the provider's model list
and edit `config/teachers.toml`. `student-koottam` fails until Lab 02, which is expected.

## 4. Score each member alone

Smoke-test first. It spends almost none of your quota and catches config mistakes:

```bash
python -m koottam eval --model gpt-oss --limit 20
```

Then do the full test set, one member at a time. Each run can be interrupted and resumed,
because answers are cached in `data/answers/<member>.jsonl`:

```bash
python -m koottam eval --model gpt-oss
python -m koottam eval --model qwen-27b
python -m koottam eval --model gemma-31b
python -m koottam eval --model gemma-local   # ~2 h on a laptop CPU
```

**Why the local model has thinking turned off:** Gemma 4 "thinks" silently before
answering. On a laptop CPU that took ~60 s per question versus ~13 s without it (measured:
420 vs 64 output tokens). `config/teachers.toml` sets `extra = { reasoning_effort = "none" }`.
The same setting goes on both student entries, so the student is always compared under
identical conditions. The setting is part of the cache key, so thinking-on answers are
never reused as thinking-off ones.

## 5. Score the council

```bash
python -m koottam eval --council
```

This makes almost no new calls. The council's votes are the answers already cached in
step 4. Only ties go to the aggregator, and the output tells you how many there were.

Now compare the `council` rows in `results.csv` with the best single member. Either
outcome is a result worth writing down. If the council *doesn't* win, look at which
members agree on wrong answers: that is shared blind spots, and the fix is a more diverse
council, not a bigger one.

## What's next

Lab 02 turns the council's answers on the *training* questions into a fine-tuning set, and
trains a small student model on it.
