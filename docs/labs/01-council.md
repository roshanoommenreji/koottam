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
python -m pytest -q                      # 24 offline tests, no network needed
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
and edit `config/teachers.toml`. `student-base` and `student-koottam` fail until Lab 02
creates them, which is expected.

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

[Lab 02](02-distill.md) trains a small student model (Gemma 3 1B) on the fine-tuning set
that `build` makes from the council's answers on the *training* questions.

## What we got (2026-09-27)

| System | Overall (616) |
|---|---|
| gpt-oss-120b | 90.3% |
| Qwen 3.8 27B | 91.7% |
| Gemma 4 31B | **94.3%** |
| Gemma 4 E2B (laptop) | 83.8% |
| Council | 92.7% |

The council beat the average member but **not the best one**. The saved answers explain
why. At least one member was right on 98.7% of questions, but gpt-oss and Qwen often shared
the same wrong answer (32 times), and equal votes let them outvote Gemma 31B, which was
alone and right on 9 questions. Diversity is necessary but not sufficient: **how votes are
weighted matters as much as who votes.** That is the next experiment, with weights learned
on training questions, never on the test set.

Lessons from running it:

- **Cap requests in flight, not just their rate.** All 616 questions were sent to the laptop
  at once; it works one at a time, so the queue's tail waited past the 3-minute HTTP timeout
  and failed. `concurrency = 1` for Ollama fixed it (~12 s per answer).
- **Always run a retry pass.** Google returned intermittent 500s and Groq some failures (58
  calls in total). None were cached, and a second `eval --council` filled every gap.
- Wall-clock time: 4 h 44 min, set by Cerebras' 150 requests/hour.

## Weighted votes (2026-09-28)

```bash
python -m koottam build --limit 500     # members answer 500 TRAIN questions (cached)
python -m koottam weights               # accuracy per member per task -> data/weights.json
python -m koottam eval --council --weighted
```

Each vote counts `log(p(k−1)/(1−p))`, where p is the member's accuracy on training questions
and k the number of options: the optimal rule for independent voters (Nitzan & Paroush). A
member at chance gets 0; each halving of its error rate adds a constant amount.

Result: 93.5%, against 92.9% for equal votes and 94.3% for the best member. It's better, but
still not a win. The weights were learned without touching the test set. If you tune them
until the test score beats 94.3%, you've learned the answer key, not a better council.
