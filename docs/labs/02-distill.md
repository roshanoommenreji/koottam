# Lab 02: Distil the council into a 1B student

**Goal:** fine-tune a small model on the council's answers and measure whether it beats the
same model untrained (claim 2).

**Time:** ~30 min hands-on, ~20 min on a free cloud GPU, then ~5 h of unattended scoring on
the laptop. **Cost:** $0 (free Colab or Kaggle T4).

Prerequisite: [Lab 01](01-council.md), through `python -m koottam build`, which wrote
`data/sft.jsonl`.

## Concepts first

**Distillation** means training a small model on a bigger model's outputs instead of on
human labels. Here the "bigger model" is the council. The student sees each training
question together with one council member's 2–4 sentence reasoning and the agreed answer,
and learns to produce both.

**Why filter the examples?** `build` keeps an example only when ≥3 of 4 members agree
*and* the dataset's answer key agrees. The student learns from answers that are right by
both measures. About 7.5% of questions had council agreement on an answer the key called
wrong; those are dropped, not taught.

**Why LoRA?** LoRA freezes the model and trains a small adapter beside some of its weight
matrices (here rank 16, ~1% of the parameters). That's enough to teach a format and a
domain from ~2,000 examples, it fits a free 16 GB GPU, and it overfits less than
updating every weight.

**Why Gemma 3 1B?** It needs to be:
- **Small:** it runs on a laptop CPU, which is the point of distilling.
- **Not on the council:** Gemma 4 E2B is a member, so it would be trained partly on its own
  answers.
- **Weak enough to have room to grow:** stock Gemma 3 1B got 3 of 8 test questions right in
  a smoke test.
- **Plain to fine-tune:** a text-only transformer with no "thinking" mode, and supported by
  Unsloth on a T4. Qwen3.5 0.8B/2B were also candidates, but they need transformers v5 and
  custom kernels that compile slowly on a T4. A deadline is not the time for that.

### The comparison has to be fair

Claim 2 compares two models, so anything that differs between them except training is a
leak. Three things are controlled:

1. **Same export path.** The notebook exports the base model *after* attaching the LoRA
   adapter but *before* training. A fresh adapter is zero (the notebook asserts it), so that
   export is exactly the base weights. It goes through the same merge, converter and Q8_0
   quantisation as the trained one. Using Ollama's stock `gemma3:1b` (Q4) as the base would
   have mixed a quantisation difference into the result.
2. **Same prompt.** Both students get the same system prompt and instruction as every council
   member (`koottam/prompts.py`), at temperature 0.
3. **Same chat template.** This one was nearly missed. See the next section.

### Lesson learned: the serving template must match the training template

Gemma has no system role. Its Hugging Face template, used during training, folds the system
prompt into the first user turn:

```
<start_of_turn>user
{system prompt}

{question}<end_of_turn>
<start_of_turn>model
```

Ollama's stock Gemma 3 template instead sends the system prompt as a **separate** user
turn. A fine-tuned model served that way sees a prompt shape it was never trained on, and
it scores worse for a reason that has nothing to do with what it learned.

`koottam/students.py` therefore serves both students with Koottam's own template. It was
checked on 2026-10-01 by sending the same three questions to the same weights two ways, at
temperature 0:

- as a chat request through the template;
- as a raw request with the training-format prompt built by hand.

The replies were byte-identical 3/3. Through the stock template they differed 3/3.

## 1. Check the training set

```bash
wc -l data/sft.jsonl        # 2094 on 2026-10-01
```

If you rebuilt it with a different `--limit`, update `EXPECTED_EXAMPLES` in the notebook's
settings cell. The notebook only warns on a mismatch, but `run.json` records the count and
sha256, so every trained model traces back to one exact file.

## 2. Run the notebook on a free GPU

Open [train/finetune.ipynb](../../train/finetune.ipynb) in one of these:

- **Colab:** File → Upload notebook. Then Runtime → Change runtime type → **T4 GPU**, and
  Runtime → Run all. When the data cell asks, upload `data/sft.jsonl`.
- **Kaggle:** New notebook → File → Import. Under Settings choose **GPU T4** and turn
  Internet on. Add `sft.jsonl` as a **private** dataset, then Run all.

`sft.jsonl` is never committed or pushed anywhere public. The repo's `.gitignore` keeps all
of `data/` out of git.

What each step prints, and what to look for:

| Cell | Look for |
|---|---|
| data | `2094 examples, sha256 …` |
| export base | `… LoRA B matrices, all zero` before the export starts |
| format | a sample whose first turn starts with the system prompt, then a blank line, then the question |
| labels | only the model's answer, no question text (the loss is on answers only) |
| train | the held-out loss after epoch 2 not clearly *above* epoch 1 (if it is, it's memorising: drop to 1 epoch) |
| sanity | most of 40 held-out answers right, and **all 40** with a parseable `ANSWER:` line |

The last cell writes `koottam-base.Q8_0.gguf`, `koottam-student.Q8_0.gguf` (~1 GB each)
and `run.json`. On Colab it copies them to Google Drive under `MyDrive/koottam/`, because a
1 GB browser download from Colab often fails. On Kaggle they're in the Output tab.

## 3. Bring the files to the laptop

Put all three files in `train/outputs/` (gitignored, like every `*.gguf`), then commit
`run.json`'s numbers into the journal, not the file itself.

## 4. Register both students with Ollama

```bash
python -m koottam students
python -m koottam status        # student-base and student-koottam should say ok
```

`students` writes one Modelfile per GGUF from a single template and runs `ollama create`.
A test (`test_students_are_served_identically`) fails if the two Modelfiles ever differ in
anything but their `FROM` line.

## 5. Score both on the test set

```bash
python -m koottam eval --model student-base --limit 20      # smoke test first
python -m koottam eval --model student-base                 # ~2.5 h on a laptop CPU
python -m koottam eval --model student-koottam              # ~2.5 h
```

Measured on 2026-10-01: ~13 s per question for Gemma 3 1B on this laptop's CPU. Run them
one after the other, not in parallel: they share one CPU, so running both at once just
makes each slower. Like every eval, both resume from cache if interrupted, so you can
rerun a command after a crash or a failed call.

## 6. Read the result

Compare the two rows in `results.csv`. Claim 2 holds if `student-koottam` beats
`student-base` overall. Also check these:

- **Per task.** Injection is only 115 of the 2,094 examples (5.5%). The student may improve on
  cyber and not on injection; if so, that's a data-balance finding, not noise.
- **`no_answer`.** A base model often loses points by not producing a parseable
  `ANSWER:` line at all. Part of what fine-tuning teaches is the format. Report how much
  of the gain is format and how much is knowledge: rescore with only the questions both
  students answered.
- **Against the teachers.** The student won't reach the 31B members. The interesting number
  is how much of the gap from base (1B) to council it closes.

## What we got

*Pending: the notebook run and both evals.*
