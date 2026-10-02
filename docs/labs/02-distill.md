# Lab 02: Distil the council into a 1B student

**Goal:** fine-tune a small model on the council's answers and measure whether it beats the
same model untrained (claim 2).

**Time:** ~30 min hands-on, ~20 min on a free cloud GPU, then ~2 h of unattended scoring on
the laptop. Run 2 adds ~3 h of unattended council calls, ~20 min of GPU and ~1 h of scoring. **Cost:** $0 (free Colab or Kaggle T4).

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
settings cell. The notebook only warns on a mismatch, but the run's `.run.json` records the
count and sha256, so every trained model traces back to one exact file.

## 2. Run the notebook on a free GPU

Open [train/finetune.ipynb](../../train/finetune.ipynb). The committed settings are for
[run 2](#run-2-balance-the-injection-lessons). For a first run, set
`STUDENT_NAME = "koottam-student"` and `EXPORT_BASE = True` in the settings cell. Then use
one of these:

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

The last cell writes `koottam-base.Q8_0.gguf` (first run only), `koottam-student.Q8_0.gguf`
(~1 GB each) and `koottam-student.run.json`. On Colab it copies them to Google Drive under
`MyDrive/koottam/`, because a 1 GB browser download from Colab often fails. On Kaggle
they're in the Output tab.

## 3. Bring the files to the laptop

Put all three files in `train/outputs/` (gitignored, like every `*.gguf`), then copy
`koottam-student.run.json`'s numbers into the journal; don't commit the file itself.

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
python -m koottam eval --model student-base                 # ~1 h on a laptop CPU
python -m koottam eval --model student-koottam              # ~1 h
```

Measured on 2026-10-02: ~6 to 7 s per question for each Q8_0 student on this laptop's CPU,
about an hour per student. Run them one after the other, not in parallel: they share one
CPU, so running both at once just makes each slower. Like every eval, both resume from cache if interrupted, so you can
rerun a command after a crash or a failed call.

## 6. Read the result

```bash
python -m koottam compare student-base student-koottam
```

`results.csv` has each student's score. `compare` says whether the gap is real. Both students
answered the same questions, so it counts the *flips* (questions only one of them got
right) and runs McNemar's exact test on them: if the students were equally good, each flip
would be a coin toss. It also prints a 95% bootstrap interval for the difference, splits
injection by true answer, and splits every question by how many council members got it
right. It reads cached answers only, so it's instant and free.

Claim 2 holds if `student-koottam` beats `student-base` overall and the p-value is small
(below 0.05). We wrote down what to check before seeing the result:

- **Per task.** Injection is only 115 of the 2,094 examples (5.5%). The student may improve on
  cyber and not on injection; if so, that's a data-balance finding, not noise.
- **`no_answer`.** A base model often loses points by not producing a parseable
  `ANSWER:` line at all. Part of what fine-tuning teaches is the format. Report how much
  of the gain is format and how much is knowledge: rescore with only the questions both
  students answered.
- **Against the teachers.** The student won't reach the 31B members. The interesting number
  is how much of the gap from base (1B) to council it closes.

## What we got

Scored 2026-10-02 on all 616 test questions:

| Student | Cyber (500) | Injection (116) | Overall |
|---|---|---|---|
| Base (Gemma 3 1B) | 63.0% | 49.1% | 60.4% |
| **Fine-tuned** (Gemma 3 1B + LoRA) | 64.4% | 76.7% | **66.7%** |

```text
  task          n     student-base  student-koottam  gained  lost  p (McNemar)
  cyber       500            0.630            0.644      71    64  0.6057
  injection   116            0.491            0.767      52    20  0.0002
  all         616            0.604            0.667     123    84  0.0081

  student-koottam minus student-base: +6.3 points (95% interval +1.8 to +11.0)
```

**Claim 2 holds, overall.** The student gained 123 questions and lost 84; a split that
uneven would happen by chance less than 1% of the time. But read the rows, not just the
total:

- **The gain is injection, the opposite of what we expected above.** Cyber moved 1.4
  points, with 71 gained and 64 lost, which is what chance looks like (p = 0.61). Two
  epochs on ~2,000 answers changed how the 1B model *behaves*, not what it *knows*.
- **The base wasn't detecting injection at all.** It called 113 of 116 texts an attack, so
  it caught 57 of 60 attacks and recognised 0 of 56 safe texts. Its 49.1% is about what
  always answering "attack" earns (attacks are 51.7% of the test set). The student recognises 52 of 56 safe texts, but now
  catches only 37 of 60 attacks.
- **The training filter made the student lean "safe".** The first 2,500 training questions
  held 128 injection questions: 78 safe and 50 attacks. The filter kept all 78 safe ones
  but only 37 of the 50 attacks, because the teachers disagree more about attacks. So 68%
  of the injection examples (78 of 115) said "safe", and the student learned that. **An
  agreement filter is not neutral about labels:** it keeps the easy class and drops the
  hard one. For a security detector, missing 23 of 60 attacks is the wrong trade.
- **It learned the council's mistakes too.** On cyber questions all four members got
  right, the student improved (69.5% → 73.3%); where two or fewer were right, it got worse
  (37.1% → 25.7%). A distilled student inherits its teachers' blind spots along with their
  knowledge.
- **None of it is format.** Each student wrote a parseable `ANSWER:` line on 615 of 616
  replies. What changed is the order: the base usually names its answer first ("The correct
  answer is …") and then justifies it; the student always reasons first, like the training
  answers. Its reasoning is often weak (it once called "123456" a strong password).
- **Against the teachers:** it closed 19% of the gap from base to the weighted council (6.3
  of 33.1 points). Gemma 4 E2B, the council's laptop member, scores 83.8%.

The run behind these numbers, from `train/outputs/koottam-student.run.json`: 1,989
training examples and 105 held out, 9.6 min on a Tesla T4, final training loss 1.27,
held-out loss 1.37, and the sha256 of `sft.jsonl` beginning `56096d55`.

## Run 2: balance the injection lessons

Run 1's student leans "safe" because 68% of its injection lessons said "safe". Run 2 keeps
the recipe and fixes the mix.

**What changes:**
- **More injection questions.** The training pool has 536 injection questions, and run 1
  only reached the 128 among its first 2,500 questions. `--all-injection` asks the council
  about all 536.
- **Equal answers.** `--balance` then keeps as many "safe" examples as "attack" ones, dropping
  the latest of the commoner answer. The council answers without seeing any label, and the
  balancing happens afterwards on its agreed answers. Here each kept answer also matches the
  key, so that's the same as balancing on the true label.

**What stays the same:** the base model, every notebook setting, the serving template, the
616 test questions, and the cyber recipe (the first 2,500 questions). One small difference:
the build also retried 62 calls that had failed in the first 2,500 (58 of them Gemma 31B), so
cyber can gain a few examples.

**It changes two things at once:** the mix (68% safe → 50%) and the amount of injection
data. If attack recall comes back, this run can't say which of the two did it. A third run
could separate them: balance only the first 2,500's injection questions, which leaves 37 of
each.

```bash
python -m koottam build --limit 2500 --all-injection --balance   # ~3 h, Cerebras-paced
```

Free servers fail some calls, and failures aren't cached, so run the same command a second time to
retry only those. Everything else comes from the cache.

The build prints how evenly the filter kept each answer, before balancing:

```text
  injection kept per true answer: A …/…  B …/…
```

The notebook is already set for this run: `STUDENT_NAME = "koottam-student-balanced"`,
`EXPORT_BASE = False`, and the file it must use (`EXPECTED_EXAMPLES`, `EXPECTED_SHA256_PREFIX`).
If you upload any other `sft.jsonl`, the data cell stops with an error instead of training on it.
Run it as in step 2 with the new `sft.jsonl`. Download only `koottam-student-balanced.Q8_0.gguf` and its
`.run.json`; the base doesn't change. Then:

```bash
python -m koottam students                         # adds koottam-student-balanced
python -m koottam eval --model student-balanced    # ~1 h
python -m koottam compare student-koottam student-balanced
python -m koottam compare student-base student-balanced
```

**What we'll check** (written down on 2026-10-02, before training):

1. **Attacks caught** rise clearly from run 1's 37 of 60. This is the point of the run.
2. **Safe texts recognised** stay close to run 1's 52 of 56. Giving all of that back would
   just swap one lean for the other: the base caught 57 attacks by calling almost
   everything an attack.
3. **Injection overall** beats run 1's 76.7%. With only 116 questions, only a big change
   will show a small p-value, so we report the flips either way.
4. **Cyber barely moves.** Its recipe is the same, but a different-sized set reshuffles the
   training order and the 5% held-out split, so the cyber difference measures run-to-run
   noise. It's the control.
5. **Claim 2 still holds** against the base.

**What the build gave us** (2026-10-02): the council answered all 536 injection questions, and the filter kept
331 of 333 safe texts but only 153 of 203 attacks. Balancing keeps all 153 attacks and the first 153 safe
texts. The file has **2,298 lessons**: 1,992 cyber and 306 injection (153 + 153), sha256 `2e660996…`. Nine
Gemma 31B calls still failed after a retry, so those questions have three votes, not four.

## Run 2: what we got

Scored 2026-10-02 (616 questions, ~1 h; `python -m koottam compare ...`). The student trained for
10.7 min on 2,183 lessons (115 held out), held-out loss 1.29, and 30 of 40 held-out answers matched the
council. All 616 test replies parsed.

| Student | Cyber (500) | Injection (116) | Overall |
|---|---|---|---|
| Base | 63.0% | 49.1% | 60.4% |
| Run 1 | 64.4% | 76.7% | 66.7% |
| **Run 2 (balanced)** | 63.8% | **89.7%** | **68.7%** |

Injection per true answer: **attacks caught 51 of 60** (run 1: 37), **safe texts recognised 53 of 56**
(run 1: 52). With attack as the positive class, precision is 94% (51 of 54 flags were real) and recall
85%, against run 1's 90% and 62%.

**Against the success checks written before training:**

1. Attacks caught rose clearly from 37/60 to 51/60. **Met.**
2. Safe texts stayed close to 52/56 (now 53/56). **Met.** Nothing was swapped for something else.
3. Injection beat 76.7%: 89.7%, 18 questions gained and 3 lost against run 1 (p = 0.0015). **Met.**
4. Cyber barely moved: 64.4% → 63.8%, 33 gained and 36 lost (p = 0.81). **Met.** That is the
   run-to-run noise, as expected from an unchanged cyber recipe.
5. Claim 2 holds: 60.4% → 68.7% against the base, 117 gained and 66 lost (p = 0.0002; 95% interval
   +4.1 to +12.5).

**What the data does not show:**
- Run 2 vs run 1 overall is +1.9 points, 95% interval −1.0 to +4.9, p = 0.25. The whole difference is
  injection; overall it is not a significant gain over run 1.
- It changed the mix **and** the amount of injection data (306 lessons against 115) at once, so this
  run can't say which of the two did the work. The third run suggested above would separate them.
- One training run each. We have no measured spread for retraining, only the cyber control.
- Cyber's blind spot remains: where two or fewer members were right, the student scores 25.7%
  (the base 37.1%). Distillation still copies the council's mistakes.
- Only 116 injection questions, so an interval on 104 right is wide. Treat 89.7% as "clearly better than 76.7%",
  not as a precise figure.

Where the new lessons helped most: injection questions the council mostly got wrong (two or fewer
right) went from 34.8% to 73.9%, and "3 of 4 right" from 64% to 84%.

## What's next

- **More cyber data.** The training set can grow towards ~3,000 examples, to see whether
  security knowledge moves at all at 1B.
