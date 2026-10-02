"""`python -m koottam students`: register the two exported students with the local Ollama.

The notebook (train/finetune.ipynb) exports the base and the fine-tuned student as GGUF
files through one pipeline. This serves them through one pipeline too: both get the same
chat template and the same stop token, so a score difference can only come from training.

The template is Koottam's own, not Ollama's stock Gemma 3 one. Ollama's stock template
sends the system prompt as a separate user turn; the Hugging Face template used in
training folds it into the first user turn as `system + "\\n\\n" + question`. A model served
in a shape it was never trained on underperforms for a reason that has nothing to do with
what it learned. Checked 2026-10-01: at temperature 0, a chat request through this
template gives the same reply as a raw request with the training-format prompt.
"""

import subprocess
from pathlib import Path

from koottam.config import ROOT, Config

OUTPUTS = ROOT / "train" / "outputs"
QUANT = "Q8_0"

# Gemma 3's training format, as rendered by its Hugging Face chat template:
#   <start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n<start_of_turn>model\n
# Newlines are Go string literals so whitespace trimming can't eat them.
TEMPLATE = (
    "{{- $open := false }}\n"
    "{{- range $i, $_ := .Messages }}\n"
    "{{- $last := eq (len (slice $.Messages $i)) 1 }}\n"
    '{{- if eq .Role "system" }}{{ "<start_of_turn>user\\n" }}{{ .Content }}{{ "\\n\\n" }}'
    "{{ $open = true }}\n"
    '{{- else if eq .Role "user" }}{{ if not $open }}{{ "<start_of_turn>user\\n" }}{{ end }}'
    '{{ .Content }}{{ "<end_of_turn>\\n" }}{{ $open = false }}'
    '{{ if $last }}{{ "<start_of_turn>model\\n" }}{{ end }}\n'
    '{{- else if eq .Role "assistant" }}{{ "<start_of_turn>model\\n" }}{{ .Content }}'
    '{{ if not $last }}{{ "<end_of_turn>\\n" }}{{ end }}\n'
    "{{- end }}\n"
    "{{- end }}"
)


def modelfile(source: str) -> str:
    """A Modelfile for one student. `source` is a GGUF path or an existing Ollama model."""
    return (
        f"FROM {source}\n"
        f'TEMPLATE """{TEMPLATE}"""\n'
        "PARAMETER stop <end_of_turn>\n"
        "PARAMETER temperature 0\n"
    )


def gguf_for(ollama_name: str) -> Path:
    return OUTPUTS / f"{ollama_name}.{QUANT}.gguf"


def serve(config: Config) -> None:
    missing = []
    for name in config.students:
        model = config.model(name).model
        gguf = gguf_for(model)
        if not gguf.exists():
            missing.append(gguf.name)
            continue
        mf = OUTPUTS / f"{model}.Modelfile"
        mf.write_text(modelfile(str(gguf.resolve())), encoding="utf-8")
        print(f"  ollama create {model}  ({gguf.name}, {gguf.stat().st_size / 1e9:.2f} GB)")
        subprocess.run(["ollama", "create", model, "-f", str(mf)], check=True)
    if missing:
        # Each notebook run adds one file, so a missing one is a run not done yet, not an error.
        print(f"  skipped, not in {OUTPUTS} yet: {', '.join(missing)}"
              " (download it from the notebook run, Lab 02 step 3)")
