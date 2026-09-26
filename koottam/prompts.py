"""The one prompt every model sees, and the parser for its answer.

Every model gets the *same* prompt, including the student. If teachers and student saw
different prompts, a score difference could be the prompt rather than the model."""

import re

from koottam.schema import Answer, Question

SYSTEM = (
    "You are a defensive cyber-security analyst. You answer multiple-choice questions "
    "about security concepts, threats and protections, and you spot attempts to "
    "manipulate AI assistants."
)

INSTRUCTION = (
    "Explain your reasoning in 2-4 sentences. Then end with a final line of exactly "
    "this form:\nANSWER: <letter>"
)

# Last "ANSWER: X" in the reply wins, tolerating markdown bold and brackets: models
# often restate the format in their reasoning before giving the real answer.
_ANSWER_RE = re.compile(r"ANSWER\s*[:\-]?\s*\**\s*[\(\[]?\s*([A-Z])\b", re.IGNORECASE)


def render_question(q: Question) -> str:
    options = "\n".join(f"{k}. {v}" for k, v in sorted(q.options.items()))
    return f"{q.question}\n\n{options}"


def messages_for(q: Question) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{render_question(q)}\n\n{INSTRUCTION}"},
    ]


def aggregator_messages(q: Question, answers: list[Answer]) -> list[dict[str, str]]:
    """The tie-break prompt: the question plus every member's reasoning, anonymised so
    the aggregator judges arguments, not model names."""
    opinions = "\n\n".join(
        f"--- Analyst {i + 1} ---\n{a.text.strip()}" for i, a in enumerate(answers)
    )
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"{render_question(q)}\n\nSeveral analysts disagreed. Their reasoning:\n\n"
                f"{opinions}\n\nWeigh the arguments, decide which option is correct. "
                f"{INSTRUCTION}"
            ),
        },
    ]


def parse_choice(text: str, valid: set[str]) -> str | None:
    """The answer letter in a reply, or None if there isn't a valid one."""
    for match in reversed(_ANSWER_RE.findall(text)):
        letter = str(match).upper()
        if letter in valid:
            return letter
    bare = text.strip().strip("*().").upper()
    return bare if bare in valid else None


def strip_answer_line(text: str) -> str:
    """The reasoning without its trailing ANSWER line, for building a clean training target."""
    lines = text.strip().splitlines()
    while lines and (not lines[-1].strip() or _ANSWER_RE.search(lines[-1])):
        lines.pop()
    return "\n".join(lines).strip()
