"""The shapes every stage passes around. Every model output is parsed into one of these
before it is written anywhere, so a malformed reply fails loudly instead of silently
poisoning the training set."""

from typing import Literal

from pydantic import BaseModel, Field

Task = Literal["cyber", "injection"]


class Question(BaseModel):
    """One multiple-choice item. Both tasks share this shape: prompt-injection detection
    is just a two-option question (A = safe, B = injection)."""

    id: str
    task: Task
    source: str
    question: str
    options: dict[str, str]
    answer: str


class Answer(BaseModel):
    """What one model said about one question."""

    member: str
    model: str
    question_id: str
    choice: str | None  # parsed letter, None if the reply had no parseable answer
    text: str
    error: str | None = None  # set when the call itself failed; such answers are never cached


class Verdict(BaseModel):
    """What the council decided about one question."""

    question_id: str
    choice: str | None
    votes: dict[str, str | None]
    agree: int = Field(description="how many members voted for `choice`")
    decided_by: Literal["vote", "aggregator", "none"]
