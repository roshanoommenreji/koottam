"""Every answer a model gives is appended to data/answers/<member>.jsonl and never asked
for again. This is what makes the project affordable on free tiers:

- a run that hits a daily limit resumes tomorrow where it stopped;
- scoring the council costs no new calls, because the council's votes are the same
  answers already collected when each member was scored alone.

The model id is stored with each answer. Change a member's model in the config and its
old answers are ignored rather than silently mixed in."""

from pathlib import Path

from koottam.config import DATA
from koottam.schema import Answer


class AnswerStore:
    def __init__(self, member: str, model: str, root: Path = DATA / "answers") -> None:
        self.member = member
        self.model = model
        self.path = root / f"{member}.jsonl"
        self._cache: dict[str, Answer] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    a = Answer.model_validate_json(line)
                    if a.model == model:
                        self._cache[a.question_id] = a

    def get(self, question_id: str) -> Answer | None:
        return self._cache.get(question_id)

    def put(self, answer: Answer) -> None:
        if answer.error:  # failures are retried next run, never cached
            return
        self._cache[answer.question_id] = answer
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(answer.model_dump_json() + "\n")

    def __len__(self) -> int:
        return len(self._cache)
