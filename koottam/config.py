"""Loads config/teachers.toml and the API keys in .env."""

import json
import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "teachers.toml"
DATA = ROOT / "data"
DEFAULT_MAX_TOKENS = 2048


class ModelConfig(BaseModel):
    name: str
    provider: str
    base_url: str
    model: str
    api_key_env: str | None = None
    rpm: float = 10  # fractional allowed: Cerebras' 150/hour cap is 2.5/min
    # Extra fields merged into every request body, e.g. {reasoning_effort = "none"}.
    extra: dict[str, Any] = {}
    # Output budget. Thinking counts against it: a model that thinks past it returns an
    # empty answer (Gemma 4 31B did at 2048 on a maths-style question).
    max_tokens: int = DEFAULT_MAX_TOKENS

    @property
    def fingerprint(self) -> str:
        """Model id plus request settings: two runs share cached answers only if both match.
        Turning thinking off changes the answers, so it must not reuse thinking-on ones."""
        settings = dict(self.extra)
        if self.max_tokens != DEFAULT_MAX_TOKENS:
            settings["max_tokens"] = self.max_tokens
        if not settings:
            return self.model
        return f"{self.model} {json.dumps(settings, sort_keys=True)}"

    @property
    def api_key(self) -> str | None:
        """The key, or None if unset or still the .env.example placeholder."""
        key = os.environ.get(self.api_key_env) if self.api_key_env else None
        return None if not key or key == "replace-me" else key


class Config(BaseModel):
    teachers: dict[str, ModelConfig]
    students: dict[str, ModelConfig]
    aggregator: ModelConfig | None
    members: list[str]
    min_agree: int

    def model(self, name: str) -> ModelConfig:
        """Any configured model by name: a teacher, a student, or 'aggregator'."""
        if name == "aggregator" and self.aggregator:
            return self.aggregator
        found = self.teachers.get(name) or self.students.get(name)
        if not found:
            known = [*self.teachers, *self.students, "aggregator"]
            raise SystemExit(f"unknown model {name!r}; configured: {', '.join(known)}")
        return found


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Tiny .env reader (KEY=value lines) so we don't need python-dotenv. Real environment
    variables win over the file."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def load(path: Path = CONFIG_PATH) -> Config:
    load_dotenv()
    raw = tomllib.loads(path.read_text(encoding="utf-8"))

    def section(name: str) -> dict[str, ModelConfig]:
        return {k: ModelConfig(name=k, **v) for k, v in raw.get(name, {}).items()}

    teachers = section("teachers")
    agg = raw.get("aggregator")
    council = raw["council"]
    missing = [m for m in council["members"] if m not in teachers]
    if missing:
        raise SystemExit(f"council members not defined under [teachers]: {missing}")
    # A kept example must be a strict majority. `build` also relies on this to skip
    # tie-breaks: a tie can never be a strict majority.
    if council["min_agree"] * 2 <= len(council["members"]):
        raise SystemExit("council.min_agree must be more than half the members")
    return Config(
        teachers=teachers,
        students=section("students"),
        aggregator=ModelConfig(name="aggregator", **agg) if agg else None,
        members=council["members"],
        min_agree=council["min_agree"],
    )
