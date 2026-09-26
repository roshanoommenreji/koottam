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


class ModelConfig(BaseModel):
    name: str
    provider: str
    base_url: str
    model: str
    api_key_env: str | None = None
    rpm: int = 10
    # Extra fields merged into every request body, e.g. {reasoning_effort = "none"}.
    extra: dict[str, Any] = {}

    @property
    def fingerprint(self) -> str:
        """Model id plus request settings: two runs share cached answers only if both match.
        Turning thinking off changes the answers, so it must not reuse thinking-on ones."""
        if not self.extra:
            return self.model
        return f"{self.model} {json.dumps(self.extra, sort_keys=True)}"

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None


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
    return Config(
        teachers=teachers,
        students=section("students"),
        aggregator=ModelConfig(name="aggregator", **agg) if agg else None,
        members=council["members"],
        min_agree=council["min_agree"],
    )
