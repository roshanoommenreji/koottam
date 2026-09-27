"""The council: ask several models on different servers, then combine their votes.

How the combination works (the "Mixture-of-Agents, lite" pattern):

1. Every member answers independently, in parallel, each on its own server.
2. A clear plurality wins outright: no extra call.
3. On a tie, the aggregator (a model family that is not on the council) reads every
   member's reasoning, anonymised, and decides.

Full Mixture-of-Agents calls the aggregator on every question. Calling it only on ties
gets most of the benefit for a fraction of the free-tier quota, and it keeps the vote
itself honest: we can report how often the aggregator had to step in.
"""

import asyncio
from collections import Counter
from collections.abc import Iterable

import httpx

from koottam.client import ChatClient
from koottam.config import Config, ModelConfig
from koottam.prompts import aggregator_messages, messages_for, parse_choice
from koottam.schema import Answer, Question, Verdict
from koottam.store import AnswerStore


def vote(votes: dict[str, str | None]) -> tuple[str | None, int]:
    """(winning choice, number of votes for it). Choice is None when nobody answered or
    the top choices are tied; the count is then the tied top count."""
    counts = Counter(v for v in votes.values() if v is not None)
    if not counts:
        return None, 0
    ranked = counts.most_common()
    top, n = ranked[0]
    if len(ranked) > 1 and ranked[1][1] == n:
        return None, n
    return top, n


class Member:
    """One model plus its cache: asks only what it hasn't answered before."""

    def __init__(self, cfg: ModelConfig, http: httpx.AsyncClient) -> None:
        self.cfg = cfg
        self.client = ChatClient(cfg, http)
        self.store = AnswerStore(cfg.name, cfg.fingerprint)

    async def answer(self, q: Question, messages: list[dict[str, str]] | None = None) -> Answer:
        cached = self.store.get(q.id)
        if cached:
            return cached
        try:
            text = await self.client.chat(messages or messages_for(q), self.cfg.max_tokens)
            if not text:
                # Usually the model thought past max_tokens. Treat as a failed call, so
                # it is retried (e.g. after raising max_tokens) instead of cached forever.
                raise ValueError("empty reply (thinking may have used up max_tokens)")
            a = Answer(
                member=self.cfg.name,
                model=self.cfg.fingerprint,
                question_id=q.id,
                choice=parse_choice(text, set(q.options)),
                text=text,
            )
        except Exception as e:  # noqa: BLE001 - one failed call must not stop the run
            a = Answer(
                member=self.cfg.name,
                model=self.cfg.fingerprint,
                question_id=q.id,
                choice=None,
                text="",
                error=f"{type(e).__name__}: {e}"[:300],
            )
        self.store.put(a)
        return a


async def ask_all(
    member: Member, questions: list[Question], label: str | None = None
) -> dict[str, Answer]:
    """Every question to one member, concurrently; the rate limiter does the pacing."""
    done = 0
    total = len(questions)
    label = label or member.cfg.name

    async def one(q: Question) -> Answer:
        nonlocal done
        a = await member.answer(q)
        done += 1
        if done % 25 == 0 or done == total:
            print(f"  {label}: {done}/{total}", flush=True)
        return a

    answers = await asyncio.gather(*(one(q) for q in questions))
    return {a.question_id: a for a in answers}


class Council:
    def __init__(self, config: Config, http: httpx.AsyncClient) -> None:
        self.config = config
        self.members = [Member(config.teachers[name], http) for name in config.members]
        self.aggregator = Member(config.aggregator, http) if config.aggregator else None

    async def collect(self, questions: list[Question]) -> dict[str, dict[str, Answer]]:
        """member name -> question id -> answer. Members run side by side, one per server."""
        results = await asyncio.gather(*(ask_all(m, questions) for m in self.members))
        return {m.cfg.name: r for m, r in zip(self.members, results, strict=True)}

    async def decide(
        self, questions: list[Question], use_aggregator: bool = True
    ) -> list[Verdict]:
        answers = await self.collect(questions)
        by_id = {q.id: q for q in questions}
        verdicts: list[Verdict] = []
        ties: list[tuple[Question, list[Answer], dict[str, str | None]]] = []
        for q in questions:
            votes = {name: answers[name][q.id].choice for name in answers}
            choice, n = vote(votes)
            if choice is not None:
                verdicts.append(Verdict(
                    question_id=q.id, choice=choice, votes=votes, agree=n, decided_by="vote"
                ))
            else:
                ties.append((q, [a[q.id] for a in answers.values() if a[q.id].text], votes))

        if ties and self.aggregator and use_aggregator:
            agg = self.aggregator
            print(f"  aggregator: breaking {len(ties)} ties", flush=True)
            results = await asyncio.gather(
                *(agg.answer(q, aggregator_messages(q, opinions)) for q, opinions, _ in ties)
            )
            for (q, _, votes), a in zip(ties, results, strict=True):
                agree = sum(v == a.choice for v in votes.values()) if a.choice else 0
                verdicts.append(Verdict(
                    question_id=q.id,
                    choice=a.choice,
                    votes=votes,
                    agree=agree,
                    decided_by="aggregator" if a.choice else "none",
                ))
        else:
            for q, _, votes in ties:
                verdicts.append(Verdict(
                    question_id=q.id, choice=None, votes=votes, agree=0, decided_by="none"
                ))

        order = {qid: i for i, qid in enumerate(by_id)}
        return sorted(verdicts, key=lambda v: order[v.question_id])


def errors(answers: Iterable[Answer]) -> list[str]:
    return [a.error for a in answers if a.error]
