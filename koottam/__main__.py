"""python -m koottam <command>

  prepare                      download datasets, write data/train.jsonl + data/test.jsonl
  status                       which keys are set, which servers answer, what is cached
  eval --model NAME [--limit]  score one model on the test set  -> results.csv
  eval --council [--limit]     score the council (reuses members' cached answers)
  build [--limit N]            council answers on train -> data/sft.jsonl
"""

import argparse
import asyncio

import httpx

from koottam import config as cfg
from koottam import data


async def status(config: cfg.Config) -> None:
    from koottam.client import ChatClient
    from koottam.store import AnswerStore

    probe = [{"role": "user", "content": "Reply with the single word: ok"}]
    names = [*config.members, *config.students, *(["aggregator"] if config.aggregator else [])]
    async with httpx.AsyncClient(timeout=60) as http:
        for name in names:
            m = config.model(name)
            if m.api_key_env and not m.api_key:
                state = f"no key ({m.api_key_env} unset)"
            else:
                try:
                    reply = await ChatClient(m, http).chat(probe, max_tokens=256)
                    state = f"ok: {reply.strip()[:30]!r}"
                except Exception as e:  # noqa: BLE001
                    state = f"FAIL {type(e).__name__}: {str(e)[:80]}"
            cached = len(AnswerStore(name, m.fingerprint))
            print(f"  {name:<16} {m.provider:<11} {m.model:<42} cached={cached:<6} {state}")


def main() -> None:
    p = argparse.ArgumentParser(prog="koottam", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    sub.add_parser("status")
    e = sub.add_parser("eval")
    who = e.add_mutually_exclusive_group(required=True)
    who.add_argument("--model")
    who.add_argument("--council", action="store_true")
    e.add_argument("--limit", type=int)
    b = sub.add_parser("build")
    b.add_argument("--limit", type=int)
    args = p.parse_args()

    if args.cmd == "prepare":
        data.prepare()
        return
    config = cfg.load()
    if args.cmd == "status":
        asyncio.run(status(config))
    elif args.cmd == "eval":
        from koottam import evaluate

        if args.council:
            asyncio.run(evaluate.eval_council(config, args.limit))
        else:
            asyncio.run(evaluate.eval_model(config, args.model, args.limit))
    elif args.cmd == "build":
        from koottam.build import build

        asyncio.run(build(config, args.limit))


if __name__ == "__main__":
    main()
