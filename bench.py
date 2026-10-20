"""Latency and cost benchmark. Same code path as the live server, no browser.

    python bench.py            # 12 turns
    python bench.py 30         # 30 turns
    CONCIERGE_MODEL=claude-haiku-5-5 python bench.py
    CONCIERGE_PROVIDER=groq python bench.py      # free tier, GROQ_API_KEY

Reports time-to-first-sentence, which is what the guest actually waits for -
time-to-first-token is only when the server starts producing, and nothing is
spoken until a sentence completes.
"""

import asyncio
import statistics
import sys
import time

from server import MODEL, PROVIDER, cost_usd, split_sentences, stream_reply

TURNS = [
    "hi, what time is breakfast?",
    "can I get a late checkout tomorrow?",
    "how much is that?",
    "what's the wifi password?",
    "is the pool open at seven in the morning?",
    "I need a pickup from the airport on Friday",
    "how long does that take?",
    "can you book me a massage for this evening?",
    "room four one two",
    "where do I get towels?",
    "do you have a gym?",
    "thanks, that's all",
]


async def one_turn(text: str) -> dict:
    started = time.perf_counter()
    first_token = first_sentence = None
    buf = ""
    usage = {"in": 0, "out": 0}
    async for chunk in stream_reply([{"role": "user", "content": text}], usage):
        now = time.perf_counter()
        if first_token is None:
            first_token = now - started
        buf += chunk
        sentences, buf = split_sentences(buf)
        if sentences and first_sentence is None:
            first_sentence = now - started
    return {
        "first_token": first_token,
        "first_sentence": first_sentence or (time.perf_counter() - started),
        "total": time.perf_counter() - started,
        "usd": cost_usd(MODEL, usage["in"], usage["out"]),
        "tokens": usage["in"] + usage["out"],
    }


def pct(values: list[float], p: float) -> float:
    return sorted(values)[min(len(values) - 1, int(len(values) * p))]


async def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(TURNS)
    print(f"provider={PROVIDER}  model={MODEL}  turns={n}\n")
    rows = []
    for i in range(n):
        r = await one_turn(TURNS[i % len(TURNS)])
        rows.append(r)
        print(
            f"  {i + 1:>3}  token {r['first_token'] * 1000:>6.0f}ms"
            f"  sentence {r['first_sentence'] * 1000:>6.0f}ms"
            f"  total {r['total'] * 1000:>6.0f}ms  ${r['usd']:.6f}"
        )

    print(f"\n{'metric':<22}{'p50':>10}{'p95':>10}")
    for label, key in [
        ("time to first token", "first_token"),
        ("time to first speech", "first_sentence"),
        ("full turn", "total"),
    ]:
        v = [r[key] for r in rows]
        print(f"{label:<22}{pct(v, 0.5) * 1000:>9.0f}ms{pct(v, 0.95) * 1000:>9.0f}ms")

    usd = [r["usd"] for r in rows]
    print(
        f"\ncost/turn  mean ${statistics.mean(usd):.6f}"
        f"  total ${sum(usd):.4f} over {n} turns"
        f"\ntokens/turn mean {statistics.mean(r['tokens'] for r in rows):.0f}"
        f"\nprojected  ${sum(usd) / n * 1000:.2f} per 1,000 turns"
    )


if __name__ == "__main__":
    asyncio.run(main())
