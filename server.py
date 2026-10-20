"""AI Voice Concierge - streaming voice agent backend.

One WebSocket per caller. Audio capture/playback is done in the browser
(Web Speech API); this process owns the LLM turn, sentence-level streaming,
barge-in cancellation, history trimming, and cost accounting.

Call history lives in Redis, keyed by a call id the browser keeps for the tab,
so a dropped socket reconnects mid-call with its context and any server
instance can serve the next turn.
"""

import asyncio
import json
import os
import re
import time
import uuid
from contextlib import suppress
from functools import lru_cache
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from redis.asyncio import Redis

load_dotenv(Path(__file__).parent / ".env")  # README's `cp .env.example .env` now works

MODEL = os.getenv("CONCIERGE_MODEL", "claude-opus-5")
MAX_TURNS = int(os.getenv("CONCIERGE_MAX_TURNS", "8"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CALL_TTL_S = int(os.getenv("CONCIERGE_CALL_TTL", "1800"))  # idle calls expire

# $ per 1M tokens (input, output). Anthropic list pricing.
PRICES = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.00, 5.00),  # rejects output_config.effort; use haiku-5-5
}

# Voice output is spoken, not read: no markdown, no lists, no long sentences.
SYSTEM = """You are the voice concierge for the Hotel Marisol in Goa. You are \
speaking out loud to a guest over the phone.

Hard rules for speech:
- Two or three short sentences. Never more.
- No markdown, no bullet points, no emoji, no headings.
- Write numbers as a person says them: "seven thirty", "two thousand rupees".
- Ask at most one question per turn.
- If you do not know something, say so and offer to put a note at the desk.

What you know:
- Check-in three PM, check-out eleven AM. Late check-out until two PM costs \
one thousand rupees if rooms are free.
- Breakfast is seven to ten thirty in the Banyan room. Included on all rates.
- The pool is open six AM to nine PM. Towels are at the pool desk.
- Airport pickup is two thousand four hundred rupees, about fifty minutes.
- Spa bookings need four hours notice. Ask for the guest's room number.
- Wifi network "Marisol Guest", password is the guest's room number twice.
Anything outside this list, you do not know. Do not invent prices or hours."""

# Split on sentence end only when followed by space/end, so "3.30" stays whole.
_SENTENCE = re.compile(r"(.+?[.!?])(?=\s|$)", re.S)

app = FastAPI(title="AI Voice Concierge")


@lru_cache(maxsize=1)
def get_client() -> anthropic.AsyncAnthropic:
    """Built on first use so the tests and imports work without a key set."""
    return anthropic.AsyncAnthropic()


def split_sentences(buf: str) -> tuple[list[str], str]:
    """Pull complete sentences out of a streaming buffer.

    Returns (sentences, remainder). Speaking sentence-by-sentence is what keeps
    time-to-first-audio near time-to-first-token instead of near end-of-turn.
    """
    out = []
    pos = 0
    for m in _SENTENCE.finditer(buf):
        out.append(m.group(1).strip())
        pos = m.end()
    return out, buf[pos:]


def trim(history: list[dict]) -> list[dict]:
    """Keep the last MAX_TURNS exchanges. Bounds cost and latency per turn.

    The API needs the first message to be the user's. A barge-in that cut the
    agent before it spoke leaves two user turns in a row, so a plain slice can
    start on an assistant message - drop those.
    """
    out = history[-MAX_TURNS * 2 :]
    while out and out[0]["role"] != "user":
        out = out[1:]
    return out


@lru_cache(maxsize=1)
def get_redis() -> Redis:
    return Redis.from_url(REDIS_URL, decode_responses=True)


def history_key(call_id: str) -> str:
    return f"call:{call_id}:history"


async def load_history(r: Redis, call_id: str) -> list[dict]:
    return [json.loads(m) for m in await r.lrange(history_key(call_id), 0, -1)]


async def remember(r: Redis, call_id: str, history: list[dict], msg: dict) -> None:
    """Append to the in-process list and to Redis in one round trip.

    LTRIM keeps Redis at the same cap trim() sends to the model; EXPIRE drops
    calls nobody has touched for CALL_TTL_S.
    """
    history.append(msg)
    key = history_key(call_id)
    async with r.pipeline(transaction=True) as p:
        p.rpush(key, json.dumps(msg))
        p.ltrim(key, -MAX_TURNS * 2, -1)
        p.expire(key, CALL_TTL_S)
        await p.execute()


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    p_in, p_out = PRICES.get(model, PRICES["claude-opus-5"])
    return (tokens_in * p_in + tokens_out * p_out) / 1_000_000


async def respond(
    sock: WebSocket, r: Redis, call_id: str, history: list[dict], meter: dict, text: str
) -> None:
    """One assistant turn. Cancellable mid-stream - that is the barge-in."""
    await remember(r, call_id, history, {"role": "user", "content": text})
    said = ""
    started = time.perf_counter()
    first_token_ms = None
    try:
        async with get_client().messages.stream(
            model=MODEL,
            max_tokens=300,
            system=SYSTEM,
            messages=trim(history),
            output_config={"effort": "low"},  # voice turns are latency-bound
        ) as stream:
            buf = ""
            async for chunk in stream.text_stream:
                if first_token_ms is None:
                    first_token_ms = round((time.perf_counter() - started) * 1000)
                    await sock.send_json({"type": "first_token", "ms": first_token_ms})
                buf += chunk
                sentences, buf = split_sentences(buf)
                for s in sentences:
                    said += s + " "
                    await sock.send_json({"type": "say", "text": s})
            if buf.strip():
                said += buf
                await sock.send_json({"type": "say", "text": buf.strip()})

            usage = (await stream.get_final_message()).usage
            meter["in"] += usage.input_tokens
            meter["out"] += usage.output_tokens
            meter["turns"] += 1
            await sock.send_json(
                {
                    "type": "done",
                    "first_token_ms": first_token_ms,
                    "turn_ms": round((time.perf_counter() - started) * 1000),
                    "tokens_in": usage.input_tokens,
                    "tokens_out": usage.output_tokens,
                    "session_usd": round(
                        cost_usd(MODEL, meter["in"], meter["out"]), 6
                    ),
                    "usd_per_turn": round(
                        cost_usd(MODEL, meter["in"], meter["out"]) / meter["turns"], 6
                    ),
                }
            )
    except asyncio.CancelledError:
        raise  # barge-in; the partial answer is kept by the finally block
    except Exception as exc:
        # Never strand the caller on "thinking" - the UI has to be told.
        with suppress(Exception):
            await sock.send_json({"type": "error", "text": str(exc)[:200]})
    finally:
        # Keep whatever was actually spoken before the cut, so the next turn's
        # context matches what the guest heard - not what the model planned.
        if said.strip():
            msg = {"role": "assistant", "content": said.strip()}
            with suppress(Exception):  # a Redis blip must not mask the turn's outcome
                await remember(r, call_id, history, msg)


# Call ids come from the browser: accept only a UUID so a client can't pick
# arbitrary Redis keys.
_CALL_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


@app.websocket("/ws")
async def ws(sock: WebSocket, call: str = "") -> None:
    await sock.accept()
    call_id = call if _CALL_ID.fullmatch(call) else str(uuid.uuid4())
    r = get_redis()
    try:
        history = await load_history(r, call_id)
    except Exception as exc:
        await sock.send_json({"type": "error", "text": f"call store unavailable: {exc}"[:200]})
        await sock.close()
        return
    meter = {"in": 0, "out": 0, "turns": 0}
    turn: asyncio.Task | None = None
    try:
        while True:
            msg = await sock.receive_json()
            kind = msg.get("type")
            if kind in ("barge_in", "user_text") and turn and not turn.done():
                turn.cancel()
                # Let the cut turn save what it said before the next turn starts,
                # so history order in Redis matches what the guest heard.
                with suppress(asyncio.CancelledError):
                    await turn
            if kind == "user_text" and msg.get("text", "").strip():
                turn = asyncio.create_task(
                    respond(sock, r, call_id, history, meter, msg["text"].strip())
                )
    except WebSocketDisconnect:
        pass
    finally:
        if turn and not turn.done():
            turn.cancel()


# Absolute so the server runs from any working directory.
STATIC = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
