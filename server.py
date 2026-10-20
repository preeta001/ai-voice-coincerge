"""AI Voice Concierge - streaming voice agent backend.

One WebSocket per caller. Audio capture/playback is done in the browser
(Web Speech API); this process owns the LLM turn, sentence-level streaming,
barge-in cancellation, history trimming, and cost accounting.

Two stores, two jobs:
- Redis holds the live call (capped history, expires), keyed by a call id the
  browser keeps for the tab, so a dropped socket reconnects mid-call with its
  context and any server instance can serve the next turn.
- MongoDB keeps the permanent record: one document per call with the full,
  uncapped transcript, which replies were cut off, and token totals. It is
  written after each turn is spoken, so it never delays speech.
"""

import asyncio
import json
import logging
import os
import re
import time
import uuid
from contextlib import suppress
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from redis.asyncio import Redis

load_dotenv(Path(__file__).parent / ".env")  # README's `cp .env.example .env` now works

# "anthropic" (Claude) or "groq" (open models on Groq; has a free tier).
PROVIDER = os.getenv("CONCIERGE_PROVIDER", "anthropic")
MODEL = os.getenv(
    # qwen3.8-27b had the lowest time to first speech of Groq's chat models (bench.py)
    "CONCIERGE_MODEL", "qwen/qwen3.8-27b" if PROVIDER == "groq" else "claude-opus-5"
)
MAX_TURNS = int(os.getenv("CONCIERGE_MAX_TURNS", "8"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CALL_TTL_S = int(os.getenv("CONCIERGE_CALL_TTL", "1800"))  # idle calls expire
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")

log = logging.getLogger("concierge")

# $ per 1M tokens (input, output), list pricing. Groq's free tier bills $0;
# its rows show what the same traffic would cost on the paid tier.
PRICES = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.00, 5.00),  # rejects output_config.effort; use haiku-5-5
    "openai/gpt-oss-120b": (0.15, 0.60),  # console.groq.com/docs/models, Oct 2026
    "openai/gpt-oss-20b": (0.075, 0.30),
    "qwen/qwen3.8-27b": (0.80, 4.00),
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


@lru_cache(maxsize=1)
def get_groq():
    from groq import AsyncGroq  # imported only when PROVIDER == "groq"

    return AsyncGroq()


async def stream_reply(messages: list[dict], usage: dict):
    """Yield one reply's text as it streams; set usage["in"]/["out"] at the end.

    The only provider-specific code. The server and bench.py both go through
    here, so the benchmark measures exactly what a caller hears.
    """
    if PROVIDER == "groq":
        # Groq's chat models reason before answering. Keep that short (latency)
        # and out of the text stream, or the guest hears the model thinking.
        if MODEL.startswith("openai/gpt-oss"):
            extra = {"reasoning_effort": "low", "include_reasoning": False}
        elif MODEL.startswith("qwen/"):
            extra = {"reasoning_format": "hidden"}
        else:
            extra = {}
        stream = await get_groq().chat.completions.create(
            model=MODEL,
            max_tokens=300,
            messages=[{"role": "system", "content": SYSTEM}, *messages],
            stream=True,
            **extra,
        )
        async with stream:  # closes the HTTP stream if barge-in cancels us
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                u = chunk.usage or (chunk.x_groq and chunk.x_groq.usage)
                if u:
                    usage["in"], usage["out"] = u.prompt_tokens, u.completion_tokens
        return
    async with get_client().messages.stream(
        model=MODEL,
        max_tokens=300,
        system=SYSTEM,
        messages=messages,
        output_config={"effort": "low"},  # voice turns are latency-bound
    ) as stream:
        async for chunk in stream.text_stream:
            yield chunk
        u = (await stream.get_final_message()).usage
        usage["in"], usage["out"] = u.input_tokens, u.output_tokens


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


@lru_cache(maxsize=1)
def get_transcripts():
    """Mongo collection `concierge.calls`: one document per call, _id = call id."""
    from pymongo import AsyncMongoClient

    # Writes go through archive_writer's queue, off the speaking path, so a
    # cold Atlas connection (~2.6 s from India) or a dead server costs the
    # caller nothing; 5 s just bounds how long one write may hang.
    return AsyncMongoClient(MONGO_URL, serverSelectionTimeoutMS=5000).concierge.calls


def transcript_update(user: str, assistant: str, cut: bool, usage: dict) -> dict:
    """One finished turn as a Mongo upsert: append its messages, add its tokens."""
    turn = [{"role": "user", "content": user}]
    if assistant:
        turn.append({"role": "assistant", "content": assistant, "cut": cut})
    now = datetime.now(timezone.utc)
    return {
        "$push": {"turns": {"$each": turn}},
        "$inc": {"tokens_in": usage["in"], "tokens_out": usage["out"]},
        "$setOnInsert": {"started_at": now, "provider": PROVIDER, "model": MODEL},
        "$set": {"updated_at": now},
    }


async def archive_turn(calls, call_id: str, update: dict) -> None:
    """Upsert into the call's transcript. A record, not a dependency: a live
    call never waits long on it or fails because of it."""
    try:
        await calls.update_one({"_id": call_id}, update, upsert=True)
    except Exception as exc:
        log.warning("transcript not saved for call %s: %r", call_id, exc)


async def archive_writer(calls, call_id: str, queue: asyncio.Queue) -> None:
    """One per connection: writes its turns to Mongo in order until None.

    Turns only enqueue (instant, can't be interrupted), so barge-in - which
    cancels turns - can never cancel a write or reorder the transcript.
    """
    while (update := await queue.get()) is not None:
        await archive_turn(calls, call_id, update)


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    p_in, p_out = PRICES.get(model, PRICES["claude-opus-5"])
    return (tokens_in * p_in + tokens_out * p_out) / 1_000_000


async def save_reply(lock: asyncio.Lock, r: Redis, call_id: str, history: list[dict], msg: dict):
    async with lock:
        with suppress(Exception):  # a Redis blip must not mask the turn's outcome
            await remember(r, call_id, history, msg)


async def respond(
    sock: WebSocket,
    r: Redis,
    lock: asyncio.Lock,
    archive: asyncio.Queue,
    call_id: str,
    history: list[dict],
    meter: dict,
    text: str,
) -> None:
    """One assistant turn. Cancellable mid-stream - that is the barge-in."""
    async with lock:  # FIFO: this turn's writes queue behind the last turn's
        await remember(r, call_id, history, {"role": "user", "content": text})
    said = ""
    cut = False
    usage = {"in": 0, "out": 0}
    started = time.perf_counter()
    first_token_ms = None
    try:
        buf = ""
        async for chunk in stream_reply(trim(history), usage):
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

        meter["in"] += usage["in"]
        meter["out"] += usage["out"]
        meter["turns"] += 1
        await sock.send_json(
            {
                "type": "done",
                "first_token_ms": first_token_ms,
                "turn_ms": round((time.perf_counter() - started) * 1000),
                "tokens_in": usage["in"],
                "tokens_out": usage["out"],
                "session_usd": round(cost_usd(MODEL, meter["in"], meter["out"]), 6),
                "usd_per_turn": round(
                    cost_usd(MODEL, meter["in"], meter["out"]) / meter["turns"], 6
                ),
            }
        )
    except asyncio.CancelledError:
        cut = True
        raise  # barge-in; the partial answer is kept by the finally block
    except Exception as exc:
        # Never strand the caller on "thinking" - the UI has to be told.
        with suppress(Exception):
            await sock.send_json({"type": "error", "text": str(exc)[:200]})
    finally:
        # A cancel can land here too: barge-in or hang-up just after the reply
        # finished. So the bookkeeping must not be interruptible.
        # 1) The permanent record, queued synchronously - nothing can cut it.
        archive.put_nowait(transcript_update(text, said.strip(), cut, usage))
        # 2) Keep whatever was actually spoken, so the next turn's context matches
        #    what the guest heard. Shielded: a cancel can't abort it halfway, and
        #    the lock keeps the next turn's write behind this one in Redis.
        if said.strip():
            msg = {"role": "assistant", "content": said.strip()}
            await asyncio.shield(save_reply(lock, r, call_id, history, msg))


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
    lock = asyncio.Lock()
    archive: asyncio.Queue = asyncio.Queue()
    writer = asyncio.create_task(archive_writer(get_transcripts(), call_id, archive))
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
                    respond(sock, r, lock, archive, call_id, history, meter, msg["text"].strip())
                )
    except WebSocketDisconnect:
        pass
    finally:
        if turn and not turn.done():
            turn.cancel()
            with suppress(asyncio.CancelledError):
                await turn  # its last words reach the queue before we close it
        archive.put_nowait(None)
        await writer  # drain: the call's transcript is complete when we return


# Absolute so the server runs from any working directory.
STATIC = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
