"""AI Voice Concierge - streaming voice agent backend.

One WebSocket per caller. Audio capture/playback is done in the browser
(Web Speech API); this process owns the LLM turn, sentence-level streaming,
barge-in cancellation, history trimming, and cost accounting.
"""

import asyncio
import os
import re
import time
from functools import lru_cache

import anthropic
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

MODEL = os.getenv("CONCIERGE_MODEL", "claude-opus-5")
MAX_TURNS = int(os.getenv("CONCIERGE_MAX_TURNS", "8"))

# $ per 1M tokens (input, output). Anthropic list pricing.
PRICES = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
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
    """Keep the last MAX_TURNS exchanges. Bounds cost and latency per turn."""
    return history[-MAX_TURNS * 2 :]


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    p_in, p_out = PRICES.get(model, PRICES["claude-opus-5"])
    return (tokens_in * p_in + tokens_out * p_out) / 1_000_000


async def respond(sock: WebSocket, history: list[dict], meter: dict, text: str) -> None:
    """One assistant turn. Cancellable mid-stream - that is the barge-in."""
    history.append({"role": "user", "content": text})
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
    finally:
        # Keep whatever was actually spoken before the cut, so the next turn's
        # context matches what the guest heard - not what the model planned.
        if said.strip():
            history.append({"role": "assistant", "content": said.strip()})


@app.websocket("/ws")
async def ws(sock: WebSocket) -> None:
    await sock.accept()
    history: list[dict] = []
    meter = {"in": 0, "out": 0, "turns": 0}
    turn: asyncio.Task | None = None
    try:
        while True:
            msg = await sock.receive_json()
            kind = msg.get("type")
            if kind in ("barge_in", "user_text") and turn and not turn.done():
                turn.cancel()
            if kind == "user_text" and msg.get("text", "").strip():
                turn = asyncio.create_task(
                    respond(sock, history, meter, msg["text"].strip())
                )
    except WebSocketDisconnect:
        pass
    finally:
        if turn and not turn.done():
            turn.cancel()


app.mount("/", StaticFiles(directory="static", html=True), name="static")
