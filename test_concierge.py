"""Self-check for the turn logic. No API key, no network, no Redis server.

    pip install fakeredis   # in-memory Redis for the call-store tests
    python test_concierge.py
"""

import asyncio

import fakeredis

from server import (
    CALL_TTL_S,
    MAX_TURNS,
    archive_turn,
    cost_usd,
    history_key,
    load_history,
    remember,
    split_sentences,
    transcript_update,
    trim,
)


def test_split_holds_partial_sentence():
    # Nothing is spoken until a sentence is complete, or TTS gets fragments.
    assert split_sentences("Breakfast is seven to") == ([], "Breakfast is seven to")


def test_split_emits_completed_sentences_only():
    out, rest = split_sentences("Breakfast is seven to ten thirty. Shall I book")
    assert out == ["Breakfast is seven to ten thirty."]
    assert rest.strip() == "Shall I book"


def test_split_does_not_break_on_decimal():
    # "2.30" must not become "2." + "30" - the guest would hear "two. thirty".
    out, rest = split_sentences("Late checkout is 2.30 pm")
    assert out == [], out
    assert rest == "Late checkout is 2.30 pm"


def test_split_handles_question_and_exclamation():
    out, _ = split_sentences("Of course! Which day? And")
    assert out == ["Of course!", "Which day?"]


def test_trim_bounds_history():
    history = [{"role": "user", "content": str(i)} for i in range(100)]
    assert len(trim(history)) == MAX_TURNS * 2
    assert trim(history)[-1] == history[-1]  # keeps the newest, drops the oldest


def test_trim_passes_short_history_through():
    history = [{"role": "user", "content": "hi"}]
    assert trim(history) == history


def test_trim_never_starts_on_assistant():
    # A barge-in before the agent spoke leaves user, user - so the cut can land
    # on an assistant turn. The API rejects a conversation that starts there.
    history = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
    history += [{"role": r, "content": "x"} for r in ["assistant", "user"] * MAX_TURNS]
    assert len(history[-MAX_TURNS * 2 :]) == MAX_TURNS * 2
    assert trim(history)[0]["role"] == "user"


def test_history_survives_reconnect_capped_and_expiring():
    async def go():
        r = fakeredis.FakeAsyncRedis(decode_responses=True)
        live = []
        for i in range(MAX_TURNS * 2 + 3):
            await remember(r, "call-1", live, {"role": "user", "content": str(i)})
        # A new socket for the same call reloads it from Redis.
        reloaded = await load_history(r, "call-1")
        assert len(reloaded) == MAX_TURNS * 2  # LTRIM cap
        assert reloaded[-1]["content"] == str(MAX_TURNS * 2 + 2)  # newest kept
        assert 0 < await r.ttl(history_key("call-1")) <= CALL_TTL_S
        assert await load_history(r, "other-call") == []  # calls don't leak

    asyncio.run(go())


def test_transcript_marks_cut_replies_and_skips_silent_ones():
    usage = {"in": 300, "out": 20}
    cut = transcript_update("pool hours?", "The pool is open six", True, usage)
    assert cut["$push"]["turns"]["$each"] == [
        {"role": "user", "content": "pool hours?"},
        {"role": "assistant", "content": "The pool is open six", "cut": True},
    ]
    assert cut["$inc"] == {"tokens_in": 300, "tokens_out": 20}
    # Barge-in before the agent said anything: only the guest's line is kept.
    silent = transcript_update("wait", "", True, {"in": 0, "out": 0})
    assert silent["$push"]["turns"]["$each"] == [{"role": "user", "content": "wait"}]


def test_archive_upserts_one_document_per_call():
    seen = []

    class FakeCalls:
        async def update_one(self, flt, update, upsert=False):
            seen.append((flt, upsert))

    asyncio.run(archive_turn(FakeCalls(), "call-1", {"$set": {}}))
    assert seen == [({"_id": "call-1"}, True)]


def test_archive_writer_keeps_order_and_drains_on_close():
    # Slow first write, fast second: the transcript must still be in order,
    # and closing the queue must wait for both rather than drop the tail.
    written = []

    class SlowThenFast:
        async def update_one(self, flt, update, upsert=False):
            await asyncio.sleep(0.05 if update["n"] == 1 else 0)
            written.append(update["n"])

    async def go():
        from server import archive_writer

        q = asyncio.Queue()
        writer = asyncio.create_task(archive_writer(SlowThenFast(), "call-1", q))
        q.put_nowait({"n": 1})
        q.put_nowait({"n": 2})
        q.put_nowait(None)
        await writer

    asyncio.run(go())
    assert written == [1, 2]


def test_hangup_right_after_reply_keeps_record_and_history():
    # Regression: the guest hangs up the instant the reply ends, while the turn
    # is still saving. The cancel used to abort the save and drop the record.
    import server

    async def fake_stream(messages, usage):
        usage.update({"in": 270, "out": 9})
        yield "Check-in is at three PM."

    slow_real = server.remember

    async def slow_remember(*a):
        await asyncio.sleep(0.05)  # widen the window the cancel lands in
        await slow_real(*a)

    class Sock:
        def __init__(self): self.sent = []
        async def send_json(self, m): self.sent.append(m["type"])

    async def go():
        r = fakeredis.FakeAsyncRedis(decode_responses=True)
        history, archive, sock = [], asyncio.Queue(), Sock()
        old = (server.stream_reply, server.remember)
        server.stream_reply, server.remember = fake_stream, slow_remember
        try:
            turn = asyncio.create_task(server.respond(
                sock, r, asyncio.Lock(), archive, "c", history,
                {"in": 0, "out": 0, "turns": 0}, "check-in time?"))
            while "done" not in sock.sent:
                await asyncio.sleep(0.001)
            turn.cancel()  # hang-up lands inside the bookkeeping
            try:
                await turn
            except asyncio.CancelledError:
                pass
            await asyncio.sleep(0.2)  # the shielded save finishes on its own
        finally:
            server.stream_reply, server.remember = old
        record = archive.get_nowait()
        assert record["$push"]["turns"]["$each"][-1] == {
            "role": "assistant", "content": "Check-in is at three PM.", "cut": False}
        assert [m["role"] for m in await load_history(r, "c")] == ["user", "assistant"]

    asyncio.run(go())


def test_archive_survives_mongo_being_down():
    # Real driver, nothing listening: the call must carry on, quickly.
    import time

    from pymongo import AsyncMongoClient

    async def go():
        dead = AsyncMongoClient("mongodb://127.0.0.1:1", serverSelectionTimeoutMS=300)
        t0 = time.perf_counter()
        await archive_turn(dead.concierge.calls, "call-1", {"$set": {"x": 1}})
        return time.perf_counter() - t0

    assert asyncio.run(go()) < 3  # gave up and logged; no exception reached the call


def test_groq_stream_yields_text_and_usage():
    # Fake Groq stream: text deltas, then a final chunk carrying usage the way
    # Groq sends it (x_groq.usage). No key, no network.
    from types import SimpleNamespace as NS

    import server

    def chunk(text=None, usage=None):
        delta = NS(content=text)
        return NS(choices=[NS(delta=delta)] if text else [], usage=None,
                  x_groq=NS(usage=usage) if usage else None)

    class FakeStream:
        async def __aenter__(self): return self
        async def __aexit__(self, *exc): return False
        def __aiter__(self):
            async def gen():
                for c in [chunk("Breakfast is "), chunk("seven to ten."),
                          chunk(usage=NS(prompt_tokens=310, completion_tokens=12))]:
                    yield c
            return gen()

    async def create(**kwargs):
        assert kwargs["messages"][0]["role"] == "system"  # SYSTEM goes first
        return FakeStream()

    fake = NS(chat=NS(completions=NS(create=create)))
    old = (server.PROVIDER, server.get_groq)
    server.PROVIDER, server.get_groq = "groq", lambda: fake
    try:
        async def go():
            usage = {"in": 0, "out": 0}
            text = [t async for t in server.stream_reply([{"role": "user", "content": "hi"}], usage)]
            return "".join(text), usage
        text, usage = asyncio.run(go())
    finally:
        server.PROVIDER, server.get_groq = old
    assert text == "Breakfast is seven to ten."
    assert usage == {"in": 310, "out": 12}


def test_cost_matches_list_pricing():
    # 1M in + 1M out on Opus 5 = $5 + $25.
    assert cost_usd("claude-opus-5", 1_000_000, 1_000_000) == 30.0
    assert cost_usd("claude-haiku-4-5", 1_000_000, 1_000_000) == 6.0
    assert cost_usd("claude-haiku-5-5", 1_000_000, 1_000_000) == 0.6
    assert cost_usd("claude-opus-5", 0, 0) == 0.0


def test_cost_falls_back_for_unknown_model():
    assert cost_usd("some-future-model", 1_000_000, 0) == 5.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("\nall passed")
