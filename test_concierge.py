"""Self-check for the turn logic. No API key, no network, no Redis server.

    pip install fakeredis   # in-memory Redis for the call-store tests
    python test_concierge.py
"""

import asyncio

import fakeredis

from server import (
    CALL_TTL_S,
    MAX_TURNS,
    cost_usd,
    history_key,
    load_history,
    remember,
    split_sentences,
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
