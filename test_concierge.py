"""Self-check for the turn logic. No API key, no network.

    python test_concierge.py
"""

from server import MAX_TURNS, cost_usd, split_sentences, trim


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


def test_cost_matches_list_pricing():
    # 1M in + 1M out on Opus 5 = $5 + $25.
    assert cost_usd("claude-opus-5", 1_000_000, 1_000_000) == 30.0
    assert cost_usd("claude-haiku-4-5", 1_000_000, 1_000_000) == 6.0
    assert cost_usd("claude-opus-5", 0, 0) == 0.0


def test_cost_falls_back_for_unknown_model():
    assert cost_usd("some-future-model", 1_000_000, 0) == 5.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("\nall passed")
