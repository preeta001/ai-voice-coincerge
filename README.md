# AI Voice Concierge

A real-time spoken concierge for a hotel front desk. You talk, it talks back, and
you can cut it off mid-sentence the way you would interrupt a person.

Built to be honest about the three things that actually make a voice agent hard:
**how long the guest waits before hearing anything, what happens when they
interrupt, and what a conversation costs.** All three are measured, not claimed.

```
browser (Web Speech API)  ──ws──►  FastAPI  ──►  Claude (streaming)
   mic → text                       sentence splitter
   text → speakers  ◄────ws────     per-turn cancel + token meter
```

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env        # add your Anthropic API key
uvicorn server:app --reload
```

Open <http://127.0.0.1:8000>, press **Talk**, and speak. Chrome or Edge — the
Web Speech API is not in Firefox or Safari.

```bash
python test_concierge.py    # turn logic, no API key needed
python bench.py             # latency + cost, needs a key
```

## The three problems

### 1. Time to first speech, not time to first token

Streaming tokens is not the same as speaking sooner. Nothing can be spoken until
a *sentence* is complete, so the metric that matters is time-to-first-sentence.
The server splits the token stream on sentence boundaries and pushes each
finished sentence to the browser immediately, so speech starts on sentence one
instead of at end-of-turn.

The splitter has to know that `2.30 pm` is not the end of a sentence — otherwise
the guest hears *"two."* … *"thirty pm."* That case is in the tests.

### 2. Interruption (barge-in)

Each turn is an `asyncio.Task`. A `barge_in` message cancels it mid-stream, and
the browser calls `speechSynthesis.cancel()`.

The subtle part is that **the microphone hears the speakers.** Naive barge-in
detection fires on the agent's own voice and the agent interrupts itself in a
loop. The client keeps the text currently in the speakers and ignores any
transcript whose every word appears in it, so only a genuinely new utterance
counts as an interruption.

When a turn is cut, the history records *what was actually spoken* before the
cut, not what the model intended to say. Otherwise the next turn answers a
question the guest never heard.

### 3. Cost and context, per turn

Every response's `usage` is accumulated and the running cost is on screen while
you talk. History is trimmed to the last 8 exchanges — a voice call that runs
twenty minutes otherwise pays to re-send the whole conversation every turn.

## Measured numbers

Run `python bench.py` and paste your own results here. It reports p50/p95 for
time-to-first-token, time-to-first-speech, and full turn, plus cost per turn and
a projection per 1,000 turns.

| metric | p50 | p95 |
| --- | --- | --- |
| time to first token | _run bench.py_ | |
| time to first speech | | |
| full turn | | |

Model is set by `CONCIERGE_MODEL`. Default is `claude-opus-5`;
`claude-haiku-4-5` is the latency- and cost-optimised swap — worth running the
benchmark on both before deciding.

| model | in $/1M | out $/1M |
| --- | --- | --- |
| `claude-opus-5` | 5.00 | 25.00 |
| `claude-sonnet-5` | 2.00 | 10.00 |
| `claude-haiku-4-5` | 1.00 | 5.00 |

## Scope

Deliberately not built, and why:

- **No WebRTC / server-side audio.** Browser speech recognition and synthesis
  are native, free, and good enough to prove the turn-taking design. Moving
  audio to the server is what you do when you need telephony, a specific voice,
  or a language the browser lacks — and then the interesting code, the part in
  this repo, is unchanged.
- **No database.** History lives in the WebSocket connection. One caller, one
  call.
- **No retrieval.** The hotel's facts are in the system prompt, which is the
  right call at this size; a fact base that outgrows a prompt is a different
  project.
- **No auth, no rate limiting.** Don't put this on the public internet as-is.

Known limitation: echo suppression is a text heuristic, not acoustic echo
cancellation. It holds for normal speech and fails if the guest repeats the
agent's own words back verbatim.
