# AI Voice Concierge

A real-time spoken concierge for a hotel front desk. You talk, it talks back, and
you can cut it off mid-sentence the way you would interrupt a person.

Built to be honest about the things that actually make a voice agent hard:
**how long the guest waits before hearing anything, what happens when they
interrupt, what a conversation costs, and what happens when the line drops.**

```
browser (Web Speech API)  ──ws──►  FastAPI  ──►  Claude / Groq (streaming)
   mic → text                       sentence splitter
   text → speakers  ◄────ws────     per-turn cancel + token meter
                                      │                 │
                       Redis: live call state     MongoDB: full transcripts
                       (capped, expires)          (one document per call)
```

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env        # add your Anthropic (or Groq) API key
docker run -d --name concierge-redis -p 6379:6379 redis:7
docker run -d --name concierge-mongo -p 27017:27017 mongo:7
uvicorn server:app --reload
```

Open <http://127.0.0.1:8000>, press **Talk**, and speak. Chrome or Edge — the
Web Speech API is not in Firefox or Safari.

```bash
pip install fakeredis       # tests use an in-memory Redis
python test_concierge.py    # turn logic, Redis + Mongo stores; no key, no servers
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

### 4. A dropped line must not wipe the call

The browser reconnects 500 ms after the socket drops. When history lived in
the socket, every reconnect started the call from zero. Now the browser keeps a
call id for the tab (`sessionStorage`) and sends it on connect; the server
loads that call's history from Redis.

- `RPUSH` + `LTRIM` keep Redis at the same 8-exchange cap the model sees, and
  `EXPIRE` drops calls idle for 30 minutes (`CONCIERGE_CALL_TTL`).
- The server holds no call state of its own, so any instance can serve a turn.
- Barge-in waits for the cut turn to save what was spoken before the next turn
  starts, so history order in Redis matches what the guest heard.
- The call id must be a UUID; anything else gets a fresh id, so a client can't
  choose its own Redis key.

The per-socket cost meter still resets on reconnect; the history does not.

### 5. Live state and the record are different jobs

Redis is the call's working memory: capped at what the model sees, gone 30
minutes after the call goes quiet. That is the wrong place for a record. Each
finished turn is also upserted into MongoDB (`concierge.calls`, `_id` = call
id): `$push` appends the guest's line and the reply to an uncapped `turns`
array, cut-off replies are marked `"cut": true`, and `$inc` keeps token totals.
One document per call reads naturally as a transcript and needs no joins.

Turns only *enqueue* their record; one writer task per connection drains the
queue to MongoDB in order. So the write never delays speech (a cold Atlas
connection from India took 2.6 s), barge-in can't cancel or reorder it, and on
hang-up the server drains the queue before closing. It is best-effort: if
MongoDB is down, the turn is logged as not archived and the call carries on
(a test points the real driver at a dead port to prove it). Token totals count
completed replies; a reply cut off mid-stream reports no usage.

A cancel can also land *after* the reply finished - a hang-up or barge-in
during the turn's bookkeeping. The record is queued first (synchronous, can't
be interrupted) and the Redis save is shielded and serialised by a per-call
lock, so neither is lost and history order holds. There is a regression test.

## Measured numbers

`bench.py` reports p50/p95 for time-to-first-token, time-to-first-speech, and
full turn, plus cost per turn. Measured 21 Oct 2026 on Groq's free tier, from
India, single-turn prompts:

| model (Groq) | turns | first speech p50 | first speech p95 | full turn p50 | $ / 1,000 turns* |
| --- | --- | --- | --- | --- | --- |
| `qwen/qwen3.8-27b` | 30 | **231 ms** | 3,518 ms | 271 ms | 0.31 |
| `openai/gpt-oss-120b` | 12 | 816 ms | 5,706 ms | 830 ms | 0.09 |
| `openai/gpt-oss-20b` | 12 | 1,294 ms | 6,900 ms | 1,309 ms | 0.04 |

\*at paid-tier list price; the free tier bills $0.

Reading it honestly: the median is what a caller usually feels, and Qwen starts
speaking in about a quarter of a second. The p95 is a few slow outliers per run,
consistent with queueing on shared free capacity, not with the model; a paid
tier or dedicated capacity is how you'd cut that tail. With 12 turns the p95 is
simply the slowest turn, so the 30-turn row is the one to trust. Claude numbers
are not in the table because they need a paid key.

**Two providers, one code path.** `CONCIERGE_PROVIDER=anthropic` (default) runs
Claude; `CONCIERGE_PROVIDER=groq` runs open models on Groq, which has a free
tier. Only `stream_reply()` in `server.py` knows which one is live; the server
and `bench.py` both stream through it, so the benchmark measures exactly what a
caller hears.

Model is set by `CONCIERGE_MODEL`. Default is `claude-opus-5`;
`claude-haiku-5-5` is the latency- and cost-optimised swap — worth running the
benchmark on both before deciding. (`claude-haiku-4-5` rejects the
`effort` setting every turn sends, so it errors here.)

| model | in $/1M | out $/1M |
| --- | --- | --- |
| `claude-opus-5` | 5.00 | 25.00 |
| `claude-sonnet-5` | 2.00 | 10.00 |
| `claude-haiku-5-5` | 0.10 | 0.50 |
| `openai/gpt-oss-120b` (Groq) | 0.15 | 0.60 |
| `qwen/qwen3.8-27b` (Groq) | 0.80 | 4.00 |

Groq's free tier bills $0; its rows show what the same traffic would cost on
the paid tier.

## Scope

Deliberately not built, and why:

- **No WebRTC / server-side audio.** Browser speech recognition and synthesis
  are native, free, and good enough to prove the turn-taking design. Moving
  audio to the server is what you do when you need telephony, a specific voice,
  or a language the browser lacks — and then the interesting code, the part in
  this repo, is unchanged.
- **No search or analytics on transcripts yet.** They are stored, one
  document per call; querying them is the next project.
- **No retrieval.** The hotel's facts are in the system prompt, which is the
  right call at this size; a fact base that outgrows a prompt is a different
  project.
- **No auth, no rate limiting.** Don't put this on the public internet as-is.

Known limitation: echo suppression is a text heuristic, not acoustic echo
cancellation. It holds for normal speech and fails if the guest repeats the
agent's own words back verbatim.
