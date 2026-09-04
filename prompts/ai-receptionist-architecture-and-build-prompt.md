# AI Voice Receptionist — Complete System Architecture & Build Spec

Stack: **LiveKit Agents + Grok (xAI) + Deepgram (STT) + FAISS RAG + FastAPI + edge-tts**
Goal: Real-time, interruptible, document-grounded voice conversation (Gemini Live / ChatGPT Voice-style UX), built free-tier.

---

## 1. System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                            CLIENT (Browser)                          │
│   LiveKit Web SDK (or LiveKit Playground for prototype)              │
│   - Captures continuous mic audio                                    │
│   - Plays back streamed TTS audio                                    │
│   - Shows live transcript (optional)                                 │
└───────────────────────────┬────────────────────────────────────────-─┘
                             │ WebRTC
                             ▼
┌────────────────────────────────────────────────────────────────────┐
│                     LiveKit Room (Cloud free tier / self-hosted)     │
│   - Voice Activity Detection (VAD)                                   │
│   - Interruption / barge-in detection                                │
│   - Audio transport, jitter buffering, echo cancellation             │
└───────────────────────────┬──────────────────────────────────────---┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────────────┐
│                    LiveKit Agent (Python worker process)            │
│                                                                       │
│   ┌───────────┐   ┌───────────────┐   ┌───────────┐   ┌──────────┐ │
│   │ STT        │──▶│ Orchestrator  │──▶│ LLM        │──▶│ TTS      │ │
│   │ Deepgram   │   │ (state +      │   │ Grok API   │   │ edge-tts │ │
│   │ Nova       │   │  guardrails)  │   │ (streamed) │   │ (streamed│ │
│   │ (streamed, │   └───────┬───────┘   └───────────┘   │ sentence-│ │
│   │ endpointing)│           │                             │ by-      │ │
│                             │                             │ sentence)│ │
│                             ▼                             └──────────┘ │
│                   ┌──────────────────┐                                │
│                   │ RAG Retriever     │                                │
│                   │ FAISS + sentence- │                                │
│                   │ transformers       │                                │
│                   │ embeddings         │                                │
│                   └─────────┬──────────┘                               │
│                             │                                          │
└─────────────────────────────┼──────────────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────────┐
│                         Persistence Layer                            │
│   Supabase (Postgres, free tier)                                     │
│   - conversation_logs, call_sessions, escalations, appointments      │
│   FAISS index files (local disk or Supabase Storage)                 │
└────────────────────────────────────────────────────────────────────┘
```

---

## 2. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Real-time transport | LiveKit Agents (Cloud free tier or self-hosted OSS) | Handles VAD, barge-in, streaming — the hard 20% |
| STT | Deepgram Nova (streaming) | Purpose-built for real-time streaming, best-in-class endpointing (turn-taking accuracy), $200 free credit, LiveKit's most-used/reference STT plugin |
| LLM | Grok API (xAI, OpenAI-compatible endpoint) | Requested; swap to Groq/Llama as free fallback |
| Embeddings | sentence-transformers (local, offline) | Zero API cost, no quota burn |
| Vector store | FAISS (in-memory / local disk) | Free, fast for small-medium doc sets |
| TTS | edge-tts | Free, natural-sounding, streams well |
| Orchestration logic | LangGraph (state machine) | You already use this — models intents as nodes |
| Backend API | FastAPI | Token generation, document ingestion endpoint, logs |
| Database | Supabase (Postgres, free tier) | Conversation logs, escalations |
| Frontend | LiveKit Playground (prototype) → React/Vite later | Zero frontend code needed to test voice loop first |
| Deployment | Render/Fly.io free tier (agent worker), Vercel (frontend) | Matches your existing deployment experience |

---

## 3. Guardrails & Edge Cases This Design Must Handle

- **Hallucination control**: LLM must answer ONLY from retrieved document context. If retrieval confidence is low (similarity score below threshold) or no relevant chunk found → respond with a graceful "I don't have that information, let me note it down" instead of guessing.
- **Interruption mid-response**: if user speaks while TTS is playing, stop generation/playback immediately and prioritize the new input — this is native to LiveKit's VAD but must be wired correctly in the agent loop.
- **Silence / no input**: after N seconds of silence, the agent should prompt once ("Are you still there?") rather than sit dead or loop endlessly.
- **Ambiguous or out-of-scope questions**: classify intent before answering — if it's outside the receptionist's domain (document scope), redirect politely rather than improvising.
- **Latency**: target under ~800ms to first audio token. Stream every stage (STT partials, LLM tokens, TTS sentence-chunks) — never wait for a full response before starting playback.
- **Document ingestion failures**: malformed PDFs, empty docs, or unsupported formats should fail with a clear error, not silently produce an empty index.
- **API failures**: Grok/Groq/edge-tts calls can fail or rate-limit — every external call needs retry-with-backoff and a safe fallback message ("Sorry, I'm having trouble right now, could you repeat that?") instead of crashing the session.
- **Sensitive/PII data**: if the conversation touches personal info (phone numbers, names, appointment details), log it to Supabase but never echo it back unnecessarily or store more than needed.
- **Concurrent sessions**: each caller gets an isolated LiveKit room + agent instance — no shared state/memory bleed between different users' conversations.
- **Cost/quota guardrails**: track token usage and TTS/STT minutes per session so a runaway loop (e.g., agent talking to itself) doesn't burn your free-tier quota.

---

## 4. The Build Prompt (paste into Google Antigravity)

```
You are building a production-quality real-time voice AI receptionist. Follow these steps
in order. Do not skip validation steps. After each step, verify it works before moving to
the next.

CONTEXT
- Stack: LiveKit Agents (Python), Deepgram Nova for STT (streaming), Grok API (xAI,
  OpenAI-compatible) for the LLM, FAISS + sentence-transformers for RAG, edge-tts for TTS,
  FastAPI for supporting REST endpoints, Supabase (Postgres) for logging.
- Goal: a continuously-listening, interruptible, real-time voice agent that answers
  questions ONLY using the knowledge from documents I provide (RAG), with graceful
  fallback when it doesn't know something.

STEP 1 — Project scaffolding
- Create a Python project with a clear structure: /agent (LiveKit agent worker),
  /api (FastAPI app for token generation, document upload, logs), /rag (ingestion +
  retrieval logic), /tests.
- Set up a .env.example listing every required key: LIVEKIT_URL, LIVEKIT_API_KEY,
  LIVEKIT_API_SECRET, DEEPGRAM_API_KEY, XAI_API_KEY, SUPABASE_URL, SUPABASE_KEY.
- Never hardcode secrets. Load via environment variables only.

STEP 2 — Document ingestion & RAG pipeline
- Build an ingestion script/endpoint that accepts PDF/DOCX/TXT, extracts text, chunks it
  with a recursive character splitter (target ~500 tokens per chunk, ~50 token overlap),
  embeds chunks with sentence-transformers (all-MiniLM-L6-v2 or similar free local model),
  and stores vectors in a FAISS index persisted to disk.
- Validate: reject empty documents, log a clear error for unsupported formats, and confirm
  chunk count > 0 before marking ingestion successful.
- Write a retrieval function: given a query, embed it, search FAISS top-k (k=4), and return
  chunks ONLY above a similarity threshold — if nothing clears the threshold, return an
  explicit "no relevant context found" signal instead of the closest-but-irrelevant chunk.

STEP 3 — LiveKit Agent: STT stage
- Implement the agent worker using LiveKit Agents framework, wired to a LiveKit Cloud
  free-tier project (or self-hosted instance).
- Use the LiveKit Deepgram plugin (Nova model) as the STT provider with streaming and
  interim results enabled — partial transcripts should be available as the user speaks,
  not just after they stop.
- Enable Deepgram's endpointing/utterance-end settings so speech-boundary detection is
  tuned for natural turn-taking, not just raw silence detection.
- Confirm VAD is active and correctly detects speech start/end boundaries.

STEP 4 — Orchestration & guardrails layer
- Between STT output and the LLM call, insert a LangGraph state machine with at least
  these nodes: intent_classify -> retrieve_context -> generate_answer -> fallback_handoff.
- intent_classify: determine if the query is in-scope (answerable from documents) or
  out-of-scope (redirect politely, do not attempt to answer).
- retrieve_context: call the RAG retrieval function from Step 2.
- generate_answer: call Grok API with a strict system prompt: "Answer ONLY using the
  provided context. If the context does not contain the answer, say you don't have that
  information and offer to take a message. Never invent facts not present in the context."
- fallback_handoff: triggered when retrieval returns no relevant chunks, or the LLM
  reports uncertainty — respond gracefully and log the unanswered question to Supabase
  for later review.
- Stream the LLM response token-by-token, not as one blocking call.

STEP 5 — TTS stage with sentence-level streaming
- As LLM tokens arrive, buffer until a complete sentence is formed, then send that
  sentence to edge-tts immediately and stream the resulting audio back through the
  LiveKit room — do not wait for the full LLM response before starting speech.
- Confirm audio playback begins within ~800ms of the first LLM tokens arriving.

STEP 6 — Interruption (barge-in) handling
- When the user starts speaking while TTS audio is playing, immediately stop TTS
  playback and cancel the in-flight LLM generation for that turn.
- Prioritize the new user input as the next turn. Verify this works with rapid
  back-to-back interruptions, not just a single clean interrupt.

STEP 7 — Silence & error handling
- If no user speech is detected for a configurable timeout (default 8s), have the agent
  prompt once ("Are you still there?"); if still silent after a second timeout, end the
  session gracefully.
- Wrap every external API call (Deepgram, Grok, edge-tts, Supabase) in retry-with-backoff
  (max 2 retries) and a safe fallback spoken message on failure — the agent must never
  hard-crash mid-conversation.

STEP 8 — Persistence & logging
- Log every session to Supabase: session_id, transcript (full turn history), unanswered
  questions (from fallback_handoff), timestamps, and any captured contact info.
- Do not log more personal data than necessary, and never expose logs via a public
  endpoint without auth.

STEP 9 — Testing
- Write tests for: RAG retrieval accuracy on sample documents, fallback triggering on
  out-of-scope questions, and interruption behavior (simulate mid-speech user input).
- Manually test end-to-end via the LiveKit Agents Playground before building any custom
  frontend.

STEP 10 — Prototype validation checklist (do not consider this done until all pass)
- [ ] Agent responds with correct, document-grounded answers to in-scope questions.
- [ ] Agent gracefully declines out-of-scope questions without hallucinating.
- [ ] User can interrupt the agent mid-sentence and it stops immediately.
- [ ] First audio response begins in under ~1 second after the user finishes speaking.
- [ ] No API key or secret appears in logs, client code, or version control.
- [ ] A malformed or empty uploaded document fails ingestion with a clear error, not a
      silent empty index.
- [ ] Session survives at least one simulated API failure (e.g., Grok timeout) without
      crashing, and recovers with a spoken fallback message.

Build this incrementally, step by step, and show me the working result of each step
before moving to the next.
```

---

## 5. Suggested Build Order (if doing it yourself, not via the prompt)

1. RAG pipeline standalone (validate retrieval quality on your real document first)
2. LiveKit Agent with STT + LLM (no RAG yet) — confirm the voice loop and interruption work
3. Wire RAG into the orchestration node
4. Add guardrails (fallback, silence handling, retries)
5. Add Supabase logging
6. Test via LiveKit Playground
7. Only then build a custom frontend
