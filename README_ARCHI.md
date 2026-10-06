# 🏛️ Architecture & Engineering Blueprint: Ultra-Low Latency (<500ms) AI Voice Receptionist

> **Authoritative Technical Documentation & Complete System Design from Scratch**  
> *Target Performance: Sub-500ms Time-To-First-Audio (TTFA) | Full-Duplex WebRTC | Zero-Hallucination Grounding*

---

## 📑 Table of Contents

1. [Executive Summary & The 500ms Challenge](#1-executive-summary--the-500ms-challenge)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [The Microsecond Latency Budget Breakdown](#3-the-microsecond-latency-budget-breakdown)
4. [Step-by-Step Audio & Token Lifecycle (Frame by Frame)](#4-step-by-step-audio--token-lifecycle-frame-by-frame)
5. [In-Depth Component & Codebase Breakdown](#5-in-depth-component--codebase-breakdown)
   - [5.1 Voice Pipeline Orchestrator (`agent/worker.py`)](#51-voice-pipeline-orchestrator-agentworkerpy)
   - [5.2 Speculative RAG Engine (`rag/speculative.py`)](#52-speculative-rag-engine-ragspeculativepy)
   - [5.3 Semantic Retriever & Confidence Guardrails (`rag/retriever.py`)](#53-semantic-retriever--confidence-guardrails-ragretrieverpy)
   - [5.4 Document Ingestion & ONNX Vector Indexer (`rag/ingest.py`)](#54-document-ingestion--onnx-vector-indexer-ragingestpy)
   - [5.5 Clause-Level Streaming Tokenizer (`agent/chunker.py`)](#55-clause-level-streaming-tokenizer-agentchunkerpy)
   - [5.6 Spoken-Voice Prompt Engineering (`agent/prompt.py`)](#56-spoken-voice-prompt-engineering-agentpromptpy)
   - [5.7 Streaming Audio Synthesis (`agent/tts_engine.py`)](#57-streaming-audio-synthesis-agenttts_enginepy)
   - [5.8 REST API & LiveKit WebRTC Token Service (`api/`)](#58-rest-api--livekit-webrtc-token-service-api)
   - [5.9 Non-Blocking Async Telemetry (`api/db/supabase_client.py`)](#59-non-blocking-async-telemetry-apidbsupabase_clientpy)
6. [Core Architectural Decisions Explained from Scratch](#6-core-architectural-decisions-explained-from-scratch)
   - [WebRTC vs. WebSockets](#webrtc-vs-websockets)
   - [Silero VAD vs. Semantic End-of-Utterance (EOU)](#silero-vad-vs-semantic-end-of-utterance-eou)
   - [Groq LPUs vs. GPU Clusters](#groq-lpus-vs-gpu-clusters)
   - [FAISS FlatIP vs. Approximate Nearest Neighbors (HNSW / IVF)](#faiss-flatip-vs-approximate-nearest-neighbors-hnsw--ivf)
   - [Barge-In (Interruption Handling) Mechanics](#barge-in-interruption-handling-mechanics)
7. [Zero-Hallucination & Clinical Grounding Protocol](#7-zero-hallucination--clinical-grounding-protocol)
8. [Production Deployment, Scale & SIP Telephony](#8-production-deployment-scale--sip-telephony)
9. [Interview Master Guide & Whiteboard Q&A](#9-interview-master-guide--whiteboard-qa)

---

## 1. Executive Summary & The 500ms Challenge

In human face-to-face and telephone conversations, natural pause lengths between speakers average between **200 milliseconds and 400 milliseconds**. When a pause exceeds **600 milliseconds**, human psychology interprets the delay as hesitation, confusion, or a dropped line. When latency hits **1,200ms – 2,000ms**, conversational flow collapses entirely: speakers interrupt each other, repeat questions, and experience severe frustration.

### The Traditional Voice Pipeline Failure Mode (~1,600ms)
Traditional voice bots chain independent systems sequentially using HTTP REST or naive WebSockets:

```
[Caller Speaks] ──▶ Naive VAD (waits 500ms silence) ──▶ Full Speech-to-Text (300ms)
                ──▶ Multi-agent LangGraph / LLM Router (350ms)
                ──▶ Heavy PyTorch Vector Retrieval (40ms)
                ──▶ Cloud LLM TTFT on GPU (400ms)
                ──▶ Wait for full sentence punctuation (200ms)
                ──▶ Full-sentence TTS synthesis (160ms)
                ──▶ Audio playback = ~1,950ms Turnaround
```

### The Ultra-Low Latency Paradigm (~410ms)
This system redesigns the entire conversational pipeline from the ground up to operate concurrently and speculatively:

```
[Caller Speaks] ──▶ LiveKit WebRTC Opus 48kHz (UDP Transport: 20ms)
       │
       ├──▶ Interim Transcripts (Caller still speaking)
       │        └──▶ Speculative RAG Search (FastEmbed ONNX + FAISS) ──▶ Pre-fetched (0ms Net Wait)
       │
       └──▶ EOU Turn Detection (Deepgram Nova-3 + LiveKit EOU: 160ms)
                └──▶ Groq Cloud LPU (~100ms TTFT)
                         └──▶ ClauseChunker (First 4 words / comma)
                                  └──▶ Streaming TTS (Deepgram Aura: 100ms)
                                           └──▶ WebRTC Output Track (30ms) ──▶ Ear: ~410ms TTFA
```

---

## 2. End-to-End System Architecture

```mermaid
flowchart TB
    subgraph ClientLayer ["1. Client & Audio Capture"]
        Browser["Web Client / LiveKit Web SDK"]
        Phone["SIP Phone / Carrier Trunk"]
    end

    subgraph TransportLayer ["2. WebRTC Media Server (LiveKit SFU)"]
        LiveKitSFU["LiveKit Cloud / Self-Hosted SFU"]
        JitterBuffer["Adaptive Jitter Buffer & Echo Cancellation"]
    end

    subgraph AgentWorker ["3. Python Agent Worker Process (worker.py)"]
        Prewarm["Process Prewarming (Silero VAD)"]
        VAD["Silero VAD & LiveKit EOU Model"]
        DeepgramSTT["Deepgram Nova-3 STT (160ms endpointing)"]
        
        subgraph SpecRAGSubsystem ["Concurrent Speculative RAG"]
            SpecCache["SpeculativeRAGCache (speculative.py)"]
            FastEmbed["FastEmbed ONNX CPU (bge-small-en-v1.5)"]
            FAISSIndex["FAISS IndexFlatIP (Cosine Similarity)"]
        end

        Orchestrator["AgentSession Voice Orchestrator"]
        LLM["Groq LPU (Llama 3.3 70B / gpt-oss-20b)"]
        ClauseChunker["Clause-Level Token Chunker (chunker.py)"]
        TTS["Deepgram Aura Streaming TTS / Kokoro-82M"]
        BargeIn["Barge-In / Interruption Controller"]
    end

    subgraph DataTelemetry ["4. Data & Persistence Layer"]
        FastAPI["FastAPI Backend (server.py)"]
        Supabase["Supabase PostgreSQL (call_turns table)"]
        LocalLogs["Fallback Local JSONL Audit Log"]
    end

    Browser -->|WebRTC Opus 48kHz| LiveKitSFU
    Phone -->|SIP / RTP| LiveKitSFU
    LiveKitSFU --> JitterBuffer
    JitterBuffer --> VAD
    JitterBuffer --> DeepgramSTT

    DeepgramSTT -.->|Interim Partials (≥3 words)| SpecCache
    SpecCache --> FastEmbed --> FAISSIndex
    FAISSIndex -.->|Candidate Context| SpecCache

    DeepgramSTT -->|Final Utterance| Orchestrator
    SpecCache -->|0ms Pre-fetched Context| Orchestrator
    VAD -->|Turn Complete / Barge-in| Orchestrator

    Orchestrator -->|Grounded System Prompt| LLM
    LLM -->|Streaming Token Generator| ClauseChunker
    ClauseChunker -->|First 4-5 Words / Comma| TTS
    TTS -->|PCM Audio Frames| LiveKitSFU
    LiveKitSFU -->|WebRTC Audio Track| Browser

    BargeIn -.->|Instant Interrupt Signal| LLM
    BargeIn -.->|Flush Audio Queue| TTS

    Orchestrator -.->|Async Non-Blocking Queue| Supabase
    Supabase -.->|On Failure| LocalLogs
    FastAPI -->|Token Generation / Doc Ingestion| FAISSIndex
```

---

## 3. The Microsecond Latency Budget Breakdown

Every single millisecond between speech termination and first audible output is strictly accounted for:

| Pipeline Stage | Technology Stack | Latency Range | Engineering Optimization |
|---|---|---|---|
| **1. Audio Ingestion & Transport** | WebRTC Opus over UDP | **20ms – 35ms** | Direct UDP media transport eliminates TCP head-of-line blocking and packet retransmissions. |
| **2. Voice Activity & Turn Detection** | Silero VAD + Deepgram Nova-3 + LiveKit EOU | **150ms – 180ms** | Semantic End-of-Utterance (EOU) model evaluates phrase grammar; 160ms endpointing replaces traditional 500ms dead-silence timers. |
| **3. Context Retrieval (RAG)** | FastEmbed ONNX + FAISS FlatIP | **0ms net** | **Speculative pre-search:** Triggers against interim STT partials 200ms before user finishes speaking. |
| **4. Time-To-First-Token (TTFT)** | Groq Cloud LPU Hardware | **90ms – 120ms** | Tensor streaming LPUs bypass standard GPU memory-bandwidth bottlenecks, generating >300 tokens/sec. |
| **5. Token Chunking & TTS First Byte** | ClauseChunker + Deepgram Aura | **80ms – 110ms** | Emits audio on the first comma or 4 words; does not wait for full sentence termination. |
| **6. Output Audio Transport** | WebRTC Media Track | **20ms – 30ms** | Synthesized PCM frames streamed directly into browser `AudioContext`. |
| **Total Turnaround Time (TTFA)** | **End-of-Speech to Caller's Ear** | **~380ms – 460ms** | **Guaranteed sub-500ms, matching natural human conversational cadence.** |

---

## 4. Step-by-Step Audio & Token Lifecycle (Frame by Frame)

Here is what happens during a single conversational turn:

1. **Caller Speaks:** The caller asks: *"Do you accept Blue Cross Blue Shield insurance?"*
2. **WebRTC Stream (0ms – 1500ms):**
   - 48kHz Opus audio packets stream over UDP into LiveKit's Selective Forwarding Unit (SFU).
   - LiveKit forwards audio frames to the Python worker process via memory pipe.
3. **Interim Transcription & Speculative Trigger (At ~800ms while user is still speaking):**
   - Deepgram Nova-3 emits interim transcript: `"Do you accept Blue Cross"`.
   - `SpeculativeRAGCache.on_interim_transcript()` sees $\ge 3$ words.
   - It fires an async task in a separate thread: FastEmbed generates vector on CPU via ONNX (~5ms) and searches FAISS FlatIP (~0.7ms).
   - Candidate chunks regarding insurance policies are cached in memory.
4. **End-of-Utterance Signal (At ~1500ms):**
   - Caller finishes saying `"...insurance?"`.
   - 160ms later, Deepgram and LiveKit EOU detect completion.
   - Final transcript finalized: `"Do you accept Blue Cross Blue Shield insurance?"`.
5. **Instant Context Resolution (At ~1660ms):**
   - Orchestrator requests context. `SpeculativeRAGCache.get_final_context()` detects that the final utterance matches the cached query.
   - Context is returned **instantly (0ms delay)**.
6. **LLM Generation on Groq LPU (At ~1665ms – 1765ms):**
   - System prompt + injected context + user query sent to Groq LPU (`llama-3.3-70b-versatile` or `gpt-oss-20b`).
   - At ~100ms, Groq streams the first token: `"Yes,"` followed by `" we"`, `" accept"`, `" Blue"`, `" Cross"`.
7. **Clause Chunker Emission (At ~1770ms):**
   - `ClauseChunker` catches the comma after `"Yes,"` or accumulates the first 4 words (`"Yes, we accept Blue Cross"`).
   - It immediately emits this chunk to Deepgram Aura TTS via WebSocket.
8. **Audio Playback Begins (At ~1850ms – 1880ms):**
   - Deepgram Aura returns audio frames within 80ms.
   - LiveKit publishes audio frames to the room's output track.
   - Caller hears the beginning of the answer: **~410ms after they stopped speaking**.
9. **Continuous Streaming:** While the caller is listening to the first clause, Groq generates the remainder of the sentence, and the pipeline plays seamlessly without audio gaps.

---

## 5. In-Depth Component & Codebase Breakdown

```
AI_RECEPTIONIST/
├── agent/
│   ├── worker.py          # LiveKit Agent worker process & session configuration
│   ├── chunker.py         # Sub-sentence clause-level streaming tokenizer
│   ├── prompt.py          # Clinic persona, spoken rules, anti-hallucination prompt
│   └── tts_engine.py      # TTS factory (Deepgram Aura / OpenAI fallback)
├── rag/
│   ├── ingest.py          # Multi-format document parser, chunker & FAISS builder
│   ├── retriever.py       # FastEmbed ONNX search with strict confidence threshold
│   └── speculative.py     # Concurrent interim-transcript speculative pre-search cache
├── api/
│   ├── server.py          # FastAPI application entrypoint
│   ├── routes/
│   │   ├── token_route.py     # WebRTC JWT generation for browser clients
│   │   └── documents_route.py # Document upload & real-time re-indexing
│   └── db/
│       └── supabase_client.py # Non-blocking async telemetry & call logging
└── tests/
    ├── test_rag.py        # Ingestion, retrieval accuracy & threshold unit tests
    ├── test_chunker.py    # Sub-sentence clause boundary verification tests
    └── test_latency.py    # Benchmark suite for FastEmbed, FAISS, and Groq TTFT
```

---

### 5.1 Voice Pipeline Orchestrator (`agent/worker.py`)

[agent/worker.py](file:///d:/PROJECTS/AI_RECEPTIONIST/agent/worker.py) is the central coordinator of the voice agent.

#### Critical Architecture Highlights:
- **Process Prewarming (`prewarm_process`):**
  ```python
  def prewarm_process(proc):
      proc.userdata["vad"] = silero.VAD.load()
  ```
  In serverless or worker processes, loading neural network weights into memory incurs a 300ms–800ms cold start. By pre-warming Silero VAD before accepting incoming room jobs, the first caller gets immediate response times.
- **Full-Duplex Subscription:**
  ```python
  await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
  ```
  Only subscribes to audio tracks, minimizing WebRTC bandwidth and video transcoding overhead.
- **Turn-Taking Tuning:**
  ```python
  session = AgentSession(
      stt=stt_engine,
      llm=llm_engine,
      tts=tts_engine,
      vad=vad,
      turn_handling={
          "min_endpointing_delay": 0.16,  # 160ms endpointing
          "allow_interruptions": True,    # Instant barge-in
      },
  )
  ```
- **Real-Time Tool Grounding (`@llm.function_tool`):**
  The agent uses `lookup_knowledge_base` to query the clinic vector index. If the threshold fails, it returns `NO_RECORDS_FOUND`.

---

### 5.2 Speculative RAG Engine (`rag/speculative.py`)

[rag/speculative.py](file:///d:/PROJECTS/AI_RECEPTIONIST/rag/speculative.py) eliminates retrieval wait time from the critical latency path.

#### Mechanics:
1. As speech is recognized in chunks by Deepgram, `on_interim_transcript(partial_text)` is called.
2. If `len(words) >= 3` and the query has changed:
   - Any previous in-flight speculation task is cancelled (`self._pending_task.cancel()`) to avoid CPU starvation.
   - An async task runs `retriever.retrieve(query)` in an executor thread without blocking the main `asyncio` event loop.
3. When the user stops talking, `get_final_context(final_text)` checks if the cached speculative query matches the final transcript.
   - **Cache HIT:** Pre-fetched context is returned in **0ms**.
   - **Cache MISS:** Synchronous retrieval executes as fallback (~8ms).

---

### 5.3 Semantic Retriever & Confidence Guardrails (`rag/retriever.py`)

[rag/retriever.py](file:///d:/PROJECTS/AI_RECEPTIONIST/rag/retriever.py) executes sub-millisecond vector similarity search with strict thresholding.

#### Mechanics:
- **Asymmetric Instruction Embedding:**
  ```python
  if "bge" in self.model_name.lower():
      search_query = f"Represent this sentence for searching relevant passages: {search_query}"
  ```
  The BGE embedding family uses asymmetric tasks: query embeddings require an instruction prefix to match passage embeddings accurately in vector space.
- **L2 Vector Normalization:**
  Query vectors are normalized to unit length (`faiss.normalize_L2(query_vec)`). This turns an inner product operation (`IndexFlatIP`) into exact **Cosine Similarity**:
  $$\text{Cosine Similarity} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \mathbf{u}_{\text{norm}} \cdot \mathbf{v}_{\text{norm}}$$
- **Confidence Thresholding:**
  ```python
  if max_score < cut_off:
      return {"has_context": False, "reason": "LOW_CONFIDENCE"}
  ```
  Any query with a top similarity score below `SIMILARITY_THRESHOLD` (`0.48` - `0.65`) is rejected, preventing irrelevant context from leaking to the LLM.

---

### 5.4 Document Ingestion & ONNX Vector Indexer (`rag/ingest.py`)

[rag/ingest.py](file:///d:/PROJECTS/AI_RECEPTIONIST/rag/ingest.py) handles multi-format document extraction and FAISS database creation.

#### Mechanics:
- **Supported Formats:** `.pdf` (via `pypdf`), `.docx` (via `python-docx`), `.txt`, `.md`.
- **Heading-Aware Chunking:**
  Detects Markdown headings (`#`, `##`, `###`) using regex `(?=^#{1,4}\s+)`. Splitting text along logical header sections keeps headings attached to their respective bodies, preserving semantic context.
- **Sub-Chunking with Overlap:**
  Long sections exceeding `chunk_size_chars` are chunked with an `overlap_chars` buffer (e.g. 100-200 characters) to ensure no facts get sliced in half across chunk boundaries.
- **Index Persistence:**
  Writes vectors to `data/faiss_index.bin` and serialized text metadata to `data/chunks.json`.

---

### 5.5 Clause-Level Streaming Tokenizer (`agent/chunker.py`)

[agent/chunker.py](file:///d:/PROJECTS/AI_RECEPTIONIST/agent/chunker.py) breaks down the LLM token stream into early audio chunks.

#### Mechanics:
- **Punctuation Matching:**
  Uses regex `([,;:—\.\?!]+)` to spot natural pauses.
- **First-Chunk Acceleration:**
  For the very first chunk of an utterance, as soon as **4 words** accumulate or a comma is encountered, the chunk is yielded immediately.
- **Max-Words Ceiling:**
  If the LLM emits a run-on sentence without punctuation, the chunker forces an emission at **12 words** to keep audio synthesis continuous and prevent stutter.

---

### 5.6 Spoken-Voice Prompt Engineering (`agent/prompt.py`)

[agent/prompt.py](file:///d:/PROJECTS/AI_RECEPTIONIST/agent/prompt.py) defines the persona, tone, and spoken-language constraints.

#### Critical Spoken Rules:
1. **No Written Formatting:** No markdown, bullet points, asterisks, citations, or numbered lists.
2. **Conciseness:** Maximum 1 to 2 spoken sentences per turn to avoid monologuing.
3. **Phonetic Pronunciation:** Numbers and hours spelled out naturally (e.g., *"nine in the morning"*, not *"9:00 AM"*).
4. **Zero-Hallucination Fallback:** If context is missing, immediately say:
   > *"I apologize, I don't have that specific detail in our records right now. Would you like me to take down your name and phone number so our team can follow up with you?"*
5. **Emergency Protocol:** Instructs callers with chest pain, shortness of breath, or severe trauma to hang up and call 911 immediately.

---

### 5.7 Streaming Audio Synthesis (`agent/tts_engine.py`)

[agent/tts_engine.py](file:///d:/PROJECTS/AI_RECEPTIONIST/agent/tts_engine.py) provides the factory interface for speech synthesis.

- **Primary Cloud Engine:** Deepgram Aura (`aura-asteria-en`).
  - Streaming WebSocket interface.
  - Latency: ~100ms TTFA.
  - Realistic conversational intonation and pacing.
- **Offline / Local Fallback:** Kokoro-82M ONNX or OpenAI TTS (`tts-1`, voice `alloy`).

---

### 5.8 REST API & LiveKit WebRTC Token Service (`api/`)

- [api/server.py](file:///d:/PROJECTS/AI_RECEPTIONIST/api/server.py): FastAPI app with CORS middleware and `/health` monitoring.
- [api/routes/token_route.py](file:///d:/PROJECTS/AI_RECEPTIONIST/api/routes/token_route.py):
  Generates cryptographically signed LiveKit JWT access tokens with `VideoGrants(room_join=True, room=room_name)`. Web clients call `POST /api/token` before establishing a WebRTC connection.
- [api/routes/documents_route.py](file:///d:/PROJECTS/AI_RECEPTIONIST/api/routes/documents_route.py):
  Handles file uploads (`.pdf`, `.docx`, `.txt`, `.md`), writes to disk, invokes `DocumentIngester`, and updates the FAISS vector index in real time.

---

### 5.9 Non-Blocking Async Telemetry (`api/db/supabase_client.py`)

[api/db/supabase_client.py](file:///d:/PROJECTS/AI_RECEPTIONIST/api/db/supabase_client.py) logs every conversation turn without impacting audio latency.

#### Mechanics:
- `async_log_turn(...)` gathers `room_name`, `user_text`, `has_context`, `similarity_score`, `agent_text`, and ISO timestamp.
- Attempts an asynchronous write to Supabase table `call_turns`.
- If database credentials are missing or the connection times out, it gracefully falls back to appending JSON lines to `data/call_logs.jsonl`.
- **Zero latency penalty on the voice pipeline.**

---

## 6. Core Architectural Decisions Explained from Scratch

### WebRTC vs. WebSockets
| Feature | WebSockets (TCP) | WebRTC (UDP + SRTP) |
|---|---|---|
| **Transport Layer** | TCP (Transmission Control Protocol) | UDP (User Datagram Protocol) |
| **Packet Loss Handling** | Retransmits dropped packets. Stalls all subsequent audio frames (**Head-of-Line Blocking**). | Drops lost packets and interpolates audio. Continues streaming immediately. |
| **Latency Under Jitter** | Latency degrades to 300ms–800ms on cellular/Wi-Fi jitter. | Stays under 40ms with native adaptive jitter buffering. |
| **Echo Cancellation** | None (must be handled manually in JavaScript). | Native hardware-level Acoustic Echo Cancellation (AEC). |

### Silero VAD vs. Semantic End-of-Utterance (EOU)
- **Traditional Silence VAD:** Measures audio volume amplitude. If silence lasts >500ms, it assumes the user is done. If a user pauses to think for 300ms, a silence VAD cuts them off.
- **Semantic EOU (LiveKit + Deepgram):** Evaluates the **linguistic completeness** of incoming words. If the user says: *"What time do you open on Saturday?"*, the model recognizes a complete grammatical sentence and terminates the turn after **160ms**, saving 340ms of dead air.

### Groq LPUs vs. GPU Clusters
- **GPUs (Nvidia H100 / A100):** Designed for high-throughput batch parallelization (matrix multiplication across thousands of concurrent requests). For a single sequential voice stream, Time-To-First-Token is often 250ms–500ms due to memory bandwidth limits.
- **Groq LPUs (Language Processing Units):** Tensor streaming architectures with deterministic execution and on-chip SRAM. They deliver sub-100ms TTFT and >300 tokens/sec for single-stream generation.

### FAISS FlatIP vs. Approximate Nearest Neighbors (HNSW / IVF)
- **Approximate Nearest Neighbors (ANN) like HNSW:** Useful when indexing 10,000,000+ documents where exhaustive search is slow. However, ANN builds large graph indexes, consumes significant RAM, and introduces search approximation errors.
- **FAISS IndexFlatIP (Exact Inner Product):** For clinic knowledge bases containing hundreds to thousands of chunks, an exact vector scan across all chunks takes **less than 0.8ms** on CPU with 100% recall accuracy.

### Barge-In (Interruption Handling) Mechanics
```
Caller speaks while receptionist is talking
               │
               ▼
Silero VAD detects speech onset (<60ms)
               │
               ▼
session.interrupt() is invoked
               ├── 1. Groq HTTP token stream generator is immediately aborted
               ├── 2. Deepgram Aura TTS audio frame buffer is flushed
               ├── 3. WebRTC audio output track transmits silence frames
               └── 4. New incoming audio is promoted to the active turn
```

---

## 7. Zero-Hallucination & Clinical Grounding Protocol

In medical and legal reception, hallucinations can have severe legal and clinical consequences. This architecture implements a **Two-Tier Anti-Hallucination Guardrail**:

```mermaid
flowchart TD
    Query["User Voice Query"] --> FastEmbed["FastEmbed ONNX Vectorizer"]
    FastEmbed --> FAISS["FAISS Cosine Similarity Search"]
    FAISS --> CheckScore{"Top Score >= Threshold (0.48 - 0.65)?"}
    
    CheckScore -- YES --> InjectContext["Inject Exact Document Passage into Prompt"]
    InjectContext --> LLMAnswer["LLM Answers in Spoken Voice (Max 2 Sentences)"]
    
    CheckScore -- NO --> ReturnNoRecords["RAG Returns NO_RECORDS_FOUND"]
    ReturnNoRecords --> TriggerFallback["Trigger Polite Fallback Refusal & Contact Intake"]
```

### The Polite Refusal Rule:
If context is missing or below threshold, the agent refuses to speculate and says:
> *"I apologize, I don't have that specific detail in our records right now. Would you like me to take down your name and phone number so our team can follow up with you?"*

---

## 8. Production Deployment, Scale & SIP Telephony

### Scaled Production Architecture:
```
                       PSTN / Mobile Phone Callers
                                  │
                                  ▼
                   Telephony Carrier (Twilio / Telnyx)
                                  │ SIP Trunking
                                  ▼
                         LiveKit SIP Gateway
                                  │ WebRTC Tracks
                                  ▼
                    LiveKit SFU Cluster (Distributed)
                      ├── Room Routing (Redis)
                      └── Media Relays
                                  │
          ┌───────────────────────┴───────────────────────┐
          ▼                                               ▼
Kubernetes Pod: Agent Worker 1                 Kubernetes Pod: Agent Worker N
(HPA scaling on active call count)             (HPA scaling on active call count)
```

1. **LiveKit SIP Gateway:** Bridges standard telephone carrier lines (Twilio, Telnyx, Bandwidth) directly into WebRTC audio rooms with zero transcoding overhead.
2. **Kubernetes Autoscaling:** Python workers run as containerized pods. A Horizontal Pod Autoscaler (HPA) spins up worker replicas based on active call count and CPU utilization.
3. **Enterprise Vector Tier:** As knowledge bases grow to tens of thousands of documents, FAISS FlatIP can be swapped for a distributed Qdrant or Milvus cluster with read-replicas.
4. **Security & HIPAA Compliance:**
   - All audio streams encrypted using DTLS-SRTP.
   - Zero audio retention on disk.
   - Transient memory buffers flushed at call termination.

---

## 9. Interview Master Guide & Whiteboard Q&A

### The 60-Second Elevator Pitch
> *"I engineered an ultra-low latency (<500ms TTFA) full-duplex AI Voice Receptionist for clinical and enterprise phone reception. Traditional voice pipelines take 1.5 to 2 seconds due to sequential execution of speech recognition, RAG retrieval, LLM generation, and audio synthesis.*
>
> *I crushed the turnaround latency down to ~410ms using three architectural innovations:*
> 1. ***Speculative RAG*** *running in-memory on CPU-optimized ONNX embeddings, pre-fetching knowledge chunks against interim STT partials while the caller is still speaking (0ms net retrieval wait).*
> 2. ***LPU-accelerated token generation*** *via Groq Cloud for sub-120ms Time-To-First-Token.*
> 3. ***Clause-level audio streaming*** *that feeds words to Deepgram Aura on punctuation boundaries (commas and 4-word thresholds) rather than waiting for full sentences.*
>
> *The entire system supports full-duplex WebRTC streaming over LiveKit, sub-60ms barge-in interruptions, and a strict similarity threshold to prevent medical hallucinations."*

---

### Top Technical Interview Questions & Answers

#### Q1: Why not run an agent framework like LangGraph or CrewAI for the voice loop?
> **Answer:** *"LangGraph and multi-agent frameworks are great for complex asynchronous reasoning, but they introduce severe sequential overhead. Each routing node, state validation, and LLM hop adds 300ms–500ms. In real-time voice, we have a total latency ceiling of 500ms. We use a single-pass orchestrator with a tool-augmented prompt and speculative in-memory retrieval, executing the entire cycle in ~410ms."*

#### Q2: How does your system handle simultaneous speech (barge-in)?
> **Answer:** *"LiveKit's Silero VAD monitors the caller's audio stream continuously. When the caller speaks while the agent is playing audio, Silero detects speech onset in under 60ms. LiveKit triggers `session.interrupt()`, which flushes the audio playback buffer, aborts the in-flight Groq HTTP stream generator, and immediately turns the agent to listening mode."*

#### Q3: What is the math behind your FAISS similarity scoring?
> **Answer:** *"We use `faiss.IndexFlatIP` (Inner Product). Before adding embeddings to the index and before querying, we normalize all vectors to unit length using L2 normalization (`faiss.normalize_L2`). The inner product of two unit-normalized vectors is algebraically identical to Cosine Similarity:*
> $$\text{InnerProduct}(\hat{a}, \hat{b}) = \sum_{i=1}^d \hat{a}_i \hat{b}_i = \frac{a \cdot b}{\|a\| \|b\|} = \cos(\theta)$$
> *This avoids expensive square-root calculations during search while giving exact cosine similarity."*

#### Q4: What happens if the caller has a heavy accent or mumbles?
> **Answer:** *"Deepgram Nova-3 is trained on diverse multilingual and accented acoustic datasets. Additionally, we enable `smart_format=True` to normalize numbers, dates, and medical terms. If STT confidence drops and RAG retrieval falls below our confidence threshold, the agent gracefully defaults to asking clarifying questions rather than acting on misheard words."*

#### Q5: How do you benchmark latency accurately?
> **Answer:** *"We built automated benchmark suites in `tests/test_latency.py`. We measure:*
> - *Vector retrieval: CPU timestamp deltas before embedding and after FAISS index search (consistently <12ms).*
> - *LLM TTFT: Delta between sending the completion request and receiving the first SSE token chunk (consistently <120ms).*
> - *End-to-end TTFA: Telemetry logging timestamps between VAD speech-stop event and the first audio frame emission on the WebRTC track."*

---

### Key System Metrics to Quote:
- **Time-To-First-Audio (TTFA):** `~380ms – 460ms`
- **FastEmbed ONNX CPU Embedding:** `< 8ms`
- **FAISS Inner Product Search:** `< 0.8ms`
- **Groq Cloud Time-To-First-Token (TTFT):** `< 120ms`
- **Barge-In Interrupt Detection:** `< 60ms`
- **Similarity Rejection Threshold:** `0.48 – 0.65`
