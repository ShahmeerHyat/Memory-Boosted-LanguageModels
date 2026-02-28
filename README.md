# PharmaChat — MediPlus Pharmacy Information Assistant

> CS 4063 Natural Language Processing — Assignment 2
> A fully local, production-style conversational AI system.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Browser (Vercel)                   │
│              ChatGPT-style Web UI                   │
│         WebSocket + REST over HTTP/HTTPS            │
└────────────────────┬────────────────────────────────┘
                     │ WebSocket /ws/chat
┌────────────────────▼────────────────────────────────┐
│              FastAPI Backend (Docker)               │
│  ┌──────────────────────────────────────────────┐   │
│  │           Conversation Manager               │   │
│  │   Sliding Window + Importance Filter Algo    │   │
│  │   - Session store (in-memory dict)           │   │
│  │   - Context pruning: keep last 8 turns       │   │
│  │   - Importance scoring: retain allergy/med   │   │
│  │     keywords from older turns                │   │
│  └──────────────────┬───────────────────────────┘   │
└─────────────────────┼───────────────────────────────┘
                      │ HTTP (Ollama API)
┌─────────────────────▼───────────────────────────────┐
│            Ollama (Local LLM Engine)                │
│         Model: Qwen2.5-7B-Instruct (GGUF)           │
│         GPU-accelerated via CUDA (RTX 5060)         │
└─────────────────────────────────────────────────────┘
```

---

## Business Use Case

**PharmaChat** is a pharmacy information assistant for MediPlus Pharmacy. It:
- Answers questions about medications, dosages, and side effects
- Explains drug interactions
- Guides users on proper medication administration
- Redirects emergencies to professional services
- Provides **information only** — not diagnosis or prescription

---

## Context Window Management Algorithm

The core NLP challenge (no RAG, no tools) is handled by a **Sliding Window + Importance Filter**:

```
Given conversation history H = [m1, m2, ..., mn]:

If len(H) ≤ 8:
    Send: [system_prompt] + H

Else:
    old_turns  = H[:-8]
    recent     = H[-8:]

    Score each old turn by keyword presence:
      keywords = {allerg, pregnant, age, diabete, kidney, ...}
      score(m) = count of keywords in m.content

    top_important = top-3 old turns by score (if score > 0)

    context_block = compress top_important into 1 system message

    Send: [system_prompt] + [context_block?] + recent
```

This ensures:
1. Recent context is always fully preserved
2. Critical patient facts (allergies, conditions, meds) survive context pruning
3. Unimportant small-talk turns are safely dropped

---

## Model Selection

| Model | Size | Quantization | VRAM | Why |
|-------|------|-------------|------|-----|
| **Qwen2.5-7B-Instruct** | 7B | Q4_K_M | ~4.5 GB | Best instruction-following, fast on GPU |
| Phi-3.5-mini | 3.8B | Q4_K_M | ~2.5 GB | Fallback if VRAM limited |

---

## Setup Instructions

### Prerequisites
- [Ollama](https://ollama.com) installed
- Python 3.11+
- Docker (optional)
- RTX GPU recommended (CPU works, slower)

### 1. Pull the Model
```bash
ollama pull qwen2.5:7b
```

### 2. Run Backend (Local)
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 3. Run Backend (Docker)
```bash
docker-compose up --build
```

### 4. Open Frontend
Open `frontend/index.html` in your browser, or deploy to Vercel.

### 5. Deploy Frontend to Vercel
```bash
cd frontend
npx vercel --prod
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | LLM status + active sessions |
| POST | `/session/new` | Create new session |
| DELETE | `/session/{id}` | Reset session history |
| GET | `/session/{id}/stats` | Turn count, window info |
| GET | `/session/{id}/history` | Full message history |
| WS | `/ws/chat` | Streaming chat endpoint |

### WebSocket Message Format

**Send:**
```json
{ "session_id": "uuid", "message": "your question" }
```

**Receive (stream):**
```json
{ "type": "start" }
{ "type": "token", "content": "..." }
{ "type": "end", "latency_ms": 1240, "turn_count": 4 }
```

---

## Performance Benchmarks

| Metric | Value |
|--------|-------|
| First token latency | ~800ms (GPU) |
| Tokens per second | ~25-35 tok/s (RTX 5060) |
| Concurrent sessions | Tested up to 10 simultaneous |
| Context window | 4096 tokens (Qwen2.5-7B) |
| Max history kept | 8 turns verbatim + importance summary |

---

## Known Limitations
- Backend must run locally (Ollama not cloud-deployable in this setup)
- Vercel frontend requires backend URL update for production
- In-memory session store resets on server restart
- Not a substitute for professional medical advice
