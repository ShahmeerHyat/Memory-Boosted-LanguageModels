import json
import uuid
import time
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from models import MessageRole
from conversation_manager import ConversationManager
from llm_engine import LLMEngine

app = FastAPI(title="PharmaChat API", version="1.0.0", description="Local LLM-powered pharmacy information chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

conv_manager = ConversationManager()
llm_engine = LLMEngine()

# Serve frontend static files
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Track active WebSocket connections for concurrency support
active_connections: dict[str, WebSocket] = {}


# ─── REST Endpoints ───────────────────────────────────────────────────────────

@app.get("/")
async def serve_ui():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/health")
async def health():
    llm_ok = await llm_engine.health_check()
    models = await llm_engine.list_models() if llm_ok else []
    return {
        "status": "ok",
        "llm": "online" if llm_ok else "offline",
        "available_models": models,
        "active_sessions": len(conv_manager.sessions),
        "active_connections": len(active_connections),
    }


@app.post("/session/new")
async def new_session():
    session_id = str(uuid.uuid4())
    conv_manager.get_or_create_session(session_id)
    return {"session_id": session_id}


@app.delete("/session/{session_id}")
async def reset_session(session_id: str):
    conv_manager.reset_session(session_id)
    return {"status": "reset", "session_id": session_id}


@app.get("/session/{session_id}/stats")
async def session_stats(session_id: str):
    return conv_manager.get_session_stats(session_id)


@app.get("/session/{session_id}/history")
async def session_history(session_id: str):
    history = conv_manager.get_or_create_session(session_id)
    return {
        "session_id": session_id,
        "messages": [{"role": m.role.value, "content": m.content} for m in history],
    }


# ─── WebSocket Endpoint ───────────────────────────────────────────────────────

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    connection_id = str(uuid.uuid4())
    active_connections[connection_id] = websocket

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "content": "Invalid JSON payload"})
                continue

            session_id = payload.get("session_id", "").strip()
            message = payload.get("message", "").strip()

            if not session_id:
                await websocket.send_json({"type": "error", "content": "Missing session_id"})
                continue
            if not message:
                await websocket.send_json({"type": "error", "content": "Empty message"})
                continue

            # Check LLM is reachable before processing
            if not await llm_engine.health_check():
                await websocket.send_json({
                    "type": "error",
                    "content": "LLM engine offline. Make sure Ollama is running."
                })
                continue

            # Add user message to session history
            conv_manager.add_message(session_id, MessageRole.USER, message)

            # Build context-managed prompt (sliding window algorithm)
            messages = conv_manager.build_messages(session_id)

            # Signal stream start
            t_start = time.time()
            await websocket.send_json({"type": "start", "session_id": session_id})

            # Stream tokens
            full_response = ""
            async for token in llm_engine.stream_response(messages):
                full_response += token
                await websocket.send_json({"type": "token", "content": token})

            latency_ms = round((time.time() - t_start) * 1000)

            # Store assistant response in session history
            conv_manager.add_message(session_id, MessageRole.ASSISTANT, full_response)

            # Signal stream end with metadata
            stats = conv_manager.get_session_stats(session_id)
            await websocket.send_json({
                "type": "end",
                "session_id": session_id,
                "latency_ms": latency_ms,
                "turn_count": stats["turn_count"],
            })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "content": str(e)})
        except Exception:
            pass
    finally:
        active_connections.pop(connection_id, None)
