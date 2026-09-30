"""FastAPI Server exposing real-time SSE streaming for the Agentic Workflow."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, AsyncGenerator

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.framework.llm_client import LLMClient
from src.framework.workflow import WorkflowState
from src.agents.supervisor import SupervisorAgent

app = FastAPI(title="GenAI Agentic Workflow Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check() -> dict:
    """Service health check."""
    return {"status": "ok", "service": "agentic-workflow-api"}


@app.get("/api/stream")
async def stream_workflow(
    goal: str = Query(..., description="High-level goal for the agents"),
    simulation: bool = Query(True, description="Force simulation mode"),
    model: str = Query("gemini-3.8-flash", description="Gemini model"),
) -> StreamingResponse:
    """Stream live agent execution telemetry via Server-Sent Events (SSE)."""

    async def event_generator() -> AsyncGenerator[str, None]:
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def on_event(event_type: str, data: Any) -> None:
            # Thread-safe dispatch from sync supervisor execution to async queue
            loop.call_soon_threadsafe(queue.put_nowait, (event_type, data))

        def run_in_thread() -> None:
            try:
                client = LLMClient(model=model, force_simulation=simulation)
                supervisor = SupervisorAgent(llm_client=client)
                supervisor.execute_workflow(goal=goal, on_event=on_event)
            except Exception as e:
                loop.call_soon_threadsafe(
                    queue.put_nowait, ("error", {"message": str(e)})
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("DONE", None))

        # Launch workflow in background worker thread
        loop.run_in_executor(None, run_in_thread)

        while True:
            event_type, data = await queue.get()
            if event_type == "DONE":
                yield f"event: done\ndata: {json.dumps({'status': 'finished'})}\n\n"
                break

            # Serialize data safely
            serialized_data: Any = None
            if hasattr(data, "model_dump"):
                serialized_data = data.model_dump()
            elif isinstance(data, dict):
                # Recursively serialize any nested models in dict
                clean_dict = {}
                for k, v in data.items():
                    if hasattr(v, "model_dump"):
                        clean_dict[k] = v.model_dump()
                    else:
                        clean_dict[k] = v
                serialized_data = clean_dict
            else:
                serialized_data = str(data)

            payload = json.dumps({"type": event_type, "payload": serialized_data})
            yield f"data: {payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# Serve static web interface
static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server:app", host="127.0.0.1", port=8000, reload=True)
