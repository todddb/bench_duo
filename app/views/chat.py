from __future__ import annotations

from datetime import datetime
from queue import Queue
from threading import Lock, Thread
from typing import Any

from flask import Blueprint, current_app, jsonify, request
from app.connectors import Connector, MLXConnector, OllamaConnector, TensorRTConnector
from app.extensions import db, socketio
from app.models import Agent, Conversation, Message

chat_bp = Blueprint("chat", __name__, url_prefix="/api")

_chat_queue: Queue[dict[str, Any]] = Queue()
_worker_thread: Thread | None = None
_worker_lock = Lock()


def _connector_for_agent(agent: Agent) -> Connector:
    backend = agent.model.backend.lower()
    if backend == "ollama":
        return OllamaConnector(host=agent.model.host, port=agent.model.port)
    if backend == "mlx":
        return MLXConnector()
    if backend in {"tensorrt", "tensorrt_llm"}:
        return TensorRTConnector()
    raise ValueError(f"Unsupported backend: {agent.model.backend}")


def _conversation_stats(conversation_id: int) -> dict[str, Any]:
    total_messages = Message.query.filter_by(conversation_id=conversation_id).count()
    return {"total_messages": total_messages}


def _chat_worker(app) -> None:
    while True:
        task = _chat_queue.get()
        try:
            _process_chat_task(app, task)
        finally:
            _chat_queue.task_done()


def _emit(event: str, payload: dict[str, Any], sid: str | None, namespace: str) -> None:
    socketio.emit(event, payload, to=sid, namespace=namespace)


def _process_chat_task(app, task: dict[str, Any]) -> None:
    with app.app_context():
        conversation = db.session.get(Conversation, task["conversation_id"])
        if conversation is None:
            return

        prompt = task["prompt"]
        current_text = prompt
        sid = task.get("sid")
        namespace = task.get("namespace", "/")
        ttl = max(1, int(conversation.ttl))

        turns_completed = 0
        for turn in range(ttl):
            use_first_agent = turn % 2 == 0
            agent = conversation.agent1 if use_first_agent else conversation.agent2
            sender = "agent1" if use_first_agent else "agent2"

            connector = _connector_for_agent(agent)
            message_payload = [
                {"role": "system", "content": agent.system_prompt},
                {"role": "user", "content": current_text},
            ]
            settings = {
                "model": agent.model.model_name,
                "max_tokens": agent.max_tokens,
                "temperature": agent.temperature,
                "seed": conversation.random_seed,
            }
            text = connector.chat(message_payload, settings)

            db.session.add(
                Message(
                    conversation_id=conversation.id,
                    sender_role=sender,
                    agent_id=agent.id,
                    content=text,
                )
            )
            db.session.commit()

            turns_completed += 1
            is_done = turns_completed >= ttl
            _emit(
                "chat_message",
                {
                    "conversation_id": conversation.id,
                    "sender": sender,
                    "text": text,
                    "done": is_done,
                },
                sid=sid,
                namespace=namespace,
            )
            current_text = text

        conversation.status = "finished"
        conversation.finished_at = datetime.utcnow()
        db.session.commit()

        _emit(
            "chat_end",
            {
                "conversation_id": conversation.id,
                "status": conversation.status,
                "stats": _conversation_stats(conversation.id),
            },
            sid=sid,
            namespace=namespace,
        )


def init_chat_worker(app) -> None:
    global _worker_thread
    with _worker_lock:
        if _worker_thread is None or not _worker_thread.is_alive():
            _worker_thread = Thread(target=_chat_worker, args=(app,), daemon=True)
            _worker_thread.start()


@socketio.on("start_chat", namespace="/")
def handle_start_chat(data: dict[str, Any] | None) -> tuple[dict[str, Any], int] | dict[str, Any]:
    payload = data or {}
    required = {"agent1_id", "agent2_id", "prompt", "ttl", "seed"}
    missing = sorted(k for k in required if k not in payload)
    if missing:
        return {"error": f"Missing required fields: {', '.join(missing)}"}, 400

    agent1 = db.session.get(Agent, int(payload["agent1_id"]))
    agent2 = db.session.get(Agent, int(payload["agent2_id"]))
    if agent1 is None or agent2 is None:
        return {"error": "agent1_id or agent2_id does not exist"}, 404

    ttl = max(1, int(payload["ttl"]))
    conversation = Conversation(
        agent1_id=agent1.id,
        agent2_id=agent2.id,
        ttl=ttl,
        random_seed=int(payload["seed"]),
        status="running",
    )
    db.session.add(conversation)
    db.session.flush()
    db.session.add(
        Message(
            conversation_id=conversation.id,
            sender_role="user",
            content=str(payload["prompt"]),
        )
    )
    db.session.commit()

    task = {
        "conversation_id": conversation.id,
        "prompt": str(payload["prompt"]),
        "sid": request.sid,
        "namespace": request.namespace,
    }
    if current_app.config.get("TESTING"):
        _process_chat_task(current_app._get_current_object(), task)
    else:
        _chat_queue.put(task)

    return {"conversation_id": conversation.id}


@chat_bp.get("/conversations/<int:conversation_id>")
def get_conversation(conversation_id: int) -> Any:
    conversation = db.session.get(Conversation, conversation_id)
    if conversation is None:
        return jsonify({"success": False, "error": "Conversation not found"}), 404

    return jsonify(
        {
            "success": True,
            "data": {
                "id": conversation.id,
                "status": conversation.status,
                "agent1_id": conversation.agent1_id,
                "agent2_id": conversation.agent2_id,
                "ttl": conversation.ttl,
                "random_seed": conversation.random_seed,
                "finished_at": conversation.finished_at.isoformat() if conversation.finished_at else None,
                "stats": _conversation_stats(conversation.id),
            },
        }
    )
