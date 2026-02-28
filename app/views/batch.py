from __future__ import annotations

from datetime import datetime
from threading import Lock, Thread
from queue import Queue
from typing import Any

from flask import Blueprint, current_app, jsonify, request

from app.connectors import Connector, MLXConnector, OllamaConnector, TensorRTConnector
from app.extensions import db
from app.models import Agent, BatchJob, Conversation, Message

batch_bp = Blueprint("batch", __name__, url_prefix="/api")

_batch_queue: Queue[int] = Queue()
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


def _run_single_conversation(batch: BatchJob) -> Conversation:
    conversation = Conversation(
        agent1_id=batch.agent1_id,
        agent2_id=batch.agent2_id,
        ttl=batch.ttl,
        random_seed=batch.id,
        status="running",
    )
    db.session.add(conversation)
    db.session.flush()
    db.session.add(Message(conversation_id=conversation.id, sender_role="user", content=batch.prompt))

    current_text = batch.prompt
    for turn in range(max(1, batch.ttl)):
        use_first_agent = turn % 2 == 0
        agent = batch.agent1 if use_first_agent else batch.agent2
        sender = "agent1" if use_first_agent else "agent2"
        connector = _connector_for_agent(agent)
        message_payload = [
            {"role": "system", "content": agent.system_prompt},
            {"role": "user", "content": current_text},
        ]
        text = connector.chat(
            message_payload,
            {
                "model": agent.model.model_name,
                "max_tokens": agent.max_tokens,
                "temperature": agent.temperature,
                "seed": batch.id,
            },
        )
        db.session.add(
            Message(
                conversation_id=conversation.id,
                sender_role=sender,
                agent_id=agent.id,
                content=text,
            )
        )
        current_text = text

    conversation.status = "finished"
    conversation.finished_at = datetime.utcnow()
    db.session.commit()
    return conversation


def _process_batch(app, batch_id: int) -> None:
    with app.app_context():
        batch = db.session.get(BatchJob, batch_id)
        if batch is None:
            return
        batch.status = "running"
        batch.start_time = datetime.utcnow()
        db.session.commit()

        conversations = []
        try:
            for _ in range(batch.num_runs):
                batch = db.session.get(BatchJob, batch_id)
                if batch is None or batch.status == "stopped":
                    break
                conversations.append(_run_single_conversation(batch))

            batch = db.session.get(BatchJob, batch_id)
            if batch is None:
                return
            if batch.status != "stopped":
                batch.status = "completed"
            batch.end_time = datetime.utcnow()
            elapsed = (batch.end_time - batch.start_time).total_seconds() if batch.start_time else 0.0
            total_messages = sum(len(conv.messages) for conv in conversations)
            batch.summary = {
                "completed_runs": len(conversations),
                "total_runs": batch.num_runs,
                "progress_pct": round((len(conversations) / max(1, batch.num_runs)) * 100, 2),
                "total_messages": total_messages,
                "total_time_seconds": round(elapsed, 2),
                "avg_tokens_per_sec": round((total_messages * 20) / elapsed, 2) if elapsed else 0,
                "conversation_ids": [conv.id for conv in conversations],
            }
            db.session.commit()
        except Exception as exc:
            db.session.rollback()
            batch = db.session.get(BatchJob, batch_id)
            if batch is not None:
                batch.status = "failed"
                batch.end_time = datetime.utcnow()
                batch.summary = {"error": str(exc)}
                db.session.commit()


def _batch_worker(app) -> None:
    while True:
        batch_id = _batch_queue.get()
        try:
            _process_batch(app, batch_id)
        finally:
            _batch_queue.task_done()


def init_batch_worker(app) -> None:
    global _worker_thread
    with _worker_lock:
        if _worker_thread is None or not _worker_thread.is_alive():
            _worker_thread = Thread(target=_batch_worker, args=(app,), daemon=True)
            _worker_thread.start()


@batch_bp.post("/batch")
def create_batch() -> Any:
    payload = request.get_json(silent=True) or {}
    required = {"agent1_id", "agent2_id", "prompt", "ttl", "num_runs"}
    missing = sorted(field for field in required if field not in payload)
    if missing:
        return jsonify({"success": False, "error": f"Missing required fields: {', '.join(missing)}"}), 400

    try:
        agent1_id = int(payload["agent1_id"])
        agent2_id = int(payload["agent2_id"])
        ttl = max(1, int(payload["ttl"]))
        num_runs = max(1, int(payload["num_runs"]))
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "agent1_id, agent2_id, ttl, num_runs must be integers"}), 400

    agent1 = db.session.get(Agent, agent1_id)
    agent2 = db.session.get(Agent, agent2_id)
    if agent1 is None or agent2 is None:
        return jsonify({"success": False, "error": "agent1_id or agent2_id does not exist"}), 404

    batch = BatchJob(
        agent1_id=agent1_id,
        agent2_id=agent2_id,
        prompt=str(payload["prompt"]),
        ttl=ttl,
        num_runs=num_runs,
        status="pending",
        summary={"completed_runs": 0, "total_runs": num_runs, "progress_pct": 0},
    )
    db.session.add(batch)
    db.session.commit()

    if current_app.config.get("TESTING"):
        _process_batch(current_app._get_current_object(), batch.id)
    else:
        _batch_queue.put(batch.id)

    return jsonify({"success": True, "data": {"batch_id": batch.id}}), 201


@batch_bp.post("/batch/<int:batch_id>/stop")
def stop_batch(batch_id: int) -> Any:
    batch = db.session.get(BatchJob, batch_id)
    if batch is None:
        return jsonify({"success": False, "error": "Batch job not found"}), 404
    if batch.status in {"completed", "failed"}:
        return jsonify({"success": False, "error": "Batch job already finished"}), 400

    batch.status = "stopped"
    batch.end_time = datetime.utcnow()
    db.session.commit()
    return jsonify({"success": True, "data": {"message": "Batch job stopping requested"}})


@batch_bp.get("/batch")
def list_batch_jobs() -> Any:
    jobs = BatchJob.query.order_by(BatchJob.id.desc()).all()
    payload = []
    for job in jobs:
        summary = job.summary or {}
        total_time = summary.get("total_time_seconds")
        if total_time is None and job.start_time and job.end_time:
            total_time = round((job.end_time - job.start_time).total_seconds(), 2)
        payload.append(
            {
                "id": job.id,
                "agent1_id": job.agent1_id,
                "agent2_id": job.agent2_id,
                "prompt": job.prompt,
                "ttl": job.ttl,
                "num_runs": job.num_runs,
                "status": job.status,
                "summary": summary,
                "total_time_seconds": total_time,
            }
        )
    return jsonify({"success": True, "data": payload})


@batch_bp.get("/batch/<int:batch_id>")
def get_batch_job(batch_id: int) -> Any:
    batch = db.session.get(BatchJob, batch_id)
    if batch is None:
        return jsonify({"success": False, "error": "Batch job not found"}), 404

    return jsonify(
        {
            "success": True,
            "data": {
                "id": batch.id,
                "agent1_id": batch.agent1_id,
                "agent2_id": batch.agent2_id,
                "prompt": batch.prompt,
                "ttl": batch.ttl,
                "num_runs": batch.num_runs,
                "status": batch.status,
                "summary": batch.summary,
                "start_time": batch.start_time.isoformat() if batch.start_time else None,
                "end_time": batch.end_time.isoformat() if batch.end_time else None,
            },
        }
    )
