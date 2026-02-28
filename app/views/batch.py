from __future__ import annotations

from datetime import datetime
from queue import Queue
from threading import Lock, Thread
from time import perf_counter
from typing import Any

from flask import Blueprint, current_app, jsonify, request

from app.connectors import Connector, MLXConnector, OllamaConnector, TensorRTConnector
from app.extensions import db
from app.models import Agent, BatchJob, Conversation, Message

batch_bp = Blueprint("batch", __name__, url_prefix="/api")

_batch_queue: Queue[dict[str, Any]] = Queue()
_batch_worker_thread: Thread | None = None
_batch_worker_lock = Lock()


def _connector_for_agent(agent: Agent) -> Connector:
    backend = agent.model.backend.lower()
    if backend == "ollama":
        return OllamaConnector(host=agent.model.host, port=agent.model.port)
    if backend == "mlx":
        return MLXConnector()
    if backend in {"tensorrt", "tensorrt_llm"}:
        return TensorRTConnector()
    raise ValueError(f"Unsupported backend: {agent.model.backend}")


def _count_tokens(text: str) -> int:
    return max(1, len(text.split()))


def _run_single_conversation(batch: BatchJob, run_seed: int | None) -> tuple[float, int]:
    conversation = Conversation(
        agent1_id=batch.agent1_id,
        agent2_id=batch.agent2_id,
        ttl=batch.ttl,
        random_seed=run_seed,
        status="running",
    )
    db.session.add(conversation)
    db.session.flush()

    db.session.add(
        Message(
            conversation_id=conversation.id,
            sender_role="user",
            content=batch.prompt,
            tokens=_count_tokens(batch.prompt),
        )
    )
    db.session.commit()

    started_at = perf_counter()
    current_text = batch.prompt
    total_tokens = _count_tokens(batch.prompt)
    ttl = max(1, int(batch.ttl))

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
            "seed": run_seed,
        }

        text = connector.chat(message_payload, settings)
        token_count = _count_tokens(text)
        total_tokens += token_count

        db.session.add(
            Message(
                conversation_id=conversation.id,
                sender_role=sender,
                agent_id=agent.id,
                content=text,
                tokens=token_count,
            )
        )
        db.session.commit()
        current_text = text

    conversation.status = "finished"
    conversation.finished_at = datetime.utcnow()
    db.session.commit()

    return perf_counter() - started_at, total_tokens


def _process_batch_job(app, task: dict[str, Any]) -> None:
    with app.app_context():
        batch = db.session.get(BatchJob, task["batch_job_id"])
        if batch is None:
            return

        if batch.status == "cancelled":
            return

        batch.status = "running"
        batch.start_time = batch.start_time or datetime.utcnow()
        db.session.commit()

        total_elapsed = 0.0
        total_tokens = 0

        for i in range(batch.completed_runs, batch.num_runs):
            db.session.refresh(batch)
            if batch.cancel_requested:
                batch.status = "cancelled"
                batch.end_time = datetime.utcnow()
                break

            run_seed = None if batch.seed is None else batch.seed + i
            run_elapsed, run_tokens = _run_single_conversation(batch, run_seed)
            total_elapsed += run_elapsed
            total_tokens += run_tokens

            db.session.refresh(batch)
            batch.completed_runs = i + 1
            batch.total_elapsed_seconds = (batch.total_elapsed_seconds or 0.0) + run_elapsed
            batch.total_tokens = (batch.total_tokens or 0) + run_tokens
            if batch.cancel_requested:
                batch.status = "cancelled"
                batch.end_time = datetime.utcnow()
                db.session.commit()
                break
            if batch.completed_runs >= batch.num_runs:
                batch.status = "finished"
                batch.end_time = datetime.utcnow()
            db.session.commit()

        if batch.status == "running":
            batch.status = "finished"
            batch.end_time = datetime.utcnow()
            db.session.commit()


def _batch_worker() -> None:
    while True:
        task = _batch_queue.get()
        try:
            _process_batch_job(task["app"], task)
        finally:
            _batch_queue.task_done()


def init_batch_worker(app) -> None:
    global _batch_worker_thread
    with _batch_worker_lock:
        if _batch_worker_thread is None or not _batch_worker_thread.is_alive():
            _batch_worker_thread = Thread(target=_batch_worker, daemon=True)
            _batch_worker_thread.start()


def _batch_job_response(batch: BatchJob) -> dict[str, Any]:
    total = max(1, int(batch.num_runs))
    elapsed = float(batch.total_elapsed_seconds or 0.0)
    completed = int(batch.completed_runs or 0)
    avg_time = elapsed / completed if completed else None
    tokens_per_sec = (float(batch.total_tokens) / elapsed) if elapsed > 0 else None

    return {
        "id": batch.id,
        "status": batch.status,
        "completed": completed,
        "total": total,
        "agent1_id": batch.agent1_id,
        "agent2_id": batch.agent2_id,
        "prompt": batch.prompt,
        "prompt_snippet": batch.prompt[:80],
        "ttl": batch.ttl,
        "seed": batch.seed,
        "avg_time": avg_time,
        "tokens_per_sec": tokens_per_sec,
        "time_elapsed": elapsed,
        "created_at": batch.created_at.isoformat() if batch.created_at else None,
        "start_time": batch.start_time.isoformat() if batch.start_time else None,
        "end_time": batch.end_time.isoformat() if batch.end_time else None,
    }


@batch_bp.post("/batch_jobs")
def create_batch_job() -> Any:
    payload = request.get_json(silent=True) or {}
    required = {"agent1_id", "agent2_id", "prompt", "ttl", "num_runs", "seed"}
    missing = sorted(k for k in required if k not in payload)
    if missing:
        return jsonify({"success": False, "error": f"Missing required fields: {', '.join(missing)}"}), 400

    agent1 = db.session.get(Agent, int(payload["agent1_id"]))
    agent2 = db.session.get(Agent, int(payload["agent2_id"]))
    if agent1 is None or agent2 is None:
        return jsonify({"success": False, "error": "agent1_id or agent2_id does not exist"}), 404

    batch = BatchJob(
        agent1_id=agent1.id,
        agent2_id=agent2.id,
        prompt=str(payload["prompt"]),
        ttl=max(1, int(payload["ttl"])),
        num_runs=max(1, int(payload["num_runs"])),
        seed=int(payload["seed"]),
        status="queued",
    )
    db.session.add(batch)
    db.session.commit()

    task = {"batch_job_id": batch.id, "app": current_app._get_current_object()}
    inline_processing = current_app.config.get("BATCH_INLINE_PROCESSING", current_app.config.get("TESTING", False))
    if inline_processing:
        _process_batch_job(current_app._get_current_object(), task)
    else:
        _batch_queue.put(task)

    return jsonify({"success": True, "data": _batch_job_response(batch)}), 201


@batch_bp.get("/batch_jobs")
def list_batch_jobs() -> Any:
    jobs = BatchJob.query.order_by(BatchJob.created_at.desc()).all()
    return jsonify({"success": True, "data": [_batch_job_response(j) for j in jobs]})


@batch_bp.get("/batch_jobs/<int:batch_job_id>")
def get_batch_job(batch_job_id: int) -> Any:
    batch = db.session.get(BatchJob, batch_job_id)
    if batch is None:
        return jsonify({"success": False, "error": "Batch job not found"}), 404

    return jsonify({"success": True, "data": _batch_job_response(batch)})


@batch_bp.post("/batch_jobs/<int:batch_job_id>/cancel")
def cancel_batch_job(batch_job_id: int) -> Any:
    batch = db.session.get(BatchJob, batch_job_id)
    if batch is None:
        return jsonify({"success": False, "error": "Batch job not found"}), 404

    batch.cancel_requested = True
    if batch.status == "queued":
        batch.status = "cancelled"
        batch.end_time = datetime.utcnow()
    db.session.commit()

    return jsonify({"success": True, "data": _batch_job_response(batch)})
