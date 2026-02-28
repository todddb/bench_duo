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


def _summary_payload(batch: BatchJob) -> dict[str, Any]:
    elapsed = 0.0
    if batch.start_time:
        end = batch.end_time or datetime.utcnow()
        elapsed = max(0.0, (end - batch.start_time).total_seconds())

    summary = batch.summary or {}
    total_messages = int(summary.get("total_messages", 0) or 0)
    tokens_generated = int(summary.get("tokens_generated", 0) or 0)

    avg_time = (elapsed / batch.completed_runs) if batch.completed_runs else 0.0
    tokens_per_sec = (tokens_generated / elapsed) if elapsed else 0.0

    return {
        "completed": batch.completed_runs,
        "total": batch.num_runs,
        "status": batch.status,
        "avg_time": round(avg_time, 4),
        "elapsed_seconds": round(elapsed, 4),
        "tokens_per_sec": round(tokens_per_sec, 4),
        "progress_pct": round((batch.completed_runs / max(1, batch.num_runs)) * 100, 2),
        "total_messages": total_messages,
        "tokens_generated": tokens_generated,
    }


def _run_single_conversation(batch: BatchJob, run_seed: int | None) -> dict[str, Any]:
    conversation = Conversation(
        agent1_id=batch.agent1_id,
        agent2_id=batch.agent2_id,
        ttl=batch.ttl,
        random_seed=run_seed,
        status="running",
    )
    db.session.add(conversation)
    db.session.flush()
    db.session.add(Message(conversation_id=conversation.id, sender_role="user", content=batch.prompt))

    started = perf_counter()
    current_text = batch.prompt
    tokens_generated = 0
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
                "seed": run_seed,
            },
        )
        tokens = max(1, len(text.split()))
        tokens_generated += tokens
        db.session.add(
            Message(
                conversation_id=conversation.id,
                sender_role=sender,
                agent_id=agent.id,
                content=text,
                tokens=tokens,
            )
        )
        current_text = text

    conversation.status = "finished"
    conversation.finished_at = datetime.utcnow()
    db.session.commit()
    return {
        "id": conversation.id,
        "messages": max(0, len(conversation.messages) - 1),
        "tokens": tokens_generated,
        "duration": perf_counter() - started,
    }


def _process_batch(app, batch_id: int) -> None:
    with app.app_context():
        batch = db.session.get(BatchJob, batch_id)
        if batch is None:
            return
        if batch.status in {"completed", "failed", "cancelled"}:
            return

        batch.status = "running"
        batch.start_time = batch.start_time or datetime.utcnow()
        db.session.commit()

        try:
            for run_idx in range(batch.completed_runs, batch.num_runs):
                batch = db.session.get(BatchJob, batch_id)
                if batch is None:
                    return
                if batch.cancel_requested or batch.status in {"stopped", "cancelled"}:
                    batch.status = "cancelled"
                    batch.end_time = datetime.utcnow()
                    db.session.commit()
                    return

                run_seed = (batch.seed + run_idx) if batch.seed is not None else None
                run_summary = _run_single_conversation(batch, run_seed)

                batch = db.session.get(BatchJob, batch_id)
                if batch is None:
                    return
                summary = batch.summary or {}
                conversation_ids = summary.get("conversation_ids", [])
                conversation_ids.append(run_summary["id"])
                summary["conversation_ids"] = conversation_ids
                summary["total_messages"] = int(summary.get("total_messages", 0)) + run_summary["messages"]
                summary["tokens_generated"] = int(summary.get("tokens_generated", 0)) + run_summary["tokens"]
                batch.summary = summary
                batch.completed_runs += 1
                db.session.commit()

            batch = db.session.get(BatchJob, batch_id)
            if batch is None:
                return
            batch.status = "completed"
            batch.end_time = datetime.utcnow()
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


def _create_batch_job(payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    required = {"agent1_id", "agent2_id", "prompt", "ttl", "num_runs"}
    missing = sorted(field for field in required if field not in payload)
    if missing:
        return {"success": False, "error": f"Missing required fields: {', '.join(missing)}"}, 400

    try:
        agent1_id = int(payload["agent1_id"])
        agent2_id = int(payload["agent2_id"])
        ttl = max(1, int(payload["ttl"]))
        num_runs = max(1, int(payload["num_runs"]))
        seed = int(payload["seed"]) if payload.get("seed") is not None else None
    except (TypeError, ValueError):
        return {"success": False, "error": "agent1_id, agent2_id, ttl, num_runs and seed must be integers"}, 400

    agent1 = db.session.get(Agent, agent1_id)
    agent2 = db.session.get(Agent, agent2_id)
    if agent1 is None or agent2 is None:
        return {"success": False, "error": "agent1_id or agent2_id does not exist"}, 404

    batch = BatchJob(
        agent1_id=agent1_id,
        agent2_id=agent2_id,
        prompt=str(payload["prompt"]),
        ttl=ttl,
        num_runs=num_runs,
        seed=seed,
        status="queued",
        completed_runs=0,
        cancel_requested=False,
        summary={"conversation_ids": [], "total_messages": 0, "tokens_generated": 0},
    )
    db.session.add(batch)
    db.session.commit()

    if current_app.config.get("TESTING"):
        _process_batch(current_app._get_current_object(), batch.id)
    else:
        _batch_queue.put(batch.id)

    return {"success": True, "data": {"batch_id": batch.id, "status": "queued"}}, 201


@batch_bp.post("/batch_jobs")
def create_batch_job() -> Any:
    payload = request.get_json(silent=True) or {}
    response, status = _create_batch_job(payload)
    return jsonify(response), status


@batch_bp.post("/batch")
def create_batch() -> Any:
    payload = request.get_json(silent=True) or {}
    response, status = _create_batch_job(payload)
    return jsonify(response), status


@batch_bp.post("/batch_jobs/<int:batch_id>/cancel")
def cancel_batch_job(batch_id: int) -> Any:
    batch = db.session.get(BatchJob, batch_id)
    if batch is None:
        return jsonify({"success": False, "error": "Batch job not found"}), 404
    if batch.status in {"completed", "failed", "cancelled"}:
        return jsonify({"success": False, "error": "Batch job already finished"}), 400

    batch.cancel_requested = True
    if batch.status == "queued":
        batch.status = "cancelled"
        batch.end_time = datetime.utcnow()
    db.session.commit()
    return jsonify({"success": True, "data": {"message": "Batch job cancel requested"}})


@batch_bp.post("/batch/<int:batch_id>/stop")
def stop_batch(batch_id: int) -> Any:
    return cancel_batch_job(batch_id)


@batch_bp.get("/batch_jobs")
def list_batch_jobs() -> Any:
    jobs = BatchJob.query.order_by(BatchJob.id.desc()).all()
    payload = []
    for job in jobs:
        metrics = _summary_payload(job)
        payload.append(
            {
                "id": job.id,
                "agent1_id": job.agent1_id,
                "agent2_id": job.agent2_id,
                "agent1_name": job.agent1.name if job.agent1 else str(job.agent1_id),
                "agent2_name": job.agent2.name if job.agent2 else str(job.agent2_id),
                "prompt": job.prompt,
                "prompt_snippet": job.prompt[:60],
                "ttl": job.ttl,
                "num_runs": job.num_runs,
                "completed_runs": job.completed_runs,
                "status": job.status,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "start_time": job.start_time.isoformat() if job.start_time else None,
                "end_time": job.end_time.isoformat() if job.end_time else None,
                "summary": {**(job.summary or {}), **metrics},
                "elapsed_seconds": metrics["elapsed_seconds"],
            }
        )
    return jsonify({"success": True, "data": payload})


@batch_bp.get("/batch")
def list_batch_jobs_legacy() -> Any:
    return list_batch_jobs()


@batch_bp.get("/batch_jobs/<int:batch_id>")
def get_batch_job(batch_id: int) -> Any:
    batch = db.session.get(BatchJob, batch_id)
    if batch is None:
        return jsonify({"success": False, "error": "Batch job not found"}), 404

    metrics = _summary_payload(batch)
    return jsonify(
        {
            "success": True,
            "data": {
                "id": batch.id,
                "agent1_id": batch.agent1_id,
                "agent2_id": batch.agent2_id,
                "agent1_name": batch.agent1.name if batch.agent1 else str(batch.agent1_id),
                "agent2_name": batch.agent2.name if batch.agent2 else str(batch.agent2_id),
                "prompt": batch.prompt,
                "ttl": batch.ttl,
                "num_runs": batch.num_runs,
                "seed": batch.seed,
                "completed_runs": batch.completed_runs,
                "status": batch.status,
                "cancel_requested": batch.cancel_requested,
                "summary": {**(batch.summary or {}), **metrics},
                "created_at": batch.created_at.isoformat() if batch.created_at else None,
                "start_time": batch.start_time.isoformat() if batch.start_time else None,
                "end_time": batch.end_time.isoformat() if batch.end_time else None,
            },
        }
    )


@batch_bp.get("/batch/<int:batch_id>")
def get_batch_job_legacy(batch_id: int) -> Any:
    return get_batch_job(batch_id)
