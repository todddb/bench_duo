from __future__ import annotations

from typing import Any

from flask import Blueprint, jsonify, request

from app.evaluator import conversation_to_text, run_aggregator, run_judge
from app.extensions import db
from app.models import Conversation, EvaluationJob, Message, Model


evaluate_bp = Blueprint("evaluate", __name__, url_prefix="/api")


@evaluate_bp.post("/evaluate")
def create_evaluation() -> Any:
    payload = request.get_json(silent=True) or {}
    required = {"conversation_id", "main_model_id", "judge_model_ids"}
    missing = sorted(field for field in required if field not in payload)
    if missing:
        return jsonify({"success": False, "error": f"Missing required fields: {', '.join(missing)}"}), 400

    try:
        conversation_id = int(payload["conversation_id"])
        main_model_id = int(payload["main_model_id"])
        judge_model_ids = [int(model_id) for model_id in payload["judge_model_ids"]]
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "conversation_id, main_model_id, and judge_model_ids must be integers"}), 400

    if not judge_model_ids:
        return jsonify({"success": False, "error": "judge_model_ids must include at least one model id"}), 400

    conversation = db.session.get(Conversation, conversation_id)
    if conversation is None:
        return jsonify({"success": False, "error": "Conversation not found"}), 404

    main_model = db.session.get(Model, main_model_id)
    if main_model is None:
        return jsonify({"success": False, "error": "Main model not found"}), 404

    judge_models = Model.query.filter(Model.id.in_(judge_model_ids)).all()
    if len(judge_models) != len(set(judge_model_ids)):
        return jsonify({"success": False, "error": "One or more judge models were not found"}), 404

    messages = (
        Message.query.filter_by(conversation_id=conversation.id)
        .order_by(Message.created_at.asc(), Message.id.asc())
        .all()
    )
    conversation_text = conversation_to_text(messages)

    evaluation_job = EvaluationJob(
        conversation_id=conversation.id,
        main_model_id=main_model.id,
        judge_model_ids=judge_model_ids,
        status="running",
    )
    db.session.add(evaluation_job)
    db.session.commit()

    try:
        judge_results = [run_judge(judge_model, conversation_text) for judge_model in judge_models]
        aggregate_report = run_aggregator(main_model, conversation_text, judge_results)
        flagged_instances = aggregate_report.get("flagged_instances", [])
        flagged_lines = []
        for item in flagged_instances:
            try:
                message_index = int(item.get("message_index"))
            except (TypeError, ValueError):
                continue
            if 0 <= message_index < len(messages):
                flagged_lines.append(
                    {
                        "message_id": messages[message_index].id,
                        "message_index": message_index,
                        "reason": item.get("category", "other"),
                        "excerpt": item.get("excerpt", ""),
                        "severity": item.get("severity"),
                    }
                )
        aggregate_report["scores"] = {
            "overall": aggregate_report.get("overall_score"),
            "completion": aggregate_report.get("completion_score"),
            "realistic": aggregate_report.get("realistic_score"),
            "highest_severity": aggregate_report.get("highest_severity"),
        }
        aggregate_report["flagged_lines"] = flagged_lines

        evaluation_job.results = {"judges": judge_results}
        evaluation_job.report = aggregate_report
        evaluation_job.status = "completed"
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        evaluation_job = db.session.get(EvaluationJob, evaluation_job.id)
        if evaluation_job is not None:
            evaluation_job.status = "failed"
            evaluation_job.results = {"error": str(exc)}
            db.session.commit()
        return jsonify({"success": False, "error": f"Evaluation failed: {exc}"}), 500

    return jsonify({"success": True, "data": {"eval_job_id": evaluation_job.id}}), 201


@evaluate_bp.get("/evaluate/<int:evaluation_id>")
def get_evaluation(evaluation_id: int) -> Any:
    evaluation = db.session.get(EvaluationJob, evaluation_id)
    if evaluation is None:
        return jsonify({"success": False, "error": "Evaluation job not found"}), 404

    return jsonify(
        {
            "success": True,
            "data": {
                "id": evaluation.id,
                "conversation_id": evaluation.conversation_id,
                "batch_id": evaluation.batch_id,
                "main_model_id": evaluation.main_model_id,
                "judge_model_ids": evaluation.judge_model_ids,
                "status": evaluation.status,
                "judge_results": (evaluation.results or {}).get("judges"),
                "aggregate_report": evaluation.report,
            },
        }
    )
