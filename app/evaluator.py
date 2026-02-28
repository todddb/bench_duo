from __future__ import annotations

import json
from typing import Any

from app.connectors import Connector, MLXConnector, OllamaConnector, TensorRTConnector
from app.models import Message, Model


JUDGE_TEMPLATE = (
    "You are an expert evaluator. Analyze the conversation and return strict JSON with this exact shape: "
    '{{"issues":[{{"message_index":0,"category":"hallucination|forbidden|other","excerpt":"text","severity":1}}],'
    '"completion_score":0,"realistic_score":0,"notes":"short summary"}}. '
    "Severity range is 1-5. completion_score and realistic_score are 0-100 integers. "
    "Only include an issue if a concrete problem exists and always map message_index to the conversation list index.\n\n"
    "Conversation:\n{conversation_text}"
)

AGGREGATOR_TEMPLATE = (
    "You are the main evaluation aggregator. Given the conversation and judge outputs, return strict JSON with shape: "
    '{{"summary":"...","overall_score":0.0,"total_issues":0,"highest_severity":0,'
    '"completion_score":0,"realistic_score":0,"flagged_instances":[{{"message_index":0,"category":"...","excerpt":"...","severity":1}}]}}.\n\n'
    "Conversation:\n{conversation_text}\n\nJudge Outputs:\n{judge_outputs}"
)


def connector_for_model(model: Model) -> Connector:
    backend = model.backend.lower()
    if backend == "ollama":
        return OllamaConnector(host=model.host, port=model.port)
    if backend == "mlx":
        return MLXConnector()
    if backend in {"tensorrt", "tensorrt_llm"}:
        return TensorRTConnector()
    raise ValueError(f"Unsupported backend: {model.backend}")


def conversation_to_text(messages: list[Message]) -> str:
    lines = []
    for idx, message in enumerate(messages):
        lines.append(f"[{idx}] {message.sender_role}: {message.content}")
    return "\n".join(lines)


def _extract_json_block(raw_text: str) -> Any:
    raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    start_obj = raw_text.find("{")
    end_obj = raw_text.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        candidate = raw_text[start_obj : end_obj + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    start_arr = raw_text.find("[")
    end_arr = raw_text.rfind("]")
    if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
        candidate = raw_text[start_arr : end_arr + 1]
        return json.loads(candidate)

    raise ValueError("Model output is not valid JSON")


def normalize_judge_output(parsed: Any) -> dict[str, Any]:
    if isinstance(parsed, list):
        return {
            "issues": parsed,
            "completion_score": None,
            "realistic_score": None,
            "notes": "",
        }
    if isinstance(parsed, dict):
        issues = parsed.get("issues")
        if not isinstance(issues, list):
            issues = []
        return {
            "issues": issues,
            "completion_score": parsed.get("completion_score"),
            "realistic_score": parsed.get("realistic_score"),
            "notes": parsed.get("notes", ""),
        }
    raise ValueError("Judge output JSON must be an object or array")


def run_judge(model: Model, conversation_text: str) -> dict[str, Any]:
    connector = connector_for_model(model)
    prompt = JUDGE_TEMPLATE.format(conversation_text=conversation_text)
    response_text = connector.chat(
        messages=[{"role": "user", "content": prompt}],
        settings={"model": model.model_name, "temperature": 0, "max_tokens": 800},
    )
    parsed = _extract_json_block(response_text)
    normalized = normalize_judge_output(parsed)
    normalized["judge_model_id"] = model.id
    normalized["judge_model_name"] = model.name
    return normalized


def _code_aggregate(judge_results: list[dict[str, Any]]) -> dict[str, Any]:
    flagged_instances: list[dict[str, Any]] = []
    highest_severity = 0
    completion_scores: list[float] = []
    realistic_scores: list[float] = []

    for result in judge_results:
        for issue in result.get("issues", []):
            severity = int(issue.get("severity", 0) or 0)
            highest_severity = max(highest_severity, severity)
            flagged_instances.append(
                {
                    "message_index": issue.get("message_index"),
                    "category": issue.get("category", "other"),
                    "excerpt": issue.get("excerpt", ""),
                    "severity": severity,
                    "judge_model_id": result.get("judge_model_id"),
                }
            )

        completion = result.get("completion_score")
        if isinstance(completion, (int, float)):
            completion_scores.append(float(completion))
        realistic = result.get("realistic_score")
        if isinstance(realistic, (int, float)):
            realistic_scores.append(float(realistic))

    completion_avg = sum(completion_scores) / len(completion_scores) if completion_scores else 0.0
    realistic_avg = sum(realistic_scores) / len(realistic_scores) if realistic_scores else 0.0
    issue_penalty = min(0.5, len(flagged_instances) * 0.05 + highest_severity * 0.03)
    base_score = ((completion_avg / 100.0) * 0.5) + ((realistic_avg / 100.0) * 0.5)
    overall_score = max(0.0, min(1.0, base_score - issue_penalty))

    return {
        "summary": (
            f"Total issues: {len(flagged_instances)}; highest severity: {highest_severity}; "
            f"Completeness: {completion_avg:.1f}; Realistic: {realistic_avg:.1f}."
        ),
        "overall_score": round(overall_score, 3),
        "total_issues": len(flagged_instances),
        "highest_severity": highest_severity,
        "completion_score": round(completion_avg, 1),
        "realistic_score": round(realistic_avg, 1),
        "flagged_instances": flagged_instances,
    }


def run_aggregator(main_model: Model, conversation_text: str, judge_results: list[dict[str, Any]]) -> dict[str, Any]:
    connector = connector_for_model(main_model)
    prompt = AGGREGATOR_TEMPLATE.format(
        conversation_text=conversation_text,
        judge_outputs=json.dumps(judge_results),
    )
    response_text = connector.chat(
        messages=[{"role": "user", "content": prompt}],
        settings={"model": main_model.model_name, "temperature": 0, "max_tokens": 1000},
    )

    try:
        parsed = _extract_json_block(response_text)
        if not isinstance(parsed, dict):
            raise ValueError("aggregator output must be an object")
        if "flagged_instances" not in parsed:
            parsed["flagged_instances"] = []
        return parsed
    except Exception:
        return _code_aggregate(judge_results)
