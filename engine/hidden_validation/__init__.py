"""Hidden search validation helpers."""

from pathlib import Path
from typing import Any

from .common import (
    activate_fallback,
    default_state,
    get_runtime_state,
    infer_competition_id,
    logger,
    save_runtime_state,
)
from .lite import maybe_prepare_mlebench_lite_hidden_validation


def prepare_hidden_validation(cfg) -> dict[str, Any]:
    if not cfg.hidden_validation.enabled:
        state = default_state()
        state["enabled"] = False
        return save_runtime_state(cfg, state)

    existing = get_runtime_state(cfg)
    if existing.get("active") or existing.get("fallback_mode"):
        return existing

    competition_id = infer_competition_id(cfg)
    try:
        deterministic_state = maybe_prepare_mlebench_lite_hidden_validation(cfg)
    except Exception as exc:
        state = default_state()
        state["enabled"] = True
        fallback_reason = (
            f"Deterministic hidden-validation adapter failed for competition '{competition_id}': {exc}"
        )
        if (
            cfg.hidden_validation.allow_self_report_fallback
            and not cfg.hidden_validation.hard_fail_on_prepare_error
        ):
            return activate_fallback(cfg, state, reason=fallback_reason)
        raise
    if deterministic_state is not None:
        return deterministic_state

    state = default_state()
    state["enabled"] = True
    fallback_reason = (
        "Deterministic hidden-validation adapter is only supported for mle-bench lite competitions. "
        f"Competition '{competition_id}' will use self-reported metrics instead."
    )
    if cfg.hidden_validation.allow_self_report_fallback:
        return activate_fallback(cfg, state, reason=fallback_reason)
    return save_runtime_state(cfg, state)


def score_hidden_execution(
    cfg,
    state: dict[str, Any],
    node_id: str,
    exec_result,
    metric_maximize: bool | None = None,
) -> dict[str, Any]:
    submission_path = cfg.workspace_dir / "submission" / f"submission_validation_{node_id}.csv"
    if exec_result.exc_type is not None:
        return {
            "valid": False,
            "reason": f"hidden execution failed: {exec_result.exc_type}",
            "activate_fallback": True,
        }
    if not submission_path.exists():
        return {
            "valid": False,
            "reason": f"hidden submission not found at {submission_path}",
            "activate_fallback": True,
        }

    try:
        from mlebench.registry import registry
        from mlebench.utils import load_answers, read_csv

        competition_id = infer_competition_id(cfg)
        competition = registry.get_competition(competition_id)
        lower_is_better = not metric_maximize if metric_maximize is not None else False
        answers = load_answers(Path(state["hidden_answers_path"]))
        submission = read_csv(submission_path)
        score = competition.grader(submission, answers)
        if score is None:
            return {
                "valid": False,
                "reason": "hidden grader returned None",
                "activate_fallback": True,
            }
        return {
            "valid": True,
            "score": float(score),
            "lower_is_better": lower_is_better,
            "submission_path": str(submission_path),
            "activate_fallback": False,
        }
    except Exception as exc:
        logger.exception("Hidden scoring failed for node %s", node_id)
        return {
            "valid": False,
            "reason": str(exc) or "hidden scoring failed",
            "activate_fallback": True,
        }
