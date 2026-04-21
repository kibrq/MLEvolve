"""Hidden search validation helpers."""

from pathlib import Path
from typing import Any

from .common import (
    activate_fallback,
    default_state,
    get_runtime_state,
    infer_competition_id,
    load_hidden_scoreboard,
    logger,
    save_hidden_scoreboard,
    save_runtime_state,
)
from .lite import maybe_prepare_mlebench_lite_hidden_validation


def prepare_hidden_validation(cfg) -> dict[str, Any]:
    if not cfg.hidden_validation.enabled:
        state = default_state()
        state["enabled"] = False
        save_hidden_scoreboard(cfg, {})
        return save_runtime_state(cfg, state)

    existing = get_runtime_state(cfg)
    if existing.get("active") or existing.get("fallback_mode"):
        return existing

    competition_id = infer_competition_id(cfg)
    try:
        deterministic_state = maybe_prepare_mlebench_lite_hidden_validation(cfg)
        save_hidden_scoreboard(cfg, {})
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


def _score_submission(
    cfg,
    state: dict[str, Any],
    submission_filename: str,
    answers_path_key: str,
    node_id: str,
    exec_result,
    metric_maximize: bool | None = None,
) -> dict[str, Any]:
    submission_path = cfg.workspace_dir / "submission" / submission_filename
    if exec_result.exc_type is not None:
        return {
            "valid": False,
            "reason": f"execution failed: {exec_result.exc_type}",
        }
    if not submission_path.exists():
        return {
            "valid": False,
            "reason": f"submission not found at {submission_path}",
        }

    try:
        from mlebench.registry import registry
        from mlebench.utils import load_answers, read_csv

        competition_id = infer_competition_id(cfg)
        competition = registry.get_competition(competition_id)
        lower_is_better = not metric_maximize if metric_maximize is not None else False
        answers = load_answers(Path(state[answers_path_key]))
        submission = read_csv(submission_path)
        score = competition.grader(submission, answers)
        if score is None:
            return {
                "valid": False,
                "reason": "grader returned None",
            }
        return {
            "valid": True,
            "score": float(score),
            "lower_is_better": lower_is_better,
            "submission_path": str(submission_path),
        }
    except Exception as exc:
        logger.exception("Validation scoring failed for node %s (%s)", node_id, submission_filename)
        return {
            "valid": False,
            "reason": str(exc) or "validation scoring failed",
        }


def score_validation_execution(
    cfg,
    state: dict[str, Any],
    node_id: str,
    exec_result,
    metric_maximize: bool | None = None,
) -> dict[str, Any]:
    return {
        "visible": _score_submission(
            cfg=cfg,
            state=state,
            submission_filename=f"submission_visible_{node_id}.csv",
            answers_path_key="visible_answers_path",
            node_id=node_id,
            exec_result=exec_result,
            metric_maximize=metric_maximize,
        ),
        "hidden": _score_submission(
            cfg=cfg,
            state=state,
            submission_filename=f"submission_hidden_{node_id}.csv",
            answers_path_key="hidden_answers_path",
            node_id=node_id,
            exec_result=exec_result,
            metric_maximize=metric_maximize,
        ),
    }


def record_hidden_score(cfg, node_id: str, report: dict[str, Any] | None) -> None:
    scoreboard = load_hidden_scoreboard(cfg)
    scoreboard[node_id] = report or {}
    save_hidden_scoreboard(cfg, scoreboard)
