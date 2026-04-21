"""Shared helpers for deterministic hidden validation."""

import csv
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger("MLEvolve")

FALLBACK_LITE_COMPETITION_IDS = {
    "aerial-cactus-identification",
    "aptos2019-blindness-detection",
    "denoising-dirty-documents",
    "detecting-insults-in-social-commentary",
    "dog-breed-identification",
    "dogs-vs-cats-redux-kernels-edition",
    "histopathologic-cancer-detection",
    "jigsaw-toxic-comment-classification-challenge",
    "leaf-classification",
    "mlsp-2013-birds",
    "new-york-city-taxi-fare-prediction",
    "nomad2018-predict-transparent-conductors",
    "plant-pathology-2020-fgvc7",
    "random-acts-of-pizza",
    "ranzcr-clip-catheter-line-classification",
    "siim-isic-melanoma-classification",
    "spooky-author-identification",
    "tabular-playground-series-dec-2021",
    "tabular-playground-series-may-2022",
    "text-normalization-challenge-english-language",
    "text-normalization-challenge-russian-language",
    "the-icml-2013-whale-challenge-right-whale-redux",
}


def state_dir(cfg) -> Path:
    return cfg.workspace_dir.parent / ".hidden_validation"


def state_path(cfg) -> Path:
    return state_dir(cfg) / "state.json"


def default_state() -> dict[str, Any]:
    return {
        "enabled": False,
        "active": False,
        "fallback_mode": False,
        "fallback_reason": "",
        "visible_input_dir": "",
        "visible_validation_dir": "",
        "visible_answers_path": "",
        "visible_sample_submission_path": "",
        "hidden_validation_dir": "",
        "hidden_answers_path": "",
        "hidden_sample_submission_path": "",
        "manifest_path": "",
        "splitter_attempts": 0,
        "reviewer_status": "not_run",
    }


def get_runtime_state(cfg) -> dict[str, Any]:
    path = state_path(cfg)
    if not path.exists():
        return default_state()
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        logger.warning("Failed to load hidden validation state: %s", exc)
        return default_state()


def save_runtime_state(cfg, state: dict[str, Any]) -> dict[str, Any]:
    path = state_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))
    return state


def hidden_scoreboard_path(cfg) -> Path:
    return state_dir(cfg) / "hidden_scores.json"


def load_hidden_scoreboard(cfg) -> dict[str, Any]:
    path = hidden_scoreboard_path(cfg)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        logger.warning("Failed to load hidden score ledger: %s", exc)
        return {}


def save_hidden_scoreboard(cfg, scoreboard: dict[str, Any]) -> dict[str, Any]:
    path = hidden_scoreboard_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scoreboard, indent=2))
    return scoreboard


def activate_fallback(cfg, state: dict[str, Any], reason: str) -> dict[str, Any]:
    state = {**state}
    state["active"] = False
    state["fallback_mode"] = True
    state["fallback_reason"] = reason
    logger.warning("Hidden validation fallback activated: %s", reason)
    return save_runtime_state(cfg, state)


@lru_cache(maxsize=1)
def load_mlebench_lite_competition_ids() -> set[str]:
    try:
        from mlebench.registry import registry

        return set(registry.get_lite_competition_ids())
    except Exception as exc:
        logger.info(
            "Installed mle-bench package does not expose lite split metadata; "
            "using built-in lite competition id fallback: %s",
            exc,
        )
        return set(FALLBACK_LITE_COMPETITION_IDS)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open() as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def resolve_dataset_root(cfg) -> Path:
    dataset_dir = getattr(cfg, "dataset_dir", None)
    if dataset_dir:
        dataset_root = Path(dataset_dir)
        if str(dataset_root).strip():
            return dataset_root.resolve()
    return Path(cfg.data_dir).resolve().parents[2]


def infer_competition_id(cfg) -> str:
    for env_name in ["COMPETITION_ID", "TASK", "MLEBENCH_COMPETITION_ID"]:
        env_value = str(os.getenv(env_name, "")).strip()
        if env_value:
            return env_value
    explicit_name = str(
        getattr(getattr(cfg, "hidden_validation", None), "competition_name", "")
    ).strip()
    if explicit_name:
        return explicit_name

    source_root = Path(cfg.data_dir).resolve()
    for candidate in [source_root.name, source_root.parent.name]:
        if candidate and candidate not in {"prepared", "public", "input", "data"}:
            return candidate

    dataset_root = resolve_dataset_root(cfg)
    task_dir = getattr(cfg, "desc_file", None)
    if task_dir:
        return Path(task_dir).resolve().parent.name
    return dataset_root.name


def build_data_layout_summary(source_input_dir: Path) -> str:
    lines: list[str] = []
    for idx, path in enumerate(sorted(source_input_dir.rglob("*"))):
        if idx >= 300:
            lines.append("... truncated ...")
            break
        rel = path.relative_to(source_input_dir)
        if path.is_dir():
            lines.append(f"[dir] {rel}")
            continue
        size = path.stat().st_size
        lines.append(f"[file] {rel} ({size} bytes)")
        if path.suffix.lower() in {".csv", ".tsv", ".txt", ".jsonl"}:
            try:
                preview = path.read_text(errors="ignore").splitlines()[:3]
                for line in preview:
                    lines.append(f"    {line[:200]}")
            except Exception:
                continue
    return "\n".join(lines)
