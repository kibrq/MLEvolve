"""Verification helpers for deterministic hidden validation adapters."""

import csv
import json
from pathlib import Path
from typing import Any

from .common import infer_competition_id, logger, read_csv_rows


def verify_split_artifacts(cfg, manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        return {
            "ok": False,
            "reason": "split manifest missing",
            "report": "",
            "manifest": {},
            "evidence_summary": "",
        }

    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        return {
            "ok": False,
            "reason": f"manifest parse failed: {exc}",
            "report": "",
            "manifest": {},
            "evidence_summary": "",
        }

    required_keys = [
        "visible_input_dir",
        "visible_validation_dir",
        "hidden_answers_path",
        "hidden_sample_submission_path",
        "competition_id",
        "split_seed",
        "strategy_summary",
        "visible_count",
        "hidden_count",
        "contract_version",
        "evidence_dir",
        "authoritative_train_sources",
        "authoritative_label_sources",
        "authoritative_validation_sources",
        "authoritative_hidden_answer_sources",
        "hidden_answer_granularity",
    ]
    missing = [key for key in required_keys if key not in manifest]
    if missing:
        return {
            "ok": False,
            "reason": f"manifest missing required keys: {missing}",
            "report": "",
            "manifest": manifest,
            "evidence_summary": "",
        }

    visible_dir = Path(manifest["visible_input_dir"])
    visible_validation_dir = Path(manifest["visible_validation_dir"])
    answers_path = Path(manifest["hidden_answers_path"])
    sample_submission_path = Path(manifest["hidden_sample_submission_path"])
    evidence_dir = Path(manifest["evidence_dir"])
    if not visible_dir.exists():
        return {
            "ok": False,
            "reason": "visible input dir missing",
            "report": "",
            "manifest": manifest,
            "evidence_summary": "",
        }
    if not visible_validation_dir.exists():
        return {
            "ok": False,
            "reason": "visible validation dir missing",
            "report": "",
            "manifest": manifest,
            "evidence_summary": "",
        }
    if not answers_path.exists():
        return {
            "ok": False,
            "reason": "hidden answers path missing",
            "report": "",
            "manifest": manifest,
            "evidence_summary": "",
        }
    if not sample_submission_path.exists():
        return {
            "ok": False,
            "reason": "hidden sample submission path missing",
            "report": "",
            "manifest": manifest,
            "evidence_summary": "",
        }

    evidence_check = _validate_evidence_bundle(evidence_dir)
    if not evidence_check["ok"]:
        return {
            "ok": False,
            "reason": evidence_check["reason"],
            "report": "",
            "manifest": manifest,
            "evidence_summary": "",
        }

    try:
        if sample_submission_path.resolve().is_relative_to(visible_dir.resolve()):
            return {
                "ok": False,
                "reason": "hidden sample submission path points inside visible input dir",
                "report": "",
                "manifest": manifest,
                "evidence_summary": "",
            }
    except Exception:
        pass

    if int(manifest["visible_count"]) <= 0 or int(manifest["hidden_count"]) <= 0:
        return {
            "ok": False,
            "reason": "visible_count or hidden_count is non-positive",
            "report": "",
            "manifest": manifest,
            "evidence_summary": "",
        }

    expected_competition_id = infer_competition_id(cfg)
    if str(manifest["competition_id"]) != expected_competition_id:
        return {
            "ok": False,
            "reason": (
                "manifest competition_id does not match current task: "
                f"{manifest['competition_id']} != {expected_competition_id}"
            ),
            "report": "",
            "manifest": manifest,
            "evidence_summary": "",
        }

    source_layout_check = _validate_visible_top_level_layout(cfg, visible_dir)
    if not source_layout_check["ok"]:
        return {
            "ok": False,
            "reason": source_layout_check["reason"],
            "report": "",
            "manifest": manifest,
            "evidence_summary": "",
        }

    grade_check = _validate_hidden_sample_submission(cfg=cfg, manifest=manifest)
    if not grade_check["ok"]:
        return {
            "ok": False,
            "reason": f"{grade_check['reason']}: {grade_check['detail']}",
            "report": "",
            "manifest": manifest,
            "evidence_summary": "",
        }

    alignment = _compute_split_alignment(
        visible_dir=visible_dir,
        visible_validation_dir=visible_validation_dir,
        answers_path=answers_path,
        sample_submission_path=sample_submission_path,
        manifest=manifest,
    )
    expected_visible_count = alignment["visible_training_count"]
    expected_hidden_count = alignment["hidden_answers_count"]
    if int(manifest["visible_count"]) != expected_visible_count:
        return {
            "ok": False,
            "reason": (
                "manifest visible_count does not match visible training count: "
                f"{manifest['visible_count']} != {expected_visible_count}"
            ),
            "report": "",
            "manifest": manifest,
            "evidence_summary": "",
        }
    if int(manifest["hidden_count"]) != expected_hidden_count:
        return {
            "ok": False,
            "reason": (
                "manifest hidden_count does not match hidden answers rows: "
                f"{manifest['hidden_count']} != {expected_hidden_count}"
            ),
            "report": "",
            "manifest": manifest,
            "evidence_summary": "",
        }
    if (
        alignment["hidden_answer_granularity"] == "direct"
        and alignment["visible_validation_count"] != expected_hidden_count
    ):
        return {
            "ok": False,
            "reason": (
                "visible validation file count does not match hidden answers rows: "
                f"{alignment['visible_validation_count']} != {expected_hidden_count}"
            ),
            "report": "",
            "manifest": manifest,
            "evidence_summary": "",
        }
    if alignment["overlap_train_answers_count"] > 0:
        return {
            "ok": False,
            "reason": (
                "some hidden-answer ids still exist in visible training data: "
                f"{alignment['overlap_train_answers_examples']}"
            ),
            "report": "",
            "manifest": manifest,
            "evidence_summary": "",
        }

    evidence_summary = _build_evidence_summary(evidence_dir)
    report = (
        f"visible_input_dir={visible_dir}\n"
        f"visible_validation_dir={visible_validation_dir}\n"
        f"hidden_answers_path={answers_path}\n"
        f"hidden_sample_submission_path={sample_submission_path}\n"
        f"contract_version={manifest['contract_version']}\n"
        f"visible_count={manifest['visible_count']}\n"
        f"hidden_count={manifest['hidden_count']}\n"
        f"split_seed={manifest['split_seed']}\n"
        f"top_level_layout_check={source_layout_check['detail']}\n"
        f"authoritative_train_sources={manifest['authoritative_train_sources']}\n"
        f"authoritative_label_sources={manifest['authoritative_label_sources']}\n"
        f"authoritative_validation_sources={manifest['authoritative_validation_sources']}\n"
        f"authoritative_hidden_answer_sources={manifest['authoritative_hidden_answer_sources']}\n"
        f"evidence_dir={evidence_dir}\n"
        f"evidence_bundle_check={evidence_check['detail']}\n"
        f"strategy_summary={manifest['strategy_summary']}\n"
        f"sample_submission_validation_grade_check={grade_check['detail']}\n"
        f"visible_training_count={alignment['visible_training_count']}\n"
        f"hidden_answers_count={alignment['hidden_answers_count']}\n"
        f"sample_validation_rows={alignment['sample_validation_rows']}\n"
        f"visible_validation_count={alignment['visible_validation_count']}\n"
        f"hidden_answer_granularity={alignment['hidden_answer_granularity']}\n"
        f"overlap_train_answers_count={alignment['overlap_train_answers_count']}\n"
        f"sample_ids_match_answers={alignment['sample_ids_match_answers']}\n"
        f"overlap_train_answers_examples={alignment['overlap_train_answers_examples']}\n"
        f"visible_validation_examples={alignment['visible_validation_examples']}"
    )
    return {
        "ok": True,
        "reason": "",
        "report": report,
        "manifest": manifest,
        "evidence_summary": evidence_summary,
    }


def _extract_id(row: dict[str, str]) -> str | None:
    for key in [
        "id",
        "Id",
        "id_code",
        "image_id",
        "StudyInstanceUID",
        "image_name",
        "key",
        "clip",
        "request_id",
        "sentence_id",
        "rec_id",
    ]:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _read_structured_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        try:
            with path.open() as f:
                return list(csv.DictReader(f, delimiter=delimiter))
        except Exception:
            return []
    if suffix == ".json":
        try:
            data = json.loads(path.read_text())
        except Exception:
            return []
        return data if isinstance(data, list) else []
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        try:
            for line in path.read_text().splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                item = json.loads(stripped)
                if isinstance(item, dict):
                    rows.append(item)
        except Exception:
            return []
        return rows
    return read_csv_rows(path)


def _compute_split_alignment(
    visible_dir: Path,
    visible_validation_dir: Path,
    answers_path: Path,
    sample_submission_path: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    answer_rows = read_csv_rows(answers_path)
    sample_rows = read_csv_rows(sample_submission_path)
    answer_ids = [row.get("id") or row.get("Id") or _extract_id(row) for row in answer_rows]
    sample_ids = [row.get("id") or row.get("Id") or _extract_id(row) for row in sample_rows]
    answer_id_set = {item for item in answer_ids if item}

    visible_validation_ids, visible_validation_examples = _collect_visible_validation_ids(
        visible_dir,
        visible_validation_dir,
        manifest,
    )
    visible_training_count = _compute_visible_training_count(
        visible_dir,
        manifest,
        read_csv_rows(visible_dir / "train.csv"),
    )
    hidden_answers_count = len(answer_rows)
    sample_validation_rows = len(sample_rows)
    hidden_answer_granularity = manifest.get("hidden_answer_granularity", "direct")
    overlap_train_answers_examples: list[str] = []
    overlap_train_answers_count = 0
    for candidate in answer_id_set:
        for source in manifest.get("authoritative_train_sources", []):
            source_path = Path(source)
            path = visible_dir / source_path.name
            if path.is_dir() and (path / candidate).exists():
                overlap_train_answers_examples.append(candidate)
                overlap_train_answers_count += 1
                break

    return {
        "visible_training_count": visible_training_count,
        "hidden_answers_count": hidden_answers_count,
        "sample_validation_rows": sample_validation_rows,
        "visible_validation_count": len(visible_validation_ids),
        "hidden_answer_granularity": hidden_answer_granularity,
        "overlap_train_answers_count": overlap_train_answers_count,
        "sample_ids_match_answers": sample_ids == answer_ids,
        "overlap_train_answers_examples": overlap_train_answers_examples[:10],
        "visible_validation_examples": visible_validation_examples[:10],
    }


def _collect_visible_validation_ids(
    visible_dir: Path,
    visible_validation_dir: Path,
    manifest: dict[str, Any],
) -> tuple[set[str], list[str]]:
    ids: set[str] = set()
    examples: list[str] = []

    authoritative_sources = manifest.get("authoritative_validation_sources", [])
    candidate_paths: list[Path] = []
    if isinstance(authoritative_sources, list) and authoritative_sources:
        for source in authoritative_sources:
            try:
                candidate_paths.append(visible_dir / str(source))
            except Exception:
                continue
    else:
        candidate_paths.append(visible_validation_dir)

    # For direct-label adapters, prefer structured validation manifests such as
    # `validation/validation.csv` over also counting raw asset files in the same
    # directory. This avoids false double-counting for tasks like dogs-vs-cats.
    structured_paths = [
        path for path in candidate_paths if path.suffix.lower() in {".csv", ".tsv", ".json", ".jsonl", ".txt"}
    ]
    if structured_paths:
        scan_paths = structured_paths
    else:
        scan_paths = candidate_paths

    for root in scan_paths:
        if not root.exists():
            continue
        paths = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in paths:
            if path.is_dir():
                continue
            rel = str(path.relative_to(visible_validation_dir)) if path.is_relative_to(visible_validation_dir) else path.name
            if path.suffix.lower() in {".csv", ".tsv", ".json", ".jsonl", ".txt"}:
                for row in _read_structured_rows(path):
                    candidate = _extract_id(row)
                    if candidate:
                        ids.add(str(candidate))
                        if len(examples) < 10:
                            examples.append(str(candidate))
            else:
                ids.add(path.name)
                if len(examples) < 10:
                    examples.append(rel)
    return ids, examples


def _compute_visible_training_count(
    visible_dir: Path,
    manifest: dict[str, Any],
    train_rows: list[dict[str, str]],
) -> int:
    if train_rows:
        return len(train_rows)

    sources = manifest.get("authoritative_train_sources", [])
    if not isinstance(sources, list):
        return 0

    counts: list[int] = []
    for source in sources:
        try:
            source_path = Path(source)
        except Exception:
            continue
        if source_path.name:
            candidate = visible_dir / source_path.name
            if candidate.is_dir():
                counts.append(sum(1 for path in candidate.rglob("*") if path.is_file()))
                continue
            if candidate.is_file() and candidate.suffix.lower() in {".csv", ".tsv"}:
                counts.append(len(read_csv_rows(candidate)))
                continue

    return max(counts) if counts else 0


def _validate_hidden_sample_submission(cfg, manifest: dict[str, Any]) -> dict[str, Any]:
    try:
        from mlebench.registry import registry
        from mlebench.utils import load_answers, read_csv

        competition = registry.get_competition(infer_competition_id(cfg))
        answers = load_answers(Path(manifest["hidden_answers_path"]))
        submission = read_csv(Path(manifest["hidden_sample_submission_path"]))
        score = competition.grader(submission, answers)
        return {"ok": score is not None, "reason": "grader returned None" if score is None else "", "detail": f"score={score}"}
    except Exception as exc:
        logger.exception("Hidden sample submission validation failed")
        return {"ok": False, "reason": "hidden sample submission grading failed", "detail": str(exc)}


def _validate_evidence_bundle(evidence_dir: Path) -> dict[str, Any]:
    expected = [
        evidence_dir / "source_layout.txt",
        evidence_dir / "visible_layout.txt",
        evidence_dir / "hidden_layout.txt",
        evidence_dir / "authoritative_sources.txt",
        evidence_dir / "notes.txt",
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        return {"ok": False, "reason": f"evidence bundle missing files: {missing}", "detail": ""}
    return {"ok": True, "reason": "", "detail": f"present={len(expected)}"}


def _build_evidence_summary(evidence_dir: Path) -> str:
    parts: list[str] = []
    for name in [
        "source_layout.txt",
        "visible_layout.txt",
        "hidden_layout.txt",
        "authoritative_sources.txt",
        "notes.txt",
    ]:
        path = evidence_dir / name
        parts.append(f"## {name}")
        if not path.exists():
            parts.append("<missing>")
            continue
        try:
            for line in path.read_text(errors="ignore").splitlines()[:20]:
                parts.append(line[:300])
        except Exception as exc:
            parts.append(f"<read failed: {exc}>")
    return "\n".join(parts)


def _validate_visible_top_level_layout(cfg, visible_dir: Path) -> dict[str, Any]:
    source_dir = Path(cfg.data_dir).resolve()
    if not source_dir.exists():
        return {"ok": False, "reason": f"source input dir missing: {source_dir}", "detail": ""}

    source_entries = {path.name for path in source_dir.iterdir()}
    visible_entries = {path.name for path in visible_dir.iterdir()}

    missing_entries: list[str] = []
    accepted_extracted_archives: list[str] = []
    for entry in sorted(source_entries - visible_entries):
        source_path = source_dir / entry
        if source_path.suffix.lower() == ".zip" and source_path.stem in visible_entries:
            accepted_extracted_archives.append(entry)
            continue
        missing_entries.append(entry)
    if missing_entries:
        return {
            "ok": False,
            "reason": f"visible layout is missing top-level source entries: {missing_entries}",
            "detail": f"missing={missing_entries}; accepted_extracted_archives={accepted_extracted_archives}",
        }

    required_extra_entries = []
    if "validation" not in visible_entries:
        required_extra_entries.append("validation")
    if "sampleValidationSubmission.csv" not in visible_entries:
        required_extra_entries.append("sampleValidationSubmission.csv")
    if required_extra_entries:
        return {
            "ok": False,
            "reason": f"visible layout is missing required validation artifacts: {required_extra_entries}",
            "detail": f"missing_validation_artifacts={required_extra_entries}",
        }

    extra_entries = sorted(visible_entries - source_entries)
    return {
        "ok": True,
        "reason": "",
        "detail": (
            f"all_source_entries_present; "
            f"accepted_extracted_archives={accepted_extracted_archives}; extra={extra_entries}"
        ),
    }
