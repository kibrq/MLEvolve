from pathlib import Path
import json
import sys
import types


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.modules.setdefault("humanize", types.SimpleNamespace(naturaldelta=lambda value: str(value)))
sys.modules.setdefault("dataclasses_json", types.SimpleNamespace(DataClassJsonMixin=object))

from engine.executor import Interpreter
from engine.hidden_validation import prepare_hidden_validation, verify


def test_interpreter_isolates_main_and_hidden_submissions_only(tmp_path):
    interpreter = Interpreter(tmp_path)
    code = """
df.to_csv("submission.csv", index=False)
df.to_csv("submission_hidden.csv", index=False)
df.to_csv("submission_visible.csv", index=False)
"""

    isolated = interpreter.isolate_submission_path(code, "node123")

    assert "submission_node123.csv" in isolated
    assert "submission_hidden_node123.csv" in isolated
    assert "submission_visible.csv" in isolated
    assert "submission_visible_node123.csv" not in isolated


def test_verify_split_artifacts_accepts_single_hidden_split_manifest(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    hidden_dir = input_dir / "hidden_validation"
    evidence_dir = tmp_path / "evidence"
    answers_path = tmp_path / "hidden_answers.csv"
    sample_path = tmp_path / "sampleHiddenValidationSubmission.csv"
    manifest_path = tmp_path / "split_manifest.json"

    hidden_dir.mkdir(parents=True)
    evidence_dir.mkdir(parents=True)
    answers_path.write_text("id,value\n1,0.5\n")
    sample_path.write_text("id,value\n1,0.0\n")

    manifest = {
        "visible_input_dir": str(input_dir),
        "hidden_validation_dir": str(hidden_dir),
        "hidden_answers_path": str(answers_path),
        "hidden_sample_submission_path": str(sample_path),
        "competition_id": "demo-comp",
        "split_seed": "mlebench_prepare",
        "strategy_summary": "single hidden split",
        "visible_count": 9,
        "hidden_count": 1,
        "contract_version": "mlebench_lite_adapter_v3",
        "evidence_dir": str(evidence_dir),
        "authoritative_train_sources": ["train.csv"],
        "authoritative_hidden_validation_sources": ["hidden_validation/validation.csv"],
        "authoritative_hidden_answer_sources": [str(answers_path)],
        "hidden_answer_granularity": "direct",
        "split_fraction": 0.1,
    }
    manifest_path.write_text(json.dumps(manifest))

    monkeypatch.setattr(verify, "infer_competition_id", lambda cfg: "demo-comp")
    monkeypatch.setattr(
        verify,
        "_validate_evidence_bundle",
        lambda path: {"ok": True, "reason": "", "detail": "ok"},
    )
    monkeypatch.setattr(
        verify,
        "_validate_visible_top_level_layout",
        lambda cfg, path: {"ok": True, "reason": "", "detail": "ok"},
    )
    monkeypatch.setattr(
        verify,
        "_validate_sample_submission",
        lambda cfg, answers_path, sample_submission_path: {"ok": True, "reason": "", "detail": "ok"},
    )
    monkeypatch.setattr(
        verify,
        "_compute_split_alignment",
        lambda **kwargs: {
            "visible_training_count": 9,
            "hidden_answers_count": 1,
            "sample_validation_rows": 1,
            "visible_validation_count": 1,
            "hidden_answer_granularity": "direct",
            "overlap_train_answers_count": 0,
            "sample_ids_match_answers": True,
            "overlap_train_answers_examples": [],
            "visible_validation_examples": ["validation.csv"],
        },
    )
    monkeypatch.setattr(verify, "_build_evidence_summary", lambda path: "ok")

    result = verify.verify_split_artifacts(cfg=object(), manifest_path=manifest_path)

    assert result["ok"] is True
    assert result["manifest"]["contract_version"] == "mlebench_lite_adapter_v3"


def test_verify_single_split_ignores_hidden_file_count_mismatch(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    hidden_dir = input_dir / "hidden_validation"
    evidence_dir = tmp_path / "evidence"
    answers_path = tmp_path / "hidden_answers.csv"
    sample_path = tmp_path / "sampleHiddenValidationSubmission.csv"
    manifest_path = tmp_path / "split_manifest.json"

    hidden_dir.mkdir(parents=True)
    evidence_dir.mkdir(parents=True)
    answers_path.write_text("id,value\n1,0.5\n")
    sample_path.write_text("id,value\n1,0.0\n")

    manifest = {
        "visible_input_dir": str(input_dir),
        "hidden_validation_dir": str(hidden_dir),
        "hidden_answers_path": str(answers_path),
        "hidden_sample_submission_path": str(sample_path),
        "competition_id": "demo-comp",
        "split_seed": "mlebench_prepare",
        "strategy_summary": "single hidden split",
        "visible_count": 9,
        "hidden_count": 1,
        "contract_version": "mlebench_lite_adapter_v3",
        "evidence_dir": str(evidence_dir),
        "authoritative_train_sources": ["train.csv"],
        "authoritative_hidden_validation_sources": ["hidden_validation/validation.csv"],
        "authoritative_hidden_answer_sources": [str(answers_path)],
        "hidden_answer_granularity": "direct",
        "hidden_alignment_mode": "group_count",
        "split_fraction": 0.1,
    }
    manifest_path.write_text(json.dumps(manifest))

    monkeypatch.setattr(verify, "infer_competition_id", lambda cfg: "demo-comp")
    monkeypatch.setattr(
        verify,
        "_validate_evidence_bundle",
        lambda path: {"ok": True, "reason": "", "detail": "ok"},
    )
    monkeypatch.setattr(
        verify,
        "_validate_visible_top_level_layout",
        lambda cfg, path, contract_version=None: {"ok": True, "reason": "", "detail": "ok"},
    )
    monkeypatch.setattr(
        verify,
        "_validate_sample_submission",
        lambda cfg, answers_path, sample_submission_path: {"ok": True, "reason": "", "detail": "ok"},
    )
    monkeypatch.setattr(
        verify,
        "_compute_split_alignment",
        lambda **kwargs: {
            "visible_training_count": 0,
            "hidden_answers_count": 1,
            "sample_validation_rows": 1,
            "visible_validation_count": 999,
            "hidden_answer_granularity": "direct",
            "overlap_train_answers_count": 0,
            "sample_ids_match_answers": True,
            "overlap_train_answers_examples": [],
            "visible_validation_examples": ["validation.csv"],
        },
    )
    monkeypatch.setattr(verify, "_build_evidence_summary", lambda path: "ok")

    result = verify.verify_split_artifacts(cfg=object(), manifest_path=manifest_path)

    assert result["ok"] is True


def test_validate_sample_submission_uses_oracle_when_sample_scores_nan(tmp_path, monkeypatch):
    answers_path = tmp_path / "hidden_answers.csv"
    sample_path = tmp_path / "sampleHiddenValidationSubmission.csv"
    answers_path.write_text("id,score\n1,0.5\n")
    sample_path.write_text("id,score\n1,0.0\n")

    class DummyCompetition:
        def grader(self, submission, answers):
            row = submission.to_dicts()[0]
            if float(row["score"]) == 0.0:
                return float("nan")
            return 1.0

    class DummyFrame:
        def __init__(self, rows):
            self._rows = rows

        def to_dicts(self):
            return self._rows

    class DummyRegistry:
        def get_competition(self, _competition_id):
            return DummyCompetition()

    dummy_utils = types.SimpleNamespace(
        load_answers=lambda path: read_rows(path),
        read_csv=lambda path: DummyFrame(read_rows(path)),
    )

    def read_rows(path):
        import csv as _csv

        with Path(path).open() as f:
            return list(_csv.DictReader(f))

    monkeypatch.setitem(sys.modules, "mlebench.registry", types.SimpleNamespace(registry=DummyRegistry()))
    monkeypatch.setitem(sys.modules, "mlebench.utils", dummy_utils)
    monkeypatch.setattr(verify, "infer_competition_id", lambda cfg: "demo-comp")

    result = verify._validate_sample_submission(object(), answers_path, sample_path)

    assert result["ok"] is True
    assert "oracle_score=1.0" in result["detail"]


def test_prepare_hidden_validation_disables_unsupported_competition(tmp_path, monkeypatch):
    class HiddenCfg:
        enabled = True
        competition_name = "demo-comp"
        split_fraction = 0.1
        unsupported_policy = "disable"
        allow_self_report_fallback = True
        hard_fail_on_prepare_error = False
        stop_after_prepare = False

    cfg = types.SimpleNamespace(
        hidden_validation=HiddenCfg(),
        workspace_dir=tmp_path / "workspace",
    )
    cfg.workspace_dir.mkdir(parents=True)

    monkeypatch.setattr(
        "engine.hidden_validation.maybe_prepare_mlebench_lite_hidden_validation",
        lambda cfg: None,
    )
    monkeypatch.setattr("engine.hidden_validation.infer_competition_id", lambda cfg: "demo-comp")

    state = prepare_hidden_validation(cfg)

    assert state["active"] is False
    assert state["fallback_mode"] is False
    assert state["reviewer_status"] == "unsupported_competition"
