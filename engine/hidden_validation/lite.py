"""Deterministic mle-bench lite hidden-validation adapter preparation."""

import json
import shutil
from pathlib import Path
from typing import Any

from utils import preproc_data

from .common import (
    build_data_layout_summary,
    default_state,
    infer_competition_id,
    logger,
    read_csv_rows,
    save_runtime_state,
)
from .verify import verify_split_artifacts


GENERIC_DATAFRAME_COMPETITION_CONFIGS = {
    "aptos2019-blindness-detection": {
        "train_csv": "train.csv", "id": "id_code", "target": "diagnosis",
        "validation_cols": ["id_code"], "assets": [("train_images", "test_images", [".png"], None)],
    },
    "aerial-cactus-identification": {
        "train_csv": "train.csv", "id": "id", "target": "has_cactus",
        "validation_cols": ["id"], "assets": [("train", "test", [".jpg"], None)],
    },
    "dog-breed-identification": {
        "train_csv": "labels.csv", "id": "id", "target": "breed", "one_hot": True,
        "validation_cols": ["id"], "assets": [("train", "test", [".jpg"], None)],
    },
    "histopathologic-cancer-detection": {
        "train_csv": "train_labels.csv", "id": "id", "target": "label",
        "validation_cols": ["id"], "assets": [("train", "test", [".tif"], None)],
    },
    "jigsaw-toxic-comment-classification-challenge": {
        "train_csv": "train.csv", "id": "id",
        "target_cols": ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
        "validation_drop": ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
        "assets": [],
    },
    "jigsaw-unintended-bias-in-toxicity-classification": {
        "train_csv": "train.csv", "id": "id", "target": "target",
        "validation_cols": ["id", "comment_text"], "assets": [],
    },
    "leaf-classification": {
        "train_csv": "train.csv", "id": "id", "target": "species", "one_hot": True,
        "validation_cols": ["id"], "assets": [("images", "images", [".jpg"], "test.csv")],
    },
    "imet-2020-fgvc7": {
        "train_csv": "train.csv", "id": "id", "target": "attribute_ids",
        "validation_cols": ["id"], "assets": [("train", "test", [".png"], None)],
    },
    "new-york-city-taxi-fare-prediction": {
        "train_csv": "labels.csv", "id": "key", "target": "fare_amount",
        "validation_drop": ["fare_amount"], "assets": [],
    },
    "nomad2018-predict-transparent-conductors": {
        "train_csv": "train.csv", "id": "id",
        "target_cols": ["formation_energy_ev_natom", "bandgap_energy_ev"],
        "validation_drop": ["formation_energy_ev_natom", "bandgap_energy_ev"],
        "assets": [("train", "test", ["/geometry.xyz"], None)],
    },
    "plant-pathology-2020-fgvc7": {
        "train_csv": "train.csv", "id": "image_id",
        "target_cols": ["healthy", "multiple_diseases", "rust", "scab"],
        "validation_cols": ["image_id"], "assets": [("images", "images", [".jpg"], "test.csv")],
    },
    "plant-pathology-2021-fgvc8": {
        "train_csv": "train.csv", "id": "image", "target": "labels",
        "validation_cols": ["image"], "assets": [("train_images", "test_images", [".jpg"], None)],
    },
    "petfinder-pawpularity-score": {
        "train_csv": "train.csv", "id": "Id", "target": "Pawpularity",
        "validation_drop": ["Pawpularity"], "assets": [("train", "test", [".jpg"], None)],
    },
    "ranzcr-clip-catheter-line-classification": {
        "train_csv": "train.csv", "id": "StudyInstanceUID",
        "target_cols": [
            "ETT - Abnormal", "ETT - Borderline", "ETT - Normal",
            "NGT - Abnormal", "NGT - Borderline", "NGT - Incompletely Imaged",
            "NGT - Normal", "CVC - Abnormal", "CVC - Borderline",
        ],
        "validation_cols": ["StudyInstanceUID"], "assets": [("train", "test", [".jpg"], None)],
        "filter_annotations": True,
    },
    "siim-isic-melanoma-classification": {
        "train_csv": "train.csv", "id": "image_name", "target": "target",
        "validation_cols": ["image_name", "patient_id", "sex", "age_approx", "anatom_site_general_challenge"],
        "assets": [("train", "test", [".dcm"], None), ("jpeg/train", "jpeg/test", [".jpg"], None)],
    },
    "spooky-author-identification": {
        "train_csv": "train.csv", "id": "id", "target": "author", "one_hot": True,
        "validation_drop": ["author"], "assets": [],
    },
    "tweet-sentiment-extraction": {
        "train_csv": "train.csv", "id": "textID", "target": "selected_text",
        "validation_drop": ["selected_text"], "assets": [],
    },
    "tabular-playground-series-dec-2021": {
        "train_csv": "train.csv", "id": "Id", "target": "Cover_Type",
        "validation_drop": ["Cover_Type"], "assets": [],
    },
    "tabular-playground-series-may-2022": {
        "train_csv": "train.csv", "id": "id", "target": "target",
        "validation_drop": ["target"], "assets": [],
    },
    "us-patent-phrase-to-phrase-matching": {
        "train_csv": "train.csv", "id": "id", "target": "score",
        "validation_drop": ["score"], "assets": [],
    },
}

SPECIAL_CASE_DETERMINISTIC_COMPETITIONS = {
    "billion-word-imputation",
    "bms-molecular-translation",
    "cassava-leaf-disease-classification",
    "champs-scalar-coupling",
    "denoising-dirty-documents",
    "detecting-insults-in-social-commentary",
    "dogs-vs-cats-redux-kernels-edition",
    "freesound-audio-tagging-2019",
    "h-and-m-personalized-fashion-recommendations",
    "hms-harmful-brain-activity-classification",
    "hotel-id-2021-fgvc8",
    "hubmap-kidney-segmentation",
    "kuzushiji-recognition",
    "mlsp-2013-birds",
    "multi-modal-gesture-recognition",
    "nfl-player-contact-detection",
    "osic-pulmonary-fibrosis-progression",
    "random-acts-of-pizza",
    "smartphone-decimeter-2022",
    "stanford-covid-vaccine",
    "tensorflow2-question-answering",
    "text-normalization-challenge-english-language",
    "text-normalization-challenge-russian-language",
    "the-icml-2013-whale-challenge-right-whale-redux",
    "uw-madison-gi-tract-image-segmentation",
    "ventilator-pressure-prediction",
    "whale-categorization-playground",
}

DEFAULT_SPLIT_FRACTION = 0.10


def _configured_split_fraction() -> float:
    return max(1e-6, min(0.5, float(DEFAULT_SPLIT_FRACTION)))


def supported_deterministic_competition_ids() -> set[str]:
    return set(GENERIC_DATAFRAME_COMPETITION_CONFIGS) | set(
        SPECIAL_CASE_DETERMINISTIC_COMPETITIONS
    )


def _copy_path(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _copy_matching_files(src_dir: Path, dst_dir: Path, pattern: str) -> int:
    count = 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(src_dir.glob(pattern)):
        if src.is_file():
            shutil.copy2(src, dst_dir / src.name)
            count += 1
    return count


def _load_id_set(csv_path: Path, preferred_columns: list[str]) -> list[str]:
    rows = read_csv_rows(csv_path)
    ids: list[str] = []
    for row in rows:
        value = None
        for column in preferred_columns:
            value = row.get(column)
            if value not in (None, ""):
                break
        if value not in (None, ""):
            ids.append(str(value))
    return ids


def _copy_assets_by_id(src_dir: Path, dst_dir: Path, ids: list[str], suffixes: list[str]) -> int:
    copied = 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item_id in ids:
        for suffix in suffixes:
            src = src_dir / f"{item_id}{suffix}"
            if src.exists():
                shutil.copy2(src, dst_dir / src.name)
                copied += 1
                break
    return copied


def _copy_default_validation_layout(visible_output: Path, validation_dir: Path) -> list[str]:
    copied_rel_paths: list[str] = []
    for src in sorted(visible_output.iterdir()):
        if src.name in {
            "validation",
            "visible_validation",
            "hidden_validation",
            "sampleValidationSubmission.csv",
            "sampleVisibleValidationSubmission.csv",
            "visible_validation_answers.csv",
        }:
            continue
        lower_name = src.name.lower()
        if any(token in lower_name for token in ["sample_submission", "samplesubmission", "description"]):
            continue
        if "test" not in lower_name:
            continue
        dst = validation_dir / src.name
        _copy_path(src, dst)
        copied_rel_paths.append(str(src.relative_to(visible_output)))
    return copied_rel_paths


def _copy_leaf_validation_layout(visible_output: Path, validation_dir: Path) -> list[str]:
    test_csv = visible_output / "test.csv"
    image_ids = _load_id_set(test_csv, ["id"])
    _copy_path(test_csv, validation_dir / "test.csv")
    copied_images = _copy_assets_by_id(visible_output / "images", validation_dir / "images", image_ids, [".jpg"])
    if copied_images <= 0:
        raise RuntimeError("leaf-classification adapter could not copy validation images")
    return ["test.csv", "images"]


def _copy_plant_pathology_validation_layout(visible_output: Path, validation_dir: Path) -> list[str]:
    test_csv = visible_output / "test.csv"
    image_ids = _load_id_set(test_csv, ["image_id"])
    _copy_path(test_csv, validation_dir / "test.csv")
    copied_images = _copy_assets_by_id(visible_output / "images", validation_dir / "images", image_ids, [".jpg"])
    if copied_images <= 0:
        raise RuntimeError("plant-pathology adapter could not copy validation images")
    return ["test.csv", "images"]


def _copy_siim_validation_layout(visible_output: Path, validation_dir: Path) -> list[str]:
    copied: list[str] = []
    for rel_path in ["test.csv", "test", "jpeg/test"]:
        src = visible_output / rel_path
        if src.exists():
            _copy_path(src, validation_dir / rel_path)
            copied.append(rel_path)
    tfrecord_src = visible_output / "tfrecords"
    if tfrecord_src.exists():
        count = _copy_matching_files(tfrecord_src, validation_dir / "tfrecords", "test*.tfrec")
        if count:
            copied.append("tfrecords/test*.tfrec")
    if not copied:
        raise RuntimeError("siim-isic adapter found no validation sources")
    return copied


def _filter_mlsp_index_file(src: Path, dst: Path, keep_ids: set[int]) -> None:
    with src.open() as in_f, dst.open("w") as out_f:
        header = in_f.readline()
        if header:
            out_f.write(header)
        for line in in_f:
            stripped = line.strip()
            if not stripped:
                continue
            rec_str = stripped.split(",", 1)[0].strip()
            try:
                rec_id = int(rec_str)
            except ValueError:
                continue
            if rec_id in keep_ids:
                out_f.write(line)


def _load_mlsp_labels(path: Path):
    import pandas as pd

    rows: list[dict[str, Any]] = []
    lines = path.read_text(errors="ignore").splitlines()
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        rec_id_text, sep, labels_text = stripped.partition(",")
        try:
            rec_id = int(rec_id_text)
        except ValueError:
            continue
        rows.append({"rec_id": rec_id, "[labels]": labels_text if sep else ""})
    return pd.DataFrame(rows)


def _write_mlsp_labels(path: Path, labels_df) -> None:
    lines = ["rec_id,[labels]"]
    for _, row in labels_df.iterrows():
        rec_id = int(row["rec_id"])
        labels = str(row["[labels]"]).strip()
        if labels.lower() in {"nan", "none"}:
            labels = ""
        lines.append(f"{rec_id},{labels}" if labels else str(rec_id))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _copy_mlsp_validation_layout(visible_output: Path, validation_dir: Path) -> list[str]:
    essential_src = visible_output / "essential_data"
    supplemental_src = visible_output / "supplemental_data"
    essential_dst = validation_dir / "essential_data"
    supplemental_dst = validation_dir / "supplemental_data"
    essential_dst.mkdir(parents=True, exist_ok=True)
    supplemental_dst.mkdir(parents=True, exist_ok=True)

    cv_rows = read_csv_rows(essential_src / "CVfolds_2.txt")
    test_ids = {
        int(row["rec_id"])
        for row in cv_rows
        if row.get("rec_id") not in (None, "") and str(row.get("fold", "")).strip() == "1"
    }
    if not test_ids:
        raise RuntimeError("mlsp-2013-birds adapter found no test rec_ids")

    _filter_mlsp_index_file(essential_src / "CVfolds_2.txt", essential_dst / "CVfolds_2.txt", test_ids)
    _filter_mlsp_index_file(essential_src / "rec_id2filename.txt", essential_dst / "rec_id2filename.txt", test_ids)
    _filter_mlsp_index_file(
        essential_src / "rec_labels_test_hidden.txt",
        essential_dst / "rec_labels_test_hidden.txt",
        test_ids,
    )
    for plain_name in ["species_list.txt"]:
        src = essential_src / plain_name
        if src.exists():
            _copy_path(src, essential_dst / plain_name)

    rec_rows = read_csv_rows(essential_dst / "rec_id2filename.txt")
    filenames = [row.get("filename") for row in rec_rows if row.get("filename")]
    if not filenames:
        raise RuntimeError("mlsp-2013-birds adapter could not map rec_ids to filenames")

    for wav_name in filenames:
        src = essential_src / "src_wavs" / f"{wav_name}.wav"
        if src.exists():
            _copy_path(src, essential_dst / "src_wavs" / src.name)

    for dir_name in ["filtered_spectrograms", "segmentation_examples", "spectrograms", "supervised_segmentation"]:
        src_dir = supplemental_src / dir_name
        for stem in filenames:
            src = src_dir / f"{stem}.bmp"
            if src.exists():
                _copy_path(src, supplemental_dst / dir_name / src.name)

    for plain_name in ["segment_clusters.bmp", "segment_mosaic.bmp"]:
        src = supplemental_src / plain_name
        if src.exists():
            _copy_path(src, supplemental_dst / plain_name)

    for text_name in ["histogram_of_segments.txt", "segment_features.txt", "segment_rectangles.txt"]:
        src = supplemental_src / text_name
        if src.exists():
            _filter_mlsp_index_file(src, supplemental_dst / text_name, test_ids)

    return ["essential_data", "supplemental_data"]


def _collect_authoritative_train_sources(visible_output: Path) -> list[str]:
    sources: list[str] = []
    for src in sorted(visible_output.iterdir()):
        if src.name in {
            "validation",
            "visible_validation",
            "hidden_validation",
            "sampleValidationSubmission.csv",
            "sampleVisibleValidationSubmission.csv",
            "visible_validation_answers.csv",
        }:
            continue
        lower_name = src.name.lower()
        if any(token in lower_name for token in ["sample_submission", "samplesubmission", "description"]):
            continue
        if "test" in lower_name:
            continue
        sources.append(str(src))
    return sources


def _rewrite_validation_sources(
    sources: list[str],
    validation_dir: Path,
) -> list[str]:
    rewritten: list[str] = []
    prefix = validation_dir.name
    for source in sources:
        if source == "validation":
            rewritten.append(prefix)
        elif source.startswith("validation/"):
            rewritten.append(f"{prefix}/{source.split('/', 1)[1]}")
        else:
            rewritten.append(source)
    return rewritten


def _estimate_visible_count(visible_output: Path, authoritative_train_sources: list[str]) -> int:
    manifest_stub = {"authoritative_train_sources": authoritative_train_sources}
    from .verify import _compute_visible_training_count

    return _compute_visible_training_count(visible_output, manifest_stub, read_csv_rows(visible_output / "train.csv"))


def _write_adapter_evidence(
    cfg,
    visible_output: Path,
    hidden_output: Path,
    evidence_dir: Path,
    authoritative_train_sources: list[str],
    authoritative_validation_sources: list[str],
    authoritative_hidden_answer_sources: list[str],
    strategy_summary: str,
) -> None:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "source_layout.txt").write_text(build_data_layout_summary(Path(cfg.data_dir)))
    (evidence_dir / "visible_layout.txt").write_text(build_data_layout_summary(visible_output))
    (evidence_dir / "hidden_layout.txt").write_text(build_data_layout_summary(hidden_output))
    (evidence_dir / "authoritative_sources.txt").write_text(
        "\n".join(
            [
                "train_sources:",
                *authoritative_train_sources,
                "",
                "validation_sources:",
                *authoritative_validation_sources,
                "",
                "hidden_answer_sources:",
                *authoritative_hidden_answer_sources,
            ]
        )
    )
    (evidence_dir / "notes.txt").write_text(strategy_summary)


def _load_sample_submission_template(visible_output: Path, competition_id: str):
    import pandas as pd

    for name in ["sample_submission.csv", "sampleSubmission.csv", "en_sample_submission_2.csv", "ru_sample_submission_2.csv"]:
        path = visible_output / name
        if path.exists():
            return pd.read_csv(path), path
    raise FileNotFoundError(
        f"Could not find a sample submission template in {visible_output} for {competition_id}"
    )


def _build_placeholder_submission(answers_df, sample_template_df, id_column: str):
    import pandas as pd

    submission = pd.DataFrame({id_column: answers_df[id_column].values})
    for col in sample_template_df.columns:
        if col == id_column:
            continue
        fill_value = sample_template_df[col].iloc[0] if len(sample_template_df) else 0
        submission[col] = fill_value
    return submission


def _split_df_basic(df, competition_id: str, visible_output: Path):
    from sklearn.model_selection import train_test_split

    stratify = None
    split_fraction = _configured_split_fraction()
    if competition_id == "aerial-cactus-identification":
        test_size = 0.19
    elif competition_id == "leaf-classification":
        # Leaf has 99 classes and only 891 training rows, so use a larger
        # stratified holdout to keep the hidden validation split representative.
        test_size = 0.2
        stratify = df["species"]
    elif competition_id == "new-york-city-taxi-fare-prediction":
        test_size = min(9914, max(1, int(len(df) * split_fraction)))
    elif competition_id == "tabular-playground-series-may-2022":
        test_size = min(100_000, max(1, int(len(df) * split_fraction)))
    elif competition_id == "histopathologic-cancer-detection":
        existing_test = len(list((visible_output / "test").glob("*.tif")))
        test_size = max(1, min(len(df) - 1, existing_test))
    else:
        test_size = split_fraction
    return train_test_split(df, test_size=test_size, random_state=0, stratify=stratify)


def _split_df_by_group(df, group_col: str, test_size: float | None = None):
    from sklearn.model_selection import train_test_split

    if test_size is None:
        test_size = _configured_split_fraction()
    group_values = sorted(df[group_col].dropna().unique().tolist())
    keep_groups, validation_groups = train_test_split(
        group_values,
        test_size=test_size,
        random_state=0,
    )
    keep_df = df[df[group_col].isin(set(keep_groups))].copy()
    validation_df = df[df[group_col].isin(set(validation_groups))].copy()
    return keep_df, validation_df


def _split_asset_dir_with_ids(
    source_dir: Path,
    train_target_dir: Path,
    validation_target_dir: Path,
    train_ids: list[str],
    validation_ids: list[str],
    suffixes: list[str],
    extra_keep_ids: list[str] | None = None,
    validation_rename=None,
) -> None:
    backup_dir = source_dir.parent / f".{source_dir.name}_hidden_validation_source"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if source_dir.exists():
        source_dir.rename(backup_dir)
    train_target_dir.mkdir(parents=True, exist_ok=True)
    validation_target_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_src(item_id: str) -> Path | None:
        for suffix in suffixes:
            if suffix.startswith("/"):
                candidate = backup_dir / item_id / suffix.lstrip("/")
                if candidate.exists():
                    return candidate
                continue

            # Some prepared datasets already store the full filename, including
            # extension, in the id column (for example aerial-cactus ids end in
            # ".jpg"). Prefer the exact filename first, then fall back to
            # appending the configured suffix for extension-less ids.
            exact_candidate = backup_dir / item_id
            if exact_candidate.exists():
                return exact_candidate

            candidate = backup_dir / f"{item_id}{suffix}"
            if candidate.exists():
                return candidate
        return None

    for item_id in list(train_ids) + list(extra_keep_ids or []):
        src = _resolve_src(str(item_id))
        if src is None:
            continue
        dst = train_target_dir / src.name if src.parent == backup_dir else train_target_dir / item_id / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    for idx, item_id in enumerate(validation_ids, start=1):
        src = _resolve_src(str(item_id))
        if src is None:
            continue
        if validation_rename is None:
            dst = validation_target_dir / src.name if src.parent == backup_dir else validation_target_dir / item_id / src.name
        else:
            dst = validation_target_dir / validation_rename(item_id, idx, src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    if backup_dir.exists():
        shutil.rmtree(backup_dir)


def _write_train_count_manifest(visible_output: Path, rows: list[dict[str, Any]]) -> str:
    import pandas as pd

    path = visible_output / "hidden_validation_train_rows.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def _prepare_generic_dataframe_second_split(
    competition_id: str,
    visible_output: Path,
    validation_dir: Path,
    hidden_answers_path: Path,
    hidden_sample_submission_path: Path,
    visible_sample_submission_path: Path,
) -> dict[str, Any]:
    import pandas as pd
    from mlebench.competitions.utils import df_to_one_hot

    config = GENERIC_DATAFRAME_COMPETITION_CONFIGS[competition_id]

    train_csv_path = visible_output / config["train_csv"]
    df = pd.read_csv(train_csv_path)
    train_df, validation_df = _split_df_basic(df, competition_id, visible_output)
    sample_template_df, _ = _load_sample_submission_template(visible_output, competition_id)
    id_column = config["id"]

    if config.get("one_hot"):
        classes = [col for col in sample_template_df.columns if col != id_column]
        answers_df = df_to_one_hot(validation_df[[id_column, config["target"]]], id_column, config["target"], classes)
    elif config.get("target_cols"):
        answers_df = validation_df[[id_column] + config["target_cols"]].copy()
    else:
        answers_df = validation_df[[id_column, config["target"]]].copy()

    if config.get("validation_cols"):
        validation_visible_df = validation_df[config["validation_cols"]].copy()
    elif config.get("validation_drop"):
        validation_visible_df = validation_df.drop(columns=config["validation_drop"]).copy()
    else:
        drop_cols = [config["target"]] if config.get("target") else config.get("target_cols", [])
        validation_visible_df = validation_df.drop(columns=drop_cols).copy()

    train_df.to_csv(train_csv_path, index=False)
    if config.get("filter_annotations"):
        ann = pd.read_csv(visible_output / "train_annotations.csv")
        ann = ann[ann[id_column].isin(train_df[id_column])]
        ann.to_csv(visible_output / "train_annotations.csv", index=False)

    validation_visible_df.to_csv(validation_dir / "validation.csv", index=False)
    answers_df.to_csv(hidden_answers_path, index=False)
    placeholder_df = _build_placeholder_submission(answers_df, sample_template_df, id_column)
    placeholder_df.to_csv(hidden_sample_submission_path, index=False)
    placeholder_df.to_csv(visible_sample_submission_path, index=False)

    existing_public_test_ids: list[str] = []
    if (visible_output / "test.csv").exists():
        test_df = pd.read_csv(visible_output / "test.csv")
        if id_column in test_df.columns:
            existing_public_test_ids = [str(v) for v in test_df[id_column].tolist()]

    for src_rel, val_rel, suffixes, preserve_from_csv in config.get("assets", []):
        preserve_ids = existing_public_test_ids if preserve_from_csv == "test.csv" else []
        _split_asset_dir_with_ids(
            visible_output / src_rel,
            visible_output / src_rel,
            validation_dir / val_rel,
            [str(v) for v in train_df[id_column].tolist()],
            [str(v) for v in validation_df[id_column].tolist()],
            suffixes,
            extra_keep_ids=preserve_ids,
        )

    return {
        "authoritative_train_sources": _collect_authoritative_train_sources(visible_output),
        "authoritative_validation_sources": ["validation/validation.csv"],
        "hidden_answer_granularity": "direct",
    }


def _prepare_whale_categorization_second_split(
    visible_output,
    validation_dir,
    hidden_answers_path,
    hidden_sample_submission_path,
    visible_sample_submission_path,
):
    import pandas as pd

    train_df = pd.read_csv(visible_output / "train.csv")
    keep_df, val_df = _split_df_basic(train_df, "whale-categorization-playground", visible_output)
    unseen_ids = set(val_df["Id"]) - set(keep_df["Id"])
    answers_df = val_df[["Image", "Id"]].copy()
    if unseen_ids:
        answers_df.loc[answers_df["Id"].isin(unseen_ids), "Id"] = "new_whale"

    keep_df.to_csv(visible_output / "train.csv", index=False)
    val_df[["Image"]].to_csv(validation_dir / "validation.csv", index=False)
    answers_df.to_csv(hidden_answers_path, index=False)

    sample_df = pd.DataFrame({"Image": answers_df["Image"], "Id": "new_whale w_1287fbc w_98baff9 w_7554f44 w_1eafe46"})
    sample_df.to_csv(hidden_sample_submission_path, index=False)
    sample_df.to_csv(visible_sample_submission_path, index=False)
    _split_asset_dir_with_ids(
        visible_output / "train",
        visible_output / "train",
        validation_dir / "validation",
        [str(v) for v in keep_df["Image"].tolist()],
        [str(v) for v in val_df["Image"].tolist()],
        [""],
    )
    return {
        "authoritative_train_sources": [str(visible_output / "train.csv"), str(visible_output / "train")],
        "authoritative_validation_sources": ["validation/validation.csv"],
        "hidden_answer_granularity": "direct",
    }


def _prepare_ventilator_second_split(
    visible_output,
    validation_dir,
    hidden_answers_path,
    hidden_sample_submission_path,
    visible_sample_submission_path,
):
    import pandas as pd

    train_df = pd.read_csv(visible_output / "train.csv")
    keep_df, val_df = _split_df_by_group(train_df, "breath_id")
    keep_df = keep_df.copy()
    val_df = val_df.copy()
    keep_df["id"] = range(1, len(keep_df) + 1)
    val_df["id"] = range(1, len(val_df) + 1)

    keep_df.to_csv(visible_output / "train.csv", index=False, float_format="%.10g")
    val_df.drop(columns=["pressure"]).to_csv(
        validation_dir / "validation.csv", index=False, float_format="%.10g"
    )
    val_df[["id", "pressure"]].to_csv(hidden_answers_path, index=False, float_format="%.10g")
    sample_df = pd.DataFrame({"id": val_df["id"], "pressure": 0.0})
    sample_df.to_csv(hidden_sample_submission_path, index=False, float_format="%.10g")
    sample_df.to_csv(visible_sample_submission_path, index=False, float_format="%.10g")
    return {
        "authoritative_train_sources": [str(visible_output / "train.csv")],
        "authoritative_validation_sources": ["validation/validation.csv"],
        "hidden_answer_granularity": "direct",
    }


def _prepare_champs_second_split(
    visible_output,
    validation_dir,
    hidden_answers_path,
    hidden_sample_submission_path,
    visible_sample_submission_path,
):
    import pandas as pd

    train_df = pd.read_csv(visible_output / "train.csv")
    keep_df, val_df = _split_df_by_group(train_df, "molecule_name")
    keep_molecules = set(keep_df["molecule_name"])

    keep_df.to_csv(visible_output / "train.csv", index=False)
    val_df.drop(columns=["scalar_coupling_constant"]).to_csv(validation_dir / "validation.csv", index=False)
    val_df[["id", "scalar_coupling_constant"]].to_csv(hidden_answers_path, index=False)
    sample_df = pd.DataFrame({"id": val_df["id"], "scalar_coupling_constant": 0.0})
    sample_df.to_csv(hidden_sample_submission_path, index=False)
    sample_df.to_csv(visible_sample_submission_path, index=False)

    for name in [
        "structures.csv",
        "dipole_moments.csv",
        "magnetic_shielding_tensors.csv",
        "mulliken_charges.csv",
        "potential_energy.csv",
        "scalar_coupling_contributions.csv",
    ]:
        path = visible_output / name
        if path.exists():
            df = pd.read_csv(path)
            df[df["molecule_name"].isin(keep_molecules)].to_csv(path, index=False)

    _split_asset_dir_with_ids(
        visible_output / "structures",
        visible_output / "structures",
        validation_dir / "_unused",
        sorted(keep_molecules),
        [],
        [".xyz"],
    )
    if (validation_dir / "_unused").exists():
        shutil.rmtree(validation_dir / "_unused")

    return {
        "authoritative_train_sources": _collect_authoritative_train_sources(visible_output),
        "authoritative_validation_sources": ["validation/validation.csv"],
        "hidden_answer_granularity": "direct",
    }


def _prepare_osic_second_split(
    visible_output,
    validation_dir,
    hidden_answers_path,
    hidden_sample_submission_path,
    visible_sample_submission_path,
):
    import pandas as pd

    train_df = pd.read_csv(visible_output / "train.csv")
    keep_df, val_df = _split_df_by_group(train_df, "Patient")
    keep_patients = set(keep_df["Patient"])
    val_patients = set(val_df["Patient"])

    keep_df.to_csv(visible_output / "train.csv", index=False)
    val_public = val_df.sort_values(by="Weeks").groupby("Patient").first().reset_index()
    val_public.to_csv(validation_dir / "validation.csv", index=False)

    all_weeks = pd.DataFrame(
        [
            (patient, week)
            for patient in sorted(val_patients)
            for week in range(int(val_df["Weeks"].min()), int(val_df["Weeks"].max()) + 1)
        ],
        columns=["Patient", "Weeks"],
    )
    answers_df = all_weeks.merge(val_df, on=["Patient", "Weeks"], how="left")
    answers_df["Patient_Week"] = answers_df["Patient"] + "_" + answers_df["Weeks"].astype(str)
    answers_df["Confidence"] = 100
    # OSIC can contain multiple clinical rows for the same Patient_Week.
    # The submission contract is keyed by unique Patient_Week, so normalize
    # to one row per key before writing validation targets and answers.
    answers_df = answers_df.sort_values(["Patient", "Weeks"]).drop_duplicates(
        subset=["Patient_Week"],
        keep="first",
    )
    answers_df.to_csv(hidden_answers_path, index=False)
    answers_df[["Patient_Week"]].to_csv(validation_dir / "validation_targets.csv", index=False)

    sample_df = answers_df[["Patient_Week"]].copy()
    sample_df["FVC"] = 2000
    sample_df["Confidence"] = 100
    sample_df.to_csv(hidden_sample_submission_path, index=False)
    sample_df.to_csv(visible_sample_submission_path, index=False)

    backup = visible_output / ".train_hidden_validation_source"
    if backup.exists():
        shutil.rmtree(backup)
    if (visible_output / "train").exists():
        (visible_output / "train").rename(backup)
        (visible_output / "train").mkdir(parents=True, exist_ok=True)
        for patient in sorted(keep_patients):
            src = backup / patient
            if src.exists():
                shutil.copytree(src, visible_output / "train" / patient, dirs_exist_ok=True)
        for patient in sorted(val_patients):
            src = backup / patient
            if src.exists():
                shutil.copytree(src, validation_dir / patient, dirs_exist_ok=True)
        shutil.rmtree(backup)

    return {
        "authoritative_train_sources": [str(visible_output / "train.csv"), str(visible_output / "train")],
        "authoritative_validation_sources": ["validation/validation_targets.csv"],
        "hidden_answer_granularity": "direct",
        "hidden_alignment_mode": "group_count",
    }


def _prepare_hotel_second_split(
    visible_output,
    validation_dir,
    hidden_answers_path,
    hidden_sample_submission_path,
    visible_sample_submission_path,
):
    import pandas as pd

    train_df = pd.read_csv(visible_output / "train.csv")
    if "image" not in train_df.columns:
        raise RuntimeError("hotel-id adapter expected train.csv with image column")
    keep_df, val_df = _split_df_basic(train_df, "hotel-id-2021-fgvc8", visible_output)
    keep_df.to_csv(visible_output / "train.csv", index=False)
    val_df[["image"]].to_csv(validation_dir / "validation.csv", index=False)
    val_df[["image", "hotel_id"]].to_csv(hidden_answers_path, index=False)
    sample_df = pd.DataFrame(
        {"image": val_df["image"], "hotel_id": "36363 53586 18807 64314 60181"}
    )
    sample_df.to_csv(hidden_sample_submission_path, index=False)
    sample_df.to_csv(visible_sample_submission_path, index=False)

    backup = visible_output / ".train_images_hidden_validation_source"
    if backup.exists():
        shutil.rmtree(backup)
    (visible_output / "train_images").rename(backup)
    (visible_output / "train_images").mkdir(parents=True, exist_ok=True)
    (validation_dir / "validation_images").mkdir(parents=True, exist_ok=True)
    for _, row in keep_df.iterrows():
        chain = str(row["chain"])
        src = backup / chain / row["image"]
        if src.exists():
            dst = visible_output / "train_images" / chain / row["image"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    for _, row in val_df.iterrows():
        chain = str(row["chain"])
        src = backup / chain / row["image"]
        if src.exists():
            shutil.copy2(src, validation_dir / "validation_images" / row["image"])
    shutil.rmtree(backup)

    return {
        "authoritative_train_sources": [str(visible_output / "train.csv"), str(visible_output / "train_images")],
        "authoritative_validation_sources": ["validation/validation.csv"],
        "hidden_answer_granularity": "direct",
        "hidden_alignment_mode": "id_count",
    }


def _prepare_stanford_second_split(
    visible_output,
    validation_dir,
    hidden_answers_path,
    hidden_sample_submission_path,
    visible_sample_submission_path,
):
    import pandas as pd

    train_df = pd.read_json(visible_output / "train.json", lines=True)
    test_template_df = pd.read_json(visible_output / "test.json", lines=True)
    to_predict = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    n_test_samples = max(1, int(len(train_df) * _configured_split_fraction()))
    test_indices = train_df[train_df["SN_filter"] > 0].sample(n=n_test_samples, random_state=0).index
    keep_df = train_df.drop(index=test_indices).copy()
    val_df = train_df.loc[test_indices].copy()

    records = []
    for _, row in val_df.iterrows():
        n = int(row["seq_scored"])
        k = int(row["seq_length"])
        for j in range(n):
            records.append(
                {
                    "id_seqpos": f"{row['id']}_{j}",
                    "reactivity": row["reactivity"][j],
                    "deg_Mg_pH10": row["deg_Mg_pH10"][j],
                    "deg_pH10": row["deg_pH10"][j],
                    "deg_Mg_50C": row["deg_Mg_50C"][j],
                    "deg_50C": row["deg_50C"][j],
                }
            )
        for j in range(n, k):
            records.append(
                {
                    "id_seqpos": f"{row['id']}_{j}",
                    "reactivity": 0.0,
                    "deg_Mg_pH10": 0.0,
                    "deg_pH10": 0.0,
                    "deg_Mg_50C": 0.0,
                    "deg_50C": 0.0,
                }
            )

    keep_df["index"] = range(len(keep_df))
    keep_df.to_json(visible_output / "train.json", orient="records", lines=True)
    val_public = val_df[test_template_df.columns].copy()
    val_public["index"] = range(len(val_public))
    val_public.to_json(validation_dir / "validation.json", orient="records", lines=True)

    answers_df = pd.DataFrame(records)
    answers_df.to_csv(hidden_answers_path, index=False, float_format="%.10f")
    sample_df = answers_df.copy()
    sample_df.loc[:, to_predict] = 0.0
    sample_df.to_csv(hidden_sample_submission_path, index=False, float_format="%.10f")
    sample_df.to_csv(visible_sample_submission_path, index=False, float_format="%.10f")

    return {
        "authoritative_train_sources": [str(visible_output / "train.json")],
        "authoritative_validation_sources": ["validation/validation.json"],
        "hidden_answer_granularity": "direct",
        "hidden_alignment_mode": "group_count",
    }


def _prepare_bms_second_split(
    visible_output,
    validation_dir,
    hidden_answers_path,
    hidden_sample_submission_path,
    visible_sample_submission_path,
):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    def _make_image_subpath(image_id: str) -> Path:
        return Path(image_id[0]) / image_id[1] / image_id[2] / f"{image_id}.png"

    train_df = pd.read_csv(visible_output / "train_labels.csv")
    keep_df, val_df = train_test_split(train_df, test_size=0.2, random_state=0)
    keep_df.to_csv(visible_output / "train_labels.csv", index=False)
    val_df[["image_id"]].to_csv(validation_dir / "validation.csv", index=False)
    val_df.to_csv(hidden_answers_path, index=False)
    sample_df = val_df[["image_id"]].copy()
    sample_df["InChI"] = "InChI=1S/H2O/h1H2"
    sample_df.to_csv(hidden_sample_submission_path, index=False)
    sample_df.to_csv(visible_sample_submission_path, index=False)

    backup = visible_output / ".train_hidden_validation_source"
    if backup.exists():
        shutil.rmtree(backup)
    (visible_output / "train").rename(backup)
    (visible_output / "train").mkdir(parents=True, exist_ok=True)
    for image_id in keep_df["image_id"].tolist():
        rel = _make_image_subpath(str(image_id))
        src = backup / rel
        dst = visible_output / "train" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, dst)
    for image_id in val_df["image_id"].tolist():
        rel = _make_image_subpath(str(image_id))
        src = backup / rel
        dst = validation_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, dst)
    shutil.rmtree(backup)

    return {
        "authoritative_train_sources": [str(visible_output / "train_labels.csv"), str(visible_output / "train")],
        "authoritative_validation_sources": ["validation/validation.csv"],
        "hidden_answer_granularity": "direct",
    }


def _prepare_h_and_m_second_split(
    visible_output,
    validation_dir,
    hidden_answers_path,
    hidden_sample_submission_path,
    visible_sample_submission_path,
):
    import pandas as pd

    train_df = pd.read_csv(visible_output / "transactions_train.csv")
    train_df["t_dat_parsed"] = pd.to_datetime(train_df["t_dat"])
    max_date = train_df["t_dat_parsed"].max()
    keep_df = train_df[train_df["t_dat_parsed"] < (max_date - pd.Timedelta(days=7))].copy()
    val_df = train_df[train_df["t_dat_parsed"] >= (max_date - pd.Timedelta(days=7))].copy()
    keep_df.drop(columns=["t_dat_parsed"], inplace=True)
    val_df.drop(columns=["t_dat_parsed"], inplace=True)

    keep_df.to_csv(visible_output / "transactions_train.csv", index=False)
    answers_df = (
        val_df.groupby("customer_id")["article_id"]
        .apply(lambda x: " ".join(x.astype(str)))
        .reset_index()
        .rename(columns={"article_id": "prediction"})
    )
    answers_df.to_csv(hidden_answers_path, index=False)
    answers_df[["customer_id"]].to_csv(validation_dir / "validation.csv", index=False)
    sample_df = answers_df.copy()
    sample_df["prediction"] = ""
    sample_df.to_csv(hidden_sample_submission_path, index=False)
    sample_df.to_csv(visible_sample_submission_path, index=False)
    return {
        "authoritative_train_sources": _collect_authoritative_train_sources(visible_output),
        "authoritative_validation_sources": ["validation/validation.csv"],
        "hidden_answer_granularity": "direct",
        "hidden_alignment_mode": "group_count",
    }


def _prepare_cassava_second_split(
    visible_output,
    validation_dir,
    hidden_answers_path,
    hidden_sample_submission_path,
    visible_sample_submission_path,
):
    import pandas as pd
    from mlebench.competitions.utils import get_ids_from_tf_records
    from sklearn.model_selection import train_test_split

    tfrecord_files = [
        path
        for path in sorted((visible_output / "train_tfrecords").iterdir())
        if path.is_file() and path.suffix == ".tfrec"
    ]
    keep_tfrec, val_tfrec = train_test_split(
        tfrecord_files,
        test_size=_configured_split_fraction(),
        random_state=0,
    )
    val_ids: list[str] = []
    for path in val_tfrec:
        val_ids.extend(get_ids_from_tf_records(path))

    train_df = pd.read_csv(visible_output / "train.csv")
    keep_df = train_df[~train_df["image_id"].isin(val_ids)].copy()
    val_df = train_df[train_df["image_id"].isin(val_ids)].copy()
    keep_df.to_csv(visible_output / "train.csv", index=False)
    val_df[["image_id"]].to_csv(validation_dir / "validation.csv", index=False)
    val_df.to_csv(hidden_answers_path, index=False)
    sample_df = val_df[["image_id"]].copy()
    sample_df["label"] = 4
    sample_df.to_csv(hidden_sample_submission_path, index=False)
    sample_df.to_csv(visible_sample_submission_path, index=False)

    backup = visible_output / ".train_tfrecords_hidden_validation_source"
    if backup.exists():
        shutil.rmtree(backup)
    (visible_output / "train_tfrecords").rename(backup)
    (visible_output / "train_tfrecords").mkdir(parents=True, exist_ok=True)
    (validation_dir / "validation_tfrecords").mkdir(parents=True, exist_ok=True)
    for path in keep_tfrec:
        shutil.copy2(backup / path.name, visible_output / "train_tfrecords" / path.name)
    for path in val_tfrec:
        shutil.copy2(backup / path.name, validation_dir / "validation_tfrecords" / path.name)
    shutil.rmtree(backup)

    _split_asset_dir_with_ids(
        visible_output / "train_images",
        visible_output / "train_images",
        validation_dir / "validation_images",
        [str(v) for v in keep_df["image_id"].tolist()],
        [str(v) for v in val_df["image_id"].tolist()],
        [""],
    )
    return {
        "authoritative_train_sources": [str(visible_output / "train.csv"), str(visible_output / "train_tfrecords"), str(visible_output / "train_images")],
        "authoritative_validation_sources": ["validation/validation.csv"],
        "hidden_answer_granularity": "direct",
        "hidden_alignment_mode": "id_count",
    }


def _prepare_hms_second_split(
    visible_output,
    validation_dir,
    hidden_answers_path,
    hidden_sample_submission_path,
    visible_sample_submission_path,
):
    import pandas as pd

    train_df = pd.read_csv(visible_output / "train.csv")
    keep_df, val_df = _split_df_by_group(train_df, "spectrogram_id")
    sample_template_df, _ = _load_sample_submission_template(
        visible_output,
        "hms-harmful-brain-activity-classification",
    )
    target_cols = [col for col in sample_template_df.columns if col != "eeg_id"]

    keep_df.to_csv(visible_output / "train.csv", index=False)
    val_df[["spectrogram_id", "eeg_id", "patient_id"]].to_csv(validation_dir / "validation.csv", index=False)
    answers_df = val_df[["eeg_id"] + target_cols].copy()
    answers_df[target_cols] = answers_df[target_cols].div(answers_df[target_cols].sum(axis=1), axis=0)
    answers_df.to_csv(hidden_answers_path, index=False)
    sample_df = answers_df.copy()
    sample_df[target_cols] = 1 / len(target_cols)
    sample_df.to_csv(hidden_sample_submission_path, index=False)
    sample_df.to_csv(visible_sample_submission_path, index=False)

    keep_eegs = [str(v) for v in sorted(keep_df["eeg_id"].dropna().unique().tolist())]
    val_eegs = [str(v) for v in sorted(val_df["eeg_id"].dropna().unique().tolist())]
    keep_specs = [str(v) for v in sorted(keep_df["spectrogram_id"].dropna().unique().tolist())]
    val_specs = [str(v) for v in sorted(val_df["spectrogram_id"].dropna().unique().tolist())]
    _split_asset_dir_with_ids(
        visible_output / "train_eegs",
        visible_output / "train_eegs",
        validation_dir / "validation_eegs",
        keep_eegs,
        val_eegs,
        [".parquet"],
    )
    _split_asset_dir_with_ids(
        visible_output / "train_spectrograms",
        visible_output / "train_spectrograms",
        validation_dir / "validation_spectrograms",
        keep_specs,
        val_specs,
        [".parquet"],
    )
    return {
        "authoritative_train_sources": [str(visible_output / "train.csv"), str(visible_output / "train_eegs"), str(visible_output / "train_spectrograms")],
        "authoritative_validation_sources": ["validation/validation.csv"],
        "hidden_answer_granularity": "direct",
        "hidden_alignment_mode": "group_count",
    }


def _prepare_smartphone_second_split(
    visible_output,
    validation_dir,
    hidden_answers_path,
    hidden_sample_submission_path,
    visible_sample_submission_path,
):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    def _get_date(folder_name: str) -> str:
        year, month, day, *_rest = folder_name.split("-")
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

    train_ids = sorted([folder.name for folder in (visible_output / "train").glob("*") if folder.is_dir()])
    dates = sorted({_get_date(name) for name in train_ids})
    keep_dates, val_dates = train_test_split(
        dates,
        test_size=_configured_split_fraction(),
        random_state=0,
    )
    keep_ids = [name for name in train_ids if _get_date(name) in set(keep_dates)]
    val_ids = [name for name in train_ids if _get_date(name) in set(val_dates)]

    backup = visible_output / ".train_hidden_validation_source"
    if backup.exists():
        shutil.rmtree(backup)
    (visible_output / "train").rename(backup)
    (visible_output / "train").mkdir(parents=True, exist_ok=True)
    for folder_name in keep_ids:
        shutil.copytree(backup / folder_name, visible_output / "train" / folder_name, dirs_exist_ok=True)
    for folder_name in val_ids:
        shutil.copytree(backup / folder_name, validation_dir / folder_name, dirs_exist_ok=True)
    shutil.rmtree(backup)

    dfs = []
    for fpath in sorted(validation_dir.rglob("ground_truth.csv")):
        drive_id = fpath.parent.parent.name
        phone_id = fpath.parent.name
        raw_df = pd.read_csv(fpath)
        df = raw_df.copy()
        df.loc[:, "tripId"] = f"{drive_id}-{phone_id}"
        df = df[["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]]
        dfs.append(df)
        fpath.unlink()
    answers_df = pd.concat(dfs, ignore_index=True)
    answers_df.to_csv(hidden_answers_path, index=False)
    val_visible_df = answers_df[["tripId", "UnixTimeMillis"]].copy()
    val_visible_df.to_csv(validation_dir / "validation.csv", index=False)
    sample_df = answers_df.copy()
    sample_df.loc[:, "LatitudeDegrees"] = 37.904611315634504
    sample_df.loc[:, "LongitudeDegrees"] = -86.48107806249548
    sample_df.to_csv(hidden_sample_submission_path, index=False)
    sample_df.to_csv(visible_sample_submission_path, index=False)
    return {
        "authoritative_train_sources": [str(visible_output / "train")],
        "authoritative_validation_sources": ["validation/validation.csv"],
        "hidden_answer_granularity": "direct",
        "hidden_alignment_mode": "group_count",
    }


def _prepare_nfl_second_split(
    visible_output,
    validation_dir,
    hidden_answers_path,
    hidden_sample_submission_path,
    visible_sample_submission_path,
):
    import pandas as pd

    train_df = pd.read_csv(visible_output / "train_labels.csv")
    keep_df, val_df = _split_df_by_group(train_df, "game_play")
    keep_game_plays = set(keep_df["game_play"])
    val_game_plays = set(val_df["game_play"])

    keep_df.to_csv(visible_output / "train_labels.csv", index=False)
    val_df[["contact_id"]].to_csv(validation_dir / "validation.csv", index=False)
    val_df[["contact_id", "contact"]].to_csv(hidden_answers_path, index=False)
    sample_df = pd.DataFrame({"contact_id": val_df["contact_id"], "contact": 0})
    sample_df.to_csv(hidden_sample_submission_path, index=False)
    sample_df.to_csv(visible_sample_submission_path, index=False)

    for src_name, key in [
        ("train_baseline_helmets.csv", "game_play"),
        ("train_player_tracking.csv", "game_play"),
        ("train_video_metadata.csv", "game_play"),
    ]:
        df = pd.read_csv(visible_output / src_name)
        df[df[key].isin(keep_game_plays)].to_csv(visible_output / src_name, index=False)
        df[df[key].isin(val_game_plays)].to_csv(validation_dir / src_name.replace("train_", "validation_"), index=False)

    backup = visible_output / ".train_hidden_validation_source"
    if backup.exists():
        shutil.rmtree(backup)
    (visible_output / "train").rename(backup)
    (visible_output / "train").mkdir(parents=True, exist_ok=True)
    for path in backup.glob("*.mp4"):
        game_play = path.name.rsplit("_", 1)[0]
        if game_play in keep_game_plays:
            shutil.copy2(path, visible_output / "train" / path.name)
        elif game_play in val_game_plays:
            shutil.copy2(path, validation_dir / path.name)
    shutil.rmtree(backup)

    return {
        "authoritative_train_sources": [str(visible_output / "train_labels.csv"), str(visible_output / "train")],
        "authoritative_validation_sources": ["validation/validation.csv"],
        "hidden_answer_granularity": "direct",
        "hidden_alignment_mode": "group_count",
    }


def _prepare_kuzushiji_second_split(
    visible_output,
    validation_dir,
    hidden_answers_path,
    hidden_sample_submission_path,
    visible_sample_submission_path,
):
    import pandas as pd

    train_df = pd.read_csv(visible_output / "train.csv")
    keep_df, val_df = _split_df_basic(train_df, "kuzushiji-recognition", visible_output)
    keep_df.to_csv(visible_output / "train.csv", index=False)
    val_df[["image_id"]].to_csv(validation_dir / "validation.csv", index=False)
    val_df.to_csv(hidden_answers_path, index=False)
    sample_df = val_df[["image_id"]].copy()
    sample_df["labels"] = "U+003F 1 1 U+FF2F 2 2"
    sample_df.to_csv(hidden_sample_submission_path, index=False)
    sample_df.to_csv(visible_sample_submission_path, index=False)
    _split_asset_dir_with_ids(
        visible_output / "train_images",
        visible_output / "train_images",
        validation_dir / "validation_images",
        [str(v) for v in keep_df["image_id"].tolist()],
        [str(v) for v in val_df["image_id"].tolist()],
        [".jpg"],
    )
    return {
        "authoritative_train_sources": [str(visible_output / "train.csv"), str(visible_output / "train_images")],
        "authoritative_validation_sources": ["validation/validation.csv"],
        "hidden_answer_granularity": "direct",
        "hidden_alignment_mode": "id_count",
    }


def _prepare_hubmap_second_split(
    visible_output,
    validation_dir,
    hidden_answers_path,
    hidden_sample_submission_path,
    visible_sample_submission_path,
):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    train_df = pd.read_csv(visible_output / "train.csv")
    dataset_info = pd.read_csv(visible_output / "HuBMAP-20-dataset_information.csv")
    dataset_info["id"] = dataset_info["image_file"].str.replace(".tiff", "", regex=False)
    keep_df, val_df = train_test_split(train_df, test_size=max(1, min(3, len(train_df) - 1)), random_state=0)
    val_with_dims = val_df.merge(dataset_info[["id", "width_pixels", "height_pixels"]], on="id")

    keep_df.to_csv(visible_output / "train.csv", index=False)
    val_df[["id"]].to_csv(validation_dir / "validation.csv", index=False)
    pd.DataFrame({"id": val_with_dims["id"], "predicted": val_with_dims["encoding"]}).to_csv(
        hidden_answers_path, index=False
    )
    sample_df = pd.DataFrame({"id": val_with_dims["id"], "predicted": ""})
    sample_df.to_csv(hidden_sample_submission_path, index=False)
    sample_df.to_csv(visible_sample_submission_path, index=False)

    keep_ids = [str(v) for v in keep_df["id"].tolist()]
    val_ids = [str(v) for v in val_df["id"].tolist()]
    for suffixes in [[".tiff"], [".json"], ["-anatomical-structure.json"]]:
        _split_asset_dir_with_ids(
            visible_output / "train",
            visible_output / "train",
            validation_dir / "validation",
            keep_ids,
            val_ids,
            suffixes,
        )
    return {
        "authoritative_train_sources": [str(visible_output / "train.csv"), str(visible_output / "train")],
        "authoritative_validation_sources": ["validation/validation.csv"],
        "hidden_answer_granularity": "direct",
        "hidden_alignment_mode": "group_count",
    }


def _prepare_uw_second_split(
    visible_output,
    validation_dir,
    hidden_answers_path,
    hidden_sample_submission_path,
    visible_sample_submission_path,
):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    train_df = pd.read_csv(visible_output / "train.csv")
    train_df["case"] = train_df["id"].apply(lambda x: x.split("_")[0])
    train_df["day"] = train_df["id"].apply(lambda x: x.split("_")[1])
    train_df["slice"] = train_df["id"].apply(lambda x: x.split("_")[-1])
    unique_cases = train_df["case"].unique()
    keep_cases, val_cases = train_test_split(
        unique_cases,
        test_size=_configured_split_fraction(),
        random_state=42,
    )
    train_df["set"] = train_df["case"].apply(lambda x: "validation" if x in set(val_cases) else "train")
    days_df = train_df[train_df["set"] == "train"].groupby("case")["day"].apply(set).reset_index()
    for _, row in days_df.iterrows():
        days = row["day"]
        if len(days) > 4:
            days = sorted(days, key=lambda x: int(x[len("day"):]))
            days_to_move = days[4:]
            train_df.loc[train_df["case"].eq(row["case"]) & train_df["day"].isin(days_to_move), "set"] = "validation"

    keep_df = train_df[train_df["set"] == "train"].copy()
    val_df = train_df[train_df["set"] == "validation"].copy()

    backup = visible_output / ".train_hidden_validation_source"
    if backup.exists():
        shutil.rmtree(backup)
    (visible_output / "train").rename(backup)
    (visible_output / "train").mkdir(parents=True, exist_ok=True)
    for case in unique_cases:
        source = backup / case
        if not source.exists():
            continue
        target = (validation_dir if case in set(val_cases) else visible_output / "train") / case
        shutil.copytree(source, target, dirs_exist_ok=True)
    for _, row in val_df.iterrows():
        source = visible_output / "train" / row["case"] / f"{row['case']}_{row['day']}"
        target = validation_dir / row["case"] / f"{row['case']}_{row['day']}"
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(source.as_posix(), target.as_posix())
    shutil.rmtree(backup)

    for _, row in val_df.iterrows():
        image_paths = list((validation_dir / row["case"] / f"{row['case']}_{row['day']}" / "scans").glob(f"slice_{row['slice']}_*.png"))
        if len(image_paths) == 1:
            width, height = (int(length) for length in image_paths[0].stem.split("_")[2:4])
            val_df.loc[row.name, "image_width"] = width
            val_df.loc[row.name, "image_height"] = height
    keep_df.drop(columns=["set", "case", "day", "slice"], inplace=True)
    val_df.drop(columns=["set", "case", "day", "slice"], inplace=True)
    keep_df.to_csv(visible_output / "train.csv", index=False, na_rep="")
    val_visible = val_df.drop(columns=["segmentation", "image_width", "image_height"]).copy()
    val_visible.to_csv(validation_dir / "validation.csv", index=False, na_rep="")
    val_private = val_df.rename(columns={"segmentation": "predicted"})
    val_private.to_csv(hidden_answers_path, index=False, na_rep="")
    sample_df = val_private.drop(columns=["image_width", "image_height"]).copy()
    sample_df["predicted"] = "1 1 5 2"
    sample_df.to_csv(hidden_sample_submission_path, index=False, na_rep="")
    sample_df.to_csv(visible_sample_submission_path, index=False, na_rep="")
    return {
        "authoritative_train_sources": [str(visible_output / "train.csv"), str(visible_output / "train")],
        "authoritative_validation_sources": ["validation/validation.csv"],
        "hidden_answer_granularity": "direct",
    }


def _prepare_multimodal_second_split(
    visible_output,
    validation_dir,
    hidden_answers_path,
    hidden_sample_submission_path,
    visible_sample_submission_path,
):
    import pandas as pd
    import tarfile

    archive_candidates = sorted(visible_output.glob("training*.tar.gz"))
    if not archive_candidates:
        raise RuntimeError("multi-modal adapter expected at least one training*.tar.gz archive")
    archive_path = archive_candidates[-1]
    archive_name = archive_path.name

    with tarfile.open(archive_path, "r:gz") as tar:
        member_ids = sorted(
            {
                Path(member.name).stem[-4:]
                for member in tar.getmembers()
                if member.isfile() and member.name.endswith(".zip")
            }
        )
    if not member_ids:
        raise RuntimeError("multi-modal adapter could not derive validation ids from training3.tar.gz")

    training_df = pd.read_csv(visible_output / "training.csv", dtype={"Id": str, "Sequence": str})
    keep_df = training_df[~training_df["Id"].isin(member_ids)].copy()
    val_df = training_df[training_df["Id"].isin(member_ids)].copy()
    keep_df.to_csv(visible_output / "training.csv", index=False)
    pd.DataFrame({"Id": member_ids}).to_csv(validation_dir / "validation.csv", index=False)
    val_df.to_csv(hidden_answers_path, index=False)
    sample_df = pd.DataFrame({"Id": member_ids, "Sequence": [" ".join(str(x) for x in range(1, 21))] * len(member_ids)})
    sample_df.to_csv(hidden_sample_submission_path, index=False)
    sample_df.to_csv(visible_sample_submission_path, index=False)
    archive_path.rename(visible_output / f"holdout_{archive_name}")
    shutil.copy2(visible_output / f"holdout_{archive_name}", validation_dir / "validation.tar.gz")
    return {
        "authoritative_train_sources": [str(visible_output / "training.csv")],
        "authoritative_validation_sources": ["validation/validation.csv"],
        "hidden_answer_granularity": "direct",
        "hidden_alignment_mode": "asset_presence_only",
    }


def _prepare_billion_word_second_split(
    visible_output,
    validation_dir,
    hidden_answers_path,
    hidden_sample_submission_path,
    visible_sample_submission_path,
):
    import numpy as np
    import pandas as pd

    rng = np.random.RandomState(0)
    train_path = visible_output / "train_v2.txt"
    if not train_path.exists():
        raise RuntimeError("billion-word adapter expected extracted train_v2.txt")

    kept_lines: list[str] = []
    visible_rows: list[dict[str, Any]] = []
    answer_rows: list[dict[str, Any]] = []
    submission_lines: list[str] = []
    with train_path.open() as f:
        for sentence in f:
            stripped = sentence.strip()
            words = stripped.split()
            if len(words) > 2 and rng.uniform() <= 0.01:
                idx = rng.randint(1, len(words) - 1)
                removed = " ".join(words[:idx] + words[idx + 1 :])
                row_id = len(answer_rows)
                visible_rows.append({"id": row_id, "sentence": removed})
                answer_rows.append({"id": row_id, "sentence": stripped})
                submission_lines.append(removed)
            else:
                kept_lines.append(sentence)

    train_path.write_text("".join(kept_lines))
    pd.DataFrame(visible_rows).to_csv(validation_dir / "validation.csv", index=False)
    pd.DataFrame(answer_rows).to_csv(hidden_answers_path, index=False)
    validation_txt = validation_dir / "validation_v2.txt"
    validation_txt.write_text("\n".join(submission_lines) + ("\n" if submission_lines else ""))
    sample_submission_rows = visible_rows
    if not answer_rows:
        fallback_sentence = next(
            (line.strip() for line in kept_lines if len(line.strip().split()) > 2),
            None,
        )
        if fallback_sentence is None:
            raise RuntimeError("billion-word adapter could not derive any hidden validation examples")
        fallback_line = next(
            (line for line in kept_lines if line.strip() == fallback_sentence),
            None,
        )
        if fallback_line is not None:
            kept_lines.remove(fallback_line)
        words = fallback_sentence.split()
        idx = max(1, len(words) // 2)
        removed = " ".join(words[:idx] + words[idx + 1 :])
        pd.DataFrame([{"id": 0, "sentence": removed}]).to_csv(validation_dir / "validation.csv", index=False)
        pd.DataFrame([{"id": 0, "sentence": fallback_sentence}]).to_csv(hidden_answers_path, index=False)
        validation_txt.write_text(removed + "\n")
        train_path.write_text("".join(kept_lines))
        sample_submission_rows = [{"id": 0, "sentence": removed}]
    pd.DataFrame(sample_submission_rows).to_csv(hidden_sample_submission_path, index=False)
    pd.DataFrame(sample_submission_rows).to_csv(visible_sample_submission_path, index=False)
    return {
        "authoritative_train_sources": [str(visible_output / "train_v2.txt")],
        "authoritative_validation_sources": ["validation/validation.csv"],
        "hidden_answer_granularity": "direct",
        "hidden_alignment_mode": "group_count",
    }


def _prepare_freesound_second_split(
    visible_output,
    validation_dir,
    hidden_answers_path,
    hidden_sample_submission_path,
    visible_sample_submission_path,
):
    import pandas as pd

    train_csv = visible_output / "train_curated.csv"
    if not train_csv.exists():
        raise RuntimeError("freesound adapter expected train_curated.csv")

    train_df = pd.read_csv(train_csv)
    keep_df, val_df = _split_df_basic(train_df, "freesound-audio-tagging-2019", visible_output)
    keep_df.to_csv(train_csv, index=False)
    val_df[["fname"]].to_csv(validation_dir / "validation.csv", index=False)

    sample_template_df, _ = _load_sample_submission_template(visible_output, "freesound-audio-tagging-2019")
    class_names = [col for col in sample_template_df.columns if col != "fname"]
    answer_rows = []
    for _, row in val_df.iterrows():
        labels = {label.strip() for label in str(row["labels"]).split(",") if label.strip()}
        answer_rows.append({"fname": row["fname"], **{name: int(name in labels) for name in class_names}})
    answers_df = pd.DataFrame(answer_rows)
    answers_df.to_csv(hidden_answers_path, index=False)
    sample_df = answers_df.copy()
    sample_df.loc[:, class_names] = 0.0
    sample_df.to_csv(hidden_sample_submission_path, index=False)
    sample_df.to_csv(visible_sample_submission_path, index=False)

    _split_asset_dir_with_ids(
        visible_output / "train_curated",
        visible_output / "train_curated",
        validation_dir / "validation_audio",
        [str(v) for v in keep_df["fname"].tolist()],
        [str(v) for v in val_df["fname"].tolist()],
        [""],
    )
    return {
        "authoritative_train_sources": [str(visible_output / "train_curated.csv"), str(visible_output / "train_curated"), str(visible_output / "train_noisy.csv"), str(visible_output / "train_noisy")],
        "authoritative_validation_sources": ["validation/validation.csv"],
        "hidden_answer_granularity": "direct",
        "hidden_alignment_mode": "id_count",
    }


def _prepare_tensorflow_qa_second_split(
    visible_output,
    validation_dir,
    hidden_answers_path,
    hidden_sample_submission_path,
    visible_sample_submission_path,
):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    train_path = visible_output / "simplified-nq-train.jsonl"
    train_df = pd.read_json(train_path, orient="records", lines=True)
    train_df["example_id"] = train_df["example_id"].astype(str)
    keep_df, val_df = train_test_split(
        train_df,
        test_size=_configured_split_fraction(),
        random_state=0,
    )

    keep_df.to_json(train_path, orient="records", lines=True)
    keys_to_keep = ["example_id", "question_text", "document_text", "long_answer_candidates"]
    val_df[keys_to_keep].to_json(validation_dir / "validation.jsonl", orient="records", lines=True)

    gold_rows = []
    for _, sample in val_df[["example_id", "annotations"]].iterrows():
        annotation = sample["annotations"][0]
        if annotation["yes_no_answer"] != "NONE":
            short_answer = annotation["yes_no_answer"]
        elif len(annotation["short_answers"]) > 0:
            first_short = annotation["short_answers"][0]
            short_answer = f"{first_short['start_token']}:{first_short['end_token']}"
        else:
            short_answer = ""
        long = annotation["long_answer"]
        long_answer = f"{long['start_token']}:{long['end_token']}" if long["start_token"] != -1 else ""
        gold_rows.append({"example_id": f"{sample['example_id']}_short", "PredictionString": short_answer})
        gold_rows.append({"example_id": f"{sample['example_id']}_long", "PredictionString": long_answer})

    answers_df = pd.DataFrame(gold_rows)
    answers_df.to_csv(hidden_answers_path, index=False)
    sample_df = answers_df.copy()
    sample_df["PredictionString"] = ""
    sample_df.to_csv(hidden_sample_submission_path, index=False)
    sample_df.to_csv(visible_sample_submission_path, index=False)
    pd.DataFrame({"example_id": val_df["example_id"]}).to_csv(validation_dir / "validation.csv", index=False)
    return {
        "authoritative_train_sources": [str(train_path)],
        "authoritative_validation_sources": ["validation/validation.csv", "validation/validation.jsonl"],
        "hidden_answer_granularity": "direct",
        "hidden_alignment_mode": "group_count",
    }


def _prepare_denoising_second_split(visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path):
    import numpy as np
    import pandas as pd
    from PIL import Image
    from sklearn.model_selection import train_test_split

    train_ids = sorted(path.stem for path in (visible_output / "train").glob("*.png"))
    keep_ids, val_ids = train_test_split(
        train_ids,
        test_size=_configured_split_fraction(),
        random_state=0,
    )
    train_cleaned_backup = visible_output / ".train_cleaned_hidden_validation_answers_source"
    if train_cleaned_backup.exists():
        shutil.rmtree(train_cleaned_backup)
    shutil.copytree(visible_output / "train_cleaned", train_cleaned_backup)
    _split_asset_dir_with_ids(visible_output / "train", visible_output / "train", validation_dir, keep_ids, val_ids, [".png"])
    _split_asset_dir_with_ids(visible_output / "train_cleaned", visible_output / "train_cleaned", validation_dir / "_unused", keep_ids, [], [".png"])
    if (validation_dir / "_unused").exists():
        shutil.rmtree(validation_dir / "_unused")

    def to_df(img: Path):
        image = Image.open(img).convert("L")
        image_array = np.array(image) / 255.0
        rows, cols = image_array.shape
        data = {"id": [], "value": []}
        for row in range(rows):
            for col in range(cols):
                data["id"].append(f"{img.stem}_{row+1}_{col+1}")
                data["value"].append(float(image_array[row, col]))
        return pd.DataFrame(data)

    answers_df = pd.concat([to_df(train_cleaned_backup / f"{item_id}.png") for item_id in val_ids], ignore_index=True)
    answers_df.to_csv(hidden_answers_path, index=False)
    sample_df = answers_df.copy()
    sample_df["value"] = 1
    sample_df.to_csv(hidden_sample_submission_path, index=False)
    sample_df.to_csv(visible_sample_submission_path, index=False)
    if train_cleaned_backup.exists():
        shutil.rmtree(train_cleaned_backup)
    return {
        "authoritative_train_sources": [str(visible_output / "train"), str(visible_output / "train_cleaned")],
        "authoritative_validation_sources": ["validation"],
        "hidden_answer_granularity": "derived",
        "hidden_alignment_mode": "asset_presence_only",
    }


def _prepare_dogs_vs_cats_second_split(visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    train_files = sorted((visible_output / "train").glob("*.jpg"))
    keep_files, val_files = train_test_split(
        train_files,
        test_size=_configured_split_fraction(),
        random_state=0,
    )
    backup = visible_output / ".train_hidden_validation_source"
    if backup.exists():
        shutil.rmtree(backup)
    (visible_output / "train").rename(backup)
    (visible_output / "train").mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    for src in keep_files:
        shutil.copy2(backup / src.name, visible_output / "train" / src.name)
    answers_rows = []
    for idx, src in enumerate(sorted(val_files), start=1):
        dst_name = f"{idx}.jpg"
        shutil.copy2(backup / src.name, validation_dir / dst_name)
        answers_rows.append({"id": idx, "label": int("dog" in src.name)})
    shutil.rmtree(backup)
    answers_df = pd.DataFrame(answers_rows)
    answers_df.to_csv(hidden_answers_path, index=False)
    sample_df = pd.DataFrame({"id": answers_df["id"], "label": 0.5})
    sample_df.to_csv(hidden_sample_submission_path, index=False)
    sample_df.to_csv(visible_sample_submission_path, index=False)
    pd.DataFrame({"id": answers_df["id"]}).to_csv(validation_dir / "validation.csv", index=False)
    return {
        "authoritative_train_sources": [str(visible_output / "train")],
        "authoritative_validation_sources": ["validation", "validation/validation.csv"],
        "hidden_answer_granularity": "direct",
        "hidden_alignment_mode": "id_count",
    }


def _prepare_random_pizza_second_split(visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    with (visible_output / "train.json").open() as f:
        train_samples = json.load(f)
    with (visible_output / "test.json").open() as f:
        current_test_samples = json.load(f)
    stratify_labels = [int(sample["requester_received_pizza"]) for sample in train_samples]
    keep_samples, val_samples = train_test_split(
        train_samples,
        test_size=0.2,
        random_state=0,
        stratify=stratify_labels,
    )
    test_fields = list(current_test_samples[0].keys()) if current_test_samples else [k for k in keep_samples[0].keys() if k != "requester_received_pizza"]
    with (visible_output / "train.json").open("w") as f:
        json.dump(keep_samples, f, indent=4)
    validation_json = [{key: sample[key] for key in test_fields} for sample in val_samples]
    with (validation_dir / "validation.json").open("w") as f:
        json.dump(validation_json, f, indent=4)
    pd.DataFrame(validation_json).to_csv(validation_dir / "validation.csv", index=False)
    answers_df = pd.DataFrame([{"request_id": sample["request_id"], "requester_received_pizza": int(sample["requester_received_pizza"])} for sample in val_samples])
    answers_df.to_csv(hidden_answers_path, index=False)
    sample_df = answers_df.copy()
    sample_df["requester_received_pizza"] = 0
    sample_df.to_csv(hidden_sample_submission_path, index=False)
    sample_df.to_csv(visible_sample_submission_path, index=False)
    manifest_path = _write_train_count_manifest(visible_output, [{"request_id": sample["request_id"]} for sample in keep_samples])
    return {
        "authoritative_train_sources": [manifest_path, str(visible_output / "train.json")],
        "authoritative_validation_sources": ["validation/validation.json", "validation/validation.csv"],
        "hidden_answer_granularity": "direct",
        "hidden_alignment_mode": "id_count",
    }


def _prepare_detecting_insults_second_split(
    visible_output,
    validation_dir,
    hidden_answers_path,
    hidden_sample_submission_path,
    visible_sample_submission_path,
):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    train_df = pd.read_csv(visible_output / "train.csv")
    keep_df, val_df = train_test_split(
        train_df,
        test_size=_configured_split_fraction(),
        random_state=0,
        stratify=train_df["Insult"],
    )

    keep_df.to_csv(visible_output / "train.csv", index=False)

    validation_visible_df = val_df[["Date", "Comment"]].copy()
    validation_visible_df.to_csv(validation_dir / "validation.csv", index=False)

    answers_df = val_df[["Insult", "Date", "Comment"]].copy()
    answers_df.to_csv(hidden_answers_path, index=False)

    sample_df = answers_df.copy()
    sample_df["Insult"] = 0
    sample_df = sample_df[["Insult", "Date", "Comment"]]
    sample_df.to_csv(hidden_sample_submission_path, index=False)
    sample_df.to_csv(visible_sample_submission_path, index=False)

    return {
        "authoritative_train_sources": [str(visible_output / "train.csv")],
        "authoritative_validation_sources": ["validation/validation.csv"],
        "hidden_answer_granularity": "direct",
        "hidden_alignment_mode": "submission_rows",
    }


def _prepare_text_normalization_second_split(competition_id, visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    prefix = "en" if "english" in competition_id else "ru"
    train_name = f"{prefix}_train.csv"
    test_name = f"{prefix}_test_2.csv"
    train_df = pd.read_csv(visible_output / train_name)
    sentence_ids = train_df["sentence_id"].unique()
    keep_sentence_ids, val_sentence_ids = train_test_split(
        sentence_ids,
        test_size=_configured_split_fraction(),
        random_state=0,
    )
    keep_df = train_df[train_df["sentence_id"].isin(keep_sentence_ids)].copy()
    val_df = train_df[train_df["sentence_id"].isin(val_sentence_ids)].copy()
    keep_df.to_csv(visible_output / train_name, index=False)
    validation_visible_df = val_df[["sentence_id", "token_id", "before"]].copy()
    validation_visible_df.to_csv(validation_dir / test_name, index=False)
    answers_df = val_df[["sentence_id", "token_id", "after"]].copy()
    answers_df["id"] = answers_df["sentence_id"].astype(str) + "_" + answers_df["token_id"].astype(str)
    answers_df = answers_df[["id", "after"]]
    answers_df.to_csv(hidden_answers_path, index=False)
    sample_df = validation_visible_df.copy()
    sample_df["id"] = sample_df["sentence_id"].astype(str) + "_" + sample_df["token_id"].astype(str)
    sample_df["after"] = sample_df["before"]
    sample_df = sample_df[["id", "after"]]
    sample_df.to_csv(hidden_sample_submission_path, index=False)
    sample_df.to_csv(visible_sample_submission_path, index=False)
    return {
        "authoritative_train_sources": [str(visible_output / train_name)],
        "authoritative_validation_sources": [f"validation/{test_name}"],
        "hidden_answer_granularity": "direct",
        "hidden_alignment_mode": "row_count",
    }


def _prepare_whale_second_split(visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path):
    import pandas as pd

    train_files = sorted((visible_output / "train2").glob("*.aif"))
    samples_by_date: dict[str, list[Path]] = {}
    for path in train_files:
        date_key = path.name.split("_", 1)[0]
        samples_by_date.setdefault(date_key, []).append(path)
    dates = sorted(samples_by_date)
    cutoff = max(1, int(len(dates) * 0.8))
    keep_files = sorted([path for date in dates[:cutoff] for path in samples_by_date[date]])
    val_files = sorted([path for date in dates[cutoff:] for path in samples_by_date[date]])
    if not val_files:
        val_files = keep_files[-max(1, len(keep_files) // 2):]
        keep_files = keep_files[:-len(val_files)]
    backup = visible_output / ".train2_hidden_validation_source"
    if backup.exists():
        shutil.rmtree(backup)
    (visible_output / "train2").rename(backup)
    (visible_output / "train2").mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    for src in keep_files:
        shutil.copy2(backup / src.name, visible_output / "train2" / src.name)
    answers_rows = []
    visible_rows = []
    for idx, src in enumerate(val_files):
        dst_name = src.name.split("TRAIN")[0] + f"Test{idx}.aif"
        shutil.copy2(backup / src.name, validation_dir / dst_name)
        answers_rows.append({"clip": dst_name, "probability": 1 if src.stem.endswith("_1") else 0})
        visible_rows.append({"clip": dst_name})
    shutil.rmtree(backup)
    pd.DataFrame(visible_rows).to_csv(validation_dir / "validation.csv", index=False)
    answers_df = pd.DataFrame(answers_rows)
    answers_df.to_csv(hidden_answers_path, index=False)
    sample_df = answers_df.copy()
    sample_df["probability"] = 0
    sample_df.to_csv(hidden_sample_submission_path, index=False)
    sample_df.to_csv(visible_sample_submission_path, index=False)
    return {
        "authoritative_train_sources": [str(visible_output / "train2")],
        "authoritative_validation_sources": ["validation", "validation/validation.csv"],
        "hidden_answer_granularity": "direct",
        "hidden_alignment_mode": "id_count",
    }


def _prepare_mlsp_second_split(visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    cv_df = pd.read_csv(visible_output / "essential_data/CVfolds_2.txt")
    existing_test_ids = set(cv_df[cv_df["fold"] == 1]["rec_id"].tolist())
    train_ids = cv_df[cv_df["fold"] == 0]["rec_id"].tolist()
    keep_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=0)
    keep_id_set = set(keep_ids)
    val_id_set = set(val_ids)

    visible_cv = cv_df[cv_df["rec_id"].isin(keep_id_set | existing_test_ids)].copy()
    visible_cv.loc[visible_cv["rec_id"].isin(keep_id_set), "fold"] = 0
    visible_cv.to_csv(visible_output / "essential_data/CVfolds_2.txt", index=False)

    rec_map_df = pd.read_csv(visible_output / "essential_data/rec_id2filename.txt")
    rec_map_df[rec_map_df["rec_id"].isin(keep_id_set | existing_test_ids)].to_csv(
        visible_output / "essential_data/rec_id2filename.txt", index=False
    )
    val_rec_map_df = rec_map_df[rec_map_df["rec_id"].isin(val_id_set)].copy()
    (validation_dir / "essential_data").mkdir(parents=True, exist_ok=True)
    (validation_dir / "supplemental_data").mkdir(parents=True, exist_ok=True)
    val_rec_map_df.to_csv(validation_dir / "essential_data/rec_id2filename.txt", index=False)
    pd.DataFrame({"rec_id": sorted(val_ids)}).to_csv(validation_dir / "validation_rec_ids.csv", index=False)

    labels_df = _load_mlsp_labels(visible_output / "essential_data/rec_labels_test_hidden.txt")
    visible_labels = labels_df[labels_df["rec_id"].isin(keep_id_set | existing_test_ids)].copy()
    visible_labels.loc[visible_labels["rec_id"].isin(existing_test_ids), "[labels]"] = "?"
    _write_mlsp_labels(visible_output / "essential_data/rec_labels_test_hidden.txt", visible_labels)
    val_labels_df = labels_df[labels_df["rec_id"].isin(val_id_set)].copy()
    _write_mlsp_labels(validation_dir / "essential_data/rec_labels_test_hidden.txt", val_labels_df)
    if (visible_output / "essential_data/species_list.txt").exists():
        shutil.copy2(visible_output / "essential_data/species_list.txt", validation_dir / "essential_data/species_list.txt")

    for filename in val_rec_map_df["filename"].tolist():
        wav = visible_output / "essential_data/src_wavs" / f"{filename}.wav"
        if wav.exists():
            _copy_path(wav, validation_dir / "essential_data/src_wavs" / wav.name)
        for dir_name in ["filtered_spectrograms", "segmentation_examples", "spectrograms", "supervised_segmentation"]:
            bmp = visible_output / "supplemental_data" / dir_name / f"{filename}.bmp"
            if bmp.exists():
                _copy_path(bmp, validation_dir / "supplemental_data" / dir_name / bmp.name)

    for plain_name in ["segment_clusters.bmp", "segment_mosaic.bmp"]:
        src = visible_output / "supplemental_data" / plain_name
        if src.exists():
            _copy_path(src, validation_dir / "supplemental_data" / plain_name)

    for text_name in ["histogram_of_segments.txt", "segment_features.txt", "segment_rectangles.txt"]:
        src = visible_output / "supplemental_data" / text_name
        if src.exists():
            _filter_mlsp_index_file(src, validation_dir / "supplemental_data" / text_name, val_id_set)
            _filter_mlsp_index_file(src, visible_output / "supplemental_data" / text_name, keep_id_set | existing_test_ids)

    answers_rows = []
    for _, row in val_labels_df.iterrows():
        rec_id = int(row["rec_id"])
        labels = str(row["[labels]"]).strip()
        species_ids = [int(item) for item in labels.split(",") if item not in {"", "?"}]
        for species_id in range(19):
            answers_rows.append({"Id": rec_id * 100 + species_id, "Probability": int(species_id in species_ids)})
    answers_df = pd.DataFrame(answers_rows)
    answers_df.to_csv(hidden_answers_path, index=False)
    sample_df = answers_df.copy()
    sample_df["Probability"] = 0
    sample_df.to_csv(hidden_sample_submission_path, index=False)
    sample_df.to_csv(visible_sample_submission_path, index=False)
    manifest_path = _write_train_count_manifest(visible_output, [{"rec_id": rid} for rid in keep_ids])
    return {
        "authoritative_train_sources": [manifest_path],
        "authoritative_validation_sources": ["validation/essential_data", "validation/supplemental_data"],
        "hidden_answer_granularity": "derived",
        "hidden_alignment_mode": "group_count",
    }


def _prepare_second_split_hidden_validation_artifacts(
    competition_id: str,
    visible_output: Path,
    validation_dir: Path,
    hidden_answers_path: Path,
    hidden_sample_submission_path: Path,
    visible_sample_submission_path: Path,
) -> dict[str, Any]:
    if competition_id == "denoising-dirty-documents":
        return _prepare_denoising_second_split(visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path)
    if competition_id == "multi-modal-gesture-recognition":
        return _prepare_multimodal_second_split(visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path)
    if competition_id == "billion-word-imputation":
        return _prepare_billion_word_second_split(visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path)
    if competition_id == "whale-categorization-playground":
        return _prepare_whale_categorization_second_split(visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path)
    if competition_id == "dogs-vs-cats-redux-kernels-edition":
        return _prepare_dogs_vs_cats_second_split(visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path)
    if competition_id == "freesound-audio-tagging-2019":
        return _prepare_freesound_second_split(visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path)
    if competition_id == "random-acts-of-pizza":
        return _prepare_random_pizza_second_split(visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path)
    if competition_id == "detecting-insults-in-social-commentary":
        return _prepare_detecting_insults_second_split(visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path)
    if competition_id == "ventilator-pressure-prediction":
        return _prepare_ventilator_second_split(visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path)
    if competition_id == "champs-scalar-coupling":
        return _prepare_champs_second_split(visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path)
    if competition_id == "osic-pulmonary-fibrosis-progression":
        return _prepare_osic_second_split(visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path)
    if competition_id == "hotel-id-2021-fgvc8":
        return _prepare_hotel_second_split(visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path)
    if competition_id == "stanford-covid-vaccine":
        return _prepare_stanford_second_split(visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path)
    if competition_id == "tensorflow2-question-answering":
        return _prepare_tensorflow_qa_second_split(visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path)
    if competition_id == "bms-molecular-translation":
        return _prepare_bms_second_split(visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path)
    if competition_id == "h-and-m-personalized-fashion-recommendations":
        return _prepare_h_and_m_second_split(visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path)
    if competition_id == "cassava-leaf-disease-classification":
        return _prepare_cassava_second_split(visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path)
    if competition_id == "hms-harmful-brain-activity-classification":
        return _prepare_hms_second_split(visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path)
    if competition_id == "smartphone-decimeter-2022":
        return _prepare_smartphone_second_split(visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path)
    if competition_id == "nfl-player-contact-detection":
        return _prepare_nfl_second_split(visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path)
    if competition_id == "kuzushiji-recognition":
        return _prepare_kuzushiji_second_split(visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path)
    if competition_id == "hubmap-kidney-segmentation":
        return _prepare_hubmap_second_split(visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path)
    if competition_id == "uw-madison-gi-tract-image-segmentation":
        return _prepare_uw_second_split(visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path)
    if competition_id in {"text-normalization-challenge-english-language", "text-normalization-challenge-russian-language"}:
        return _prepare_text_normalization_second_split(competition_id, visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path)
    if competition_id == "the-icml-2013-whale-challenge-right-whale-redux":
        return _prepare_whale_second_split(visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path)
    if competition_id == "mlsp-2013-birds":
        return _prepare_mlsp_second_split(visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path)
    return _prepare_generic_dataframe_second_split(competition_id, visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path)


def maybe_prepare_mlebench_lite_hidden_validation(cfg) -> dict[str, Any] | None:
    global DEFAULT_SPLIT_FRACTION

    competition_id = str(getattr(getattr(cfg, "hidden_validation", None), "competition_name", "")).strip() or infer_competition_id(cfg)
    if competition_id not in supported_deterministic_competition_ids():
        return None
    DEFAULT_SPLIT_FRACTION = float(
        getattr(getattr(cfg, "hidden_validation", None), "split_fraction", 0.10)
    )

    try:
        from mlebench.registry import registry

        registry.get_competition(competition_id)
    except Exception:
        logger.exception(
            "Failed to initialize mle-bench lite hidden validation adapter for %s",
            competition_id,
        )
        raise RuntimeError(
            f"Failed to initialize deterministic mle-bench lite hidden validation adapter for {competition_id}"
        )

    state = default_state()
    state["enabled"] = True

    attempt_dir = Path(cfg.workspace_dir).parent / ".hidden_validation" / "mlebench_lite_adapter"
    visible_output = attempt_dir / "input"
    manifest_path = attempt_dir / "split_manifest.json"
    evidence_dir = attempt_dir / "evidence"
    hidden_validation_dir = visible_output / "hidden_validation"
    hidden_answers_path = attempt_dir / "hidden_validation_answers.csv"
    hidden_sample_submission_path = attempt_dir / "sampleHiddenValidationSubmission.csv"

    if attempt_dir.exists():
        shutil.rmtree(attempt_dir)
    attempt_dir.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copytree(Path(cfg.data_dir), visible_output)
        preproc_data(visible_output)

        hidden_validation_dir.mkdir(parents=True, exist_ok=True)
        hidden_prep_result = _prepare_second_split_hidden_validation_artifacts(
            competition_id,
            visible_output,
            hidden_validation_dir,
            hidden_answers_path,
            hidden_sample_submission_path,
            hidden_sample_submission_path,
        )
        authoritative_train_sources = hidden_prep_result["authoritative_train_sources"]
        authoritative_hidden_validation_sources = _rewrite_validation_sources(
            hidden_prep_result["authoritative_validation_sources"],
            hidden_validation_dir,
        )
        hidden_answer_granularity = hidden_prep_result["hidden_answer_granularity"]
        hidden_alignment_mode = hidden_prep_result.get("hidden_alignment_mode", "submission_rows")
        authoritative_hidden_answer_sources = [str(hidden_answers_path)]
        strategy_summary = (
            "Deterministic hidden-validation adapter for supported MLE-bench lite and MLEBench-30 tasks. "
            "Visible input is copied from cfg.data_dir, then one deterministic hidden holdout is performed "
            "over the prepared training data only. The holdout is exposed under hidden_validation/ with labels "
            "kept outside the agent workspace. Runtime search continues to use self-reported metrics, while "
            "hidden validation is used only for final selection among valid submissions."
        )

        _write_adapter_evidence(
            cfg=cfg,
            visible_output=visible_output,
            hidden_output=attempt_dir,
            evidence_dir=evidence_dir,
            authoritative_train_sources=authoritative_train_sources,
            authoritative_validation_sources=authoritative_hidden_validation_sources,
            authoritative_hidden_answer_sources=authoritative_hidden_answer_sources,
            strategy_summary=strategy_summary,
        )

        hidden_count = len(read_csv_rows(hidden_answers_path))
        visible_count = _estimate_visible_count(visible_output, authoritative_train_sources)
        manifest = {
            "visible_input_dir": str(visible_output),
            "hidden_validation_dir": str(hidden_validation_dir),
            "hidden_answers_path": str(hidden_answers_path),
            "hidden_sample_submission_path": str(hidden_sample_submission_path),
            "competition_id": competition_id,
            "split_seed": "mlebench_prepare",
            "strategy_summary": strategy_summary,
            "visible_count": visible_count,
            "hidden_count": hidden_count,
            "contract_version": "mlebench_lite_adapter_v3",
            "evidence_dir": str(evidence_dir),
            "authoritative_train_sources": authoritative_train_sources,
            "authoritative_hidden_validation_sources": authoritative_hidden_validation_sources,
            "authoritative_hidden_answer_sources": authoritative_hidden_answer_sources,
            "hidden_answer_granularity": hidden_answer_granularity,
            "hidden_alignment_mode": hidden_alignment_mode,
            "split_fraction": _configured_split_fraction(),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))

        verify_result = verify_split_artifacts(cfg, manifest_path)
        if not verify_result["ok"]:
            raise RuntimeError(verify_result["reason"])

        state.update(
            {
                "active": True,
                "fallback_mode": False,
                "fallback_reason": "",
                "visible_input_dir": manifest["visible_input_dir"],
                "hidden_validation_dir": manifest["hidden_validation_dir"],
                "hidden_answers_path": manifest["hidden_answers_path"],
                "hidden_sample_submission_path": manifest["hidden_sample_submission_path"],
                "manifest_path": str(manifest_path),
                "splitter_attempts": 0,
                "reviewer_status": "mlebench_lite_adapter",
            }
        )
        logger.info(
            "Prepared deterministic hidden validation adapter for %s",
            competition_id,
        )
        return save_runtime_state(cfg, state)
    except Exception as exc:
        logger.exception(
            "Deterministic hidden validation adapter failed for %s",
            competition_id,
        )
        raise RuntimeError(
            f"Deterministic hidden validation adapter failed for {competition_id}: {exc}"
        ) from exc
