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
    load_mlebench_lite_competition_ids,
    logger,
    read_csv_rows,
    save_runtime_state,
)
from .verify import verify_split_artifacts


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
    if competition_id == "aerial-cactus-identification":
        test_size = 0.19
    elif competition_id == "leaf-classification":
        # Leaf has 99 classes and only 891 training rows, so use a larger
        # stratified holdout to keep the hidden validation split representative.
        test_size = 0.2
        stratify = df["species"]
    elif competition_id == "new-york-city-taxi-fare-prediction":
        test_size = min(9914, max(1, len(df) // 10))
    elif competition_id == "tabular-playground-series-may-2022":
        test_size = min(100_000, max(1, len(df) // 10))
    elif competition_id == "histopathologic-cancer-detection":
        existing_test = len(list((visible_output / "test").glob("*.tif")))
        test_size = max(1, min(len(df) - 1, existing_test))
    else:
        test_size = 0.1
    return train_test_split(df, test_size=test_size, random_state=0, stratify=stratify)


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

    config = {
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
        "leaf-classification": {
            "train_csv": "train.csv", "id": "id", "target": "species", "one_hot": True,
            "validation_cols": ["id"], "assets": [("images", "images", [".jpg"], "test.csv")],
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
        "tabular-playground-series-dec-2021": {
            "train_csv": "train.csv", "id": "Id", "target": "Cover_Type",
            "validation_drop": ["Cover_Type"], "assets": [],
        },
        "tabular-playground-series-may-2022": {
            "train_csv": "train.csv", "id": "id", "target": "target",
            "validation_drop": ["target"], "assets": [],
        },
    }[competition_id]

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


def _prepare_denoising_second_split(visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path):
    import numpy as np
    import pandas as pd
    from PIL import Image
    from sklearn.model_selection import train_test_split

    train_ids = sorted(path.stem for path in (visible_output / "train").glob("*.png"))
    keep_ids, val_ids = train_test_split(train_ids, test_size=0.1, random_state=0)
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
    }


def _prepare_dogs_vs_cats_second_split(visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    train_files = sorted((visible_output / "train").glob("*.jpg"))
    keep_files, val_files = train_test_split(train_files, test_size=0.1, random_state=0)
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
        test_size=0.1,
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
    }


def _prepare_text_normalization_second_split(competition_id, visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    prefix = "en" if "english" in competition_id else "ru"
    train_name = f"{prefix}_train.csv"
    test_name = f"{prefix}_test_2.csv"
    train_df = pd.read_csv(visible_output / train_name)
    sentence_ids = train_df["sentence_id"].unique()
    keep_sentence_ids, val_sentence_ids = train_test_split(sentence_ids, test_size=0.1, random_state=0)
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
    if competition_id == "dogs-vs-cats-redux-kernels-edition":
        return _prepare_dogs_vs_cats_second_split(visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path)
    if competition_id == "random-acts-of-pizza":
        return _prepare_random_pizza_second_split(visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path)
    if competition_id == "detecting-insults-in-social-commentary":
        return _prepare_detecting_insults_second_split(visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path)
    if competition_id in {"text-normalization-challenge-english-language", "text-normalization-challenge-russian-language"}:
        return _prepare_text_normalization_second_split(competition_id, visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path)
    if competition_id == "the-icml-2013-whale-challenge-right-whale-redux":
        return _prepare_whale_second_split(visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path)
    if competition_id == "mlsp-2013-birds":
        return _prepare_mlsp_second_split(visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path)
    return _prepare_generic_dataframe_second_split(competition_id, visible_output, validation_dir, hidden_answers_path, hidden_sample_submission_path, visible_sample_submission_path)


def maybe_prepare_mlebench_lite_hidden_validation(cfg) -> dict[str, Any] | None:
    competition_id = str(getattr(getattr(cfg, "hidden_validation", None), "competition_name", "")).strip() or infer_competition_id(cfg)
    lite_ids = load_mlebench_lite_competition_ids()
    if competition_id not in lite_ids:
        return None

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
    visible_validation_dir = visible_output / "visible_validation"
    hidden_validation_dir = visible_output / "hidden_validation"
    visible_answers_path = visible_output / "visible_validation_answers.csv"
    visible_sample_submission_path = visible_output / "sampleVisibleValidationSubmission.csv"
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
        visible_validation_dir.mkdir(parents=True, exist_ok=True)
        visible_prep_result = _prepare_second_split_hidden_validation_artifacts(
            competition_id,
            visible_output,
            visible_validation_dir,
            visible_answers_path,
            visible_sample_submission_path,
            visible_sample_submission_path,
        )
        authoritative_train_sources = visible_prep_result["authoritative_train_sources"]
        authoritative_visible_validation_sources = _rewrite_validation_sources(
            visible_prep_result["authoritative_validation_sources"],
            visible_validation_dir,
        )
        authoritative_hidden_validation_sources = _rewrite_validation_sources(
            hidden_prep_result["authoritative_validation_sources"],
            hidden_validation_dir,
        )
        visible_answer_granularity = visible_prep_result["hidden_answer_granularity"]
        hidden_answer_granularity = hidden_prep_result["hidden_answer_granularity"]
        authoritative_label_sources = [str(visible_answers_path)]
        authoritative_hidden_answer_sources = [str(hidden_answers_path)]
        strategy_summary = (
            "Deterministic mle-bench lite hidden-validation adapter. "
            "Visible input is copied from cfg.data_dir, then two deterministic splits are performed "
            "over the prepared training data only. The first holdout is exposed under hidden_validation/ "
            "with labels kept outside the agent workspace, and the second holdout is exposed under "
            "visible_validation/ with labels available in visible_validation_answers.csv. Runtime search "
            "uses only the scored visible split, while hidden validation is stored separately for final selection."
        )

        _write_adapter_evidence(
            cfg=cfg,
            visible_output=visible_output,
            hidden_output=attempt_dir,
            evidence_dir=evidence_dir,
            authoritative_train_sources=authoritative_train_sources,
            authoritative_validation_sources=authoritative_visible_validation_sources + authoritative_hidden_validation_sources,
            authoritative_hidden_answer_sources=authoritative_hidden_answer_sources,
            strategy_summary=strategy_summary,
        )

        hidden_count = len(read_csv_rows(hidden_answers_path))
        visible_validation_count = len(read_csv_rows(visible_answers_path))
        visible_count = _estimate_visible_count(visible_output, authoritative_train_sources)
        manifest = {
            "visible_input_dir": str(visible_output),
            "visible_validation_dir": str(visible_validation_dir),
            "visible_answers_path": str(visible_answers_path),
            "visible_sample_submission_path": str(visible_sample_submission_path),
            "hidden_validation_dir": str(hidden_validation_dir),
            "hidden_answers_path": str(hidden_answers_path),
            "hidden_sample_submission_path": str(hidden_sample_submission_path),
            "competition_id": competition_id,
            "split_seed": "mlebench_prepare",
            "strategy_summary": strategy_summary,
            "visible_count": visible_count,
            "visible_validation_count": visible_validation_count,
            "hidden_count": hidden_count,
            "contract_version": "mlebench_lite_adapter_v2",
            "evidence_dir": str(evidence_dir),
            "authoritative_train_sources": authoritative_train_sources,
            "authoritative_label_sources": authoritative_label_sources,
            "authoritative_visible_validation_sources": authoritative_visible_validation_sources,
            "authoritative_hidden_validation_sources": authoritative_hidden_validation_sources,
            "authoritative_hidden_answer_sources": authoritative_hidden_answer_sources,
            "visible_answer_granularity": visible_answer_granularity,
            "hidden_answer_granularity": hidden_answer_granularity,
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
                "visible_validation_dir": manifest["visible_validation_dir"],
                "visible_answers_path": manifest["visible_answers_path"],
                "visible_sample_submission_path": manifest["visible_sample_submission_path"],
                "hidden_validation_dir": manifest["hidden_validation_dir"],
                "hidden_answers_path": manifest["hidden_answers_path"],
                "hidden_sample_submission_path": manifest["hidden_sample_submission_path"],
                "manifest_path": str(manifest_path),
                "splitter_attempts": 0,
                "reviewer_status": "mlebench_lite_adapter",
            }
        )
        logger.info(
            "Prepared deterministic mle-bench lite hidden validation adapter for %s",
            competition_id,
        )
        return save_runtime_state(cfg, state)
    except Exception as exc:
        logger.exception(
            "Deterministic mle-bench lite hidden validation adapter failed for %s",
            competition_id,
        )
        raise RuntimeError(
            f"Deterministic mle-bench lite hidden validation adapter failed for {competition_id}: {exc}"
        ) from exc
