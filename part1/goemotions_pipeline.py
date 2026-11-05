"""Reusable, testable helpers for the self-contained GoEmotions experiment."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MultiLabelBinarizer


def locate_data(start: Path | None = None) -> Path:
    """Find the checked-in ``NLP1_2026/data`` directory from any notebook cwd."""
    roots = []
    if start is not None:
        roots.append(start.resolve())
    roots.extend([Path.cwd().resolve(), Path(__file__).resolve().parents[2]])
    candidates = []
    for root in roots:
        candidates.extend(
            [
                root / "data",
                root / "NLP1_2026" / "data",
                root / "archive" / "data",
            ]
        )
        candidates.extend(parent / "data" for parent in root.parents)
    for candidate in candidates:
        if (candidate / "train.tsv").is_file() and (candidate / "emotions.txt").is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find GoEmotions data. Searched: {searched}")


def load_splits(
    data_dir: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray], list[str]]:
    """Load the three supplied TSV splits and return binary multilabel targets."""
    labels = (data_dir / "emotions.txt").read_text(encoding="utf-8").splitlines()
    if len(labels) != 28 or len(set(labels)) != len(labels):
        raise ValueError(f"Expected 28 unique emotion labels, found {len(labels)}")
    mlb = MultiLabelBinarizer(classes=np.arange(len(labels))).fit([[]])
    splits: dict[str, pd.DataFrame] = {}
    targets: dict[str, np.ndarray] = {}
    for name in ("train", "dev", "test"):
        frame = pd.read_csv(
            data_dir / f"{name}.tsv",
            sep="\t",
            header=None,
            names=["text", "label_ids", "id"],
            quoting=csv.QUOTE_NONE,
        )
        frame["label_ids"] = frame["label_ids"].map(
            lambda value: [int(item) for item in str(value).split(",")]
        )
        if frame["text"].isna().any() or frame["label_ids"].isna().any():
            raise ValueError(f"{name} contains missing text or labels")
        if not frame["id"].is_unique:
            raise ValueError(f"{name} contains duplicate IDs")
        splits[name] = frame
        targets[name] = mlb.transform(frame["label_ids"])
    return splits, targets, labels


def dataset_audit(
    splits: dict[str, pd.DataFrame], targets: dict[str, np.ndarray]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return split summary, label counts, and cross-split text overlaps."""
    summary = []
    label_counts = []
    for name, frame in splits.items():
        target = targets[name]
        lengths = frame["text"].str.len()
        summary.append(
            {
                "split": name,
                "examples": len(frame),
                "missing": int(frame.isna().any(axis=1).sum()),
                "duplicate_text": int(frame["text"].duplicated().sum()),
                "mean_labels": float(target.sum(axis=1).mean()),
                "multilabel_percent": float(100 * (target.sum(axis=1) > 1).mean()),
                "mean_chars": float(lengths.mean()),
                "median_chars": float(lengths.median()),
                "p95_chars": float(lengths.quantile(0.95)),
            }
        )
        label_counts.append(
            pd.DataFrame({"label": np.arange(target.shape[1]), "count": target.sum(axis=0)})
            .assign(split=name)
        )
    overlap_rows = []
    names = list(splits)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap_rows.append(
                {
                    "left": left,
                    "right": right,
                    "shared_ids": len(set(splits[left]["id"]) & set(splits[right]["id"])),
                    "shared_texts": len(
                        set(splits[left]["text"]) & set(splits[right]["text"])
                    ),
                }
            )
    return (
        pd.DataFrame(summary),
        pd.concat(label_counts, ignore_index=True),
        pd.DataFrame(overlap_rows),
    )


def multilabel_metrics(
    gold: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Compute the complete metric set used for model comparison."""
    predicted = (probabilities >= threshold).astype(int)
    return {
        "macro_f1": f1_score(gold, predicted, average="macro", zero_division=0),
        "micro_f1": f1_score(gold, predicted, average="micro", zero_division=0),
        "macro_precision": precision_score(
            gold, predicted, average="macro", zero_division=0
        ),
        "micro_precision": precision_score(
            gold, predicted, average="micro", zero_division=0
        ),
        "macro_recall": recall_score(gold, predicted, average="macro", zero_division=0),
        "micro_recall": recall_score(gold, predicted, average="micro", zero_division=0),
        "hamming_loss": hamming_loss(gold, predicted),
    }


def select_global_threshold(
    gold: np.ndarray,
    probabilities: np.ndarray,
    candidates: Iterable[float] = np.arange(0.1, 0.91, 0.05),
) -> tuple[float, pd.DataFrame]:
    """Select one global threshold using macro-F1 on validation predictions."""
    rows = []
    for threshold in candidates:
        metrics = multilabel_metrics(gold, probabilities, float(threshold))
        rows.append({"threshold": float(threshold), **metrics})
    table = pd.DataFrame(rows).sort_values(
        ["macro_f1", "micro_f1", "threshold"], ascending=[False, False, True]
    )
    return float(table.iloc[0]["threshold"]), table.reset_index(drop=True)


def validate_binary_targets(targets: Sequence[np.ndarray]) -> None:
    """Raise a clear error if any target matrix is not a 0/1 multilabel matrix."""
    for target in targets:
        values = np.unique(target)
        if not set(values).issubset({0, 1}):
            raise ValueError(f"Targets must be binary, found values {values.tolist()}")
