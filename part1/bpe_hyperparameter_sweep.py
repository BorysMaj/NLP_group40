"""Resumable BPE hyperparameter sweep for the selected log-linear system."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.trainers import BpeTrainer

sys.path.insert(0, str(Path(__file__).parent))
from goemotions_pipeline import (  # noqa: E402
    load_splits,
    locate_data,
    multilabel_metrics,
    select_global_threshold,
)


CONFIGURATIONS = [
    {"vocab_size": 1000, "min_frequency": 2, "name": "baseline_v1000_mf2"},
    {"vocab_size": 500, "min_frequency": 2, "name": "vocab_v500_mf2"},
    {"vocab_size": 1500, "min_frequency": 2, "name": "vocab_v1500_mf2"},
    {"vocab_size": 1000, "min_frequency": 3, "name": "minfreq_v1000_mf3"},
    {"vocab_size": 1000, "min_frequency": 4, "name": "minfreq_v1000_mf4"},
]
SEEDS = [0, 3, 5]
N_SPLITS = 5
THRESHOLDS = np.arange(0.10, 0.91, 0.05)


def atomic_write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def train_bpe(texts: pd.Series, vocab_size: int, min_frequency: int) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Metaspace()
    tokenizer.train_from_iterator(
        texts,
        trainer=BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["[UNK]"],
        ),
    )
    return tokenizer


def probability_array(model, texts: pd.Series) -> np.ndarray:
    probabilities = model.predict_proba(texts)
    if isinstance(probabilities, list):
        probabilities = np.column_stack([item[:, 1] for item in probabilities])
    return np.asarray(probabilities, dtype=np.float32)


def run_configuration(
    config: dict[str, int | str],
    texts: pd.Series,
    targets: np.ndarray,
    output_dir: Path,
) -> dict[str, object]:
    name = str(config["name"])
    checkpoint_path = output_dir / f"{name}.json"
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    else:
        checkpoint = {"config": config, "folds": [], "status": "running"}

    tokenizer = train_bpe(
        texts,
        int(config["vocab_size"]),
        int(config["min_frequency"]),
    )
    vocab = tokenizer.get_vocab()
    analyzer = lambda text: tokenizer.encode(text).tokens
    prototype = make_pipeline(
        CountVectorizer(
            binary=True,
            analyzer=analyzer,
            vocabulary=vocab,
            lowercase=False,
        ),
        OneVsRestClassifier(
            LogisticRegression(solver="liblinear", max_iter=100, random_state=0),
            n_jobs=1,
        ),
    )

    existing = {
        (int(row["run"]), int(row["fold"])): row for row in checkpoint["folds"]
    }
    fold_parts = []
    for run, seed in enumerate(SEEDS):
        splitter = MultilabelStratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=seed,
        )
        for fold, (train_idx, validation_idx) in enumerate(
            splitter.split(texts, targets)
        ):
            key = (run, fold)
            if key in existing:
                row = existing[key]
                fold_parts.append(
                    (
                        run,
                        fold,
                        np.asarray(row["gold"], dtype=np.int8),
                        np.asarray(row["probabilities"], dtype=np.float32),
                    )
                )
                continue

            started = time.time()
            model = prototype
            model.fit(texts.iloc[train_idx], targets[train_idx])
            probabilities = probability_array(model, texts.iloc[validation_idx])
            gold = targets[validation_idx]
            row = {
                "run": run,
                "seed": seed,
                "fold": fold,
                "n_train": len(train_idx),
                "n_validation": len(validation_idx),
                "fit_seconds": time.time() - started,
                "gold": gold.tolist(),
                "probabilities": probabilities.tolist(),
            }
            checkpoint["folds"].append(row)
            checkpoint["folds"].sort(key=lambda item: (item["run"], item["fold"]))
            atomic_write_json(checkpoint_path, checkpoint)
            fold_parts.append((run, fold, gold, probabilities))
            print(
                f"{name}: run {run + 1}/{len(SEEDS)}, fold {fold + 1}/{N_SPLITS} "
                f"({row['fit_seconds'] / 60:.1f} min)",
                flush=True,
            )

    gold = np.vstack([part[2] for part in fold_parts])
    probabilities = np.vstack([part[3] for part in fold_parts])
    threshold, _ = select_global_threshold(gold, probabilities, THRESHOLDS)
    metrics = multilabel_metrics(gold, probabilities, threshold)
    result = {
        **config,
        "actual_vocab_size": len(vocab),
        "median_bpe_tokens": float(
            np.median([len(tokenizer.encode(text).ids) for text in texts])
        ),
        "p95_bpe_tokens": float(
            np.percentile([len(tokenizer.encode(text).ids) for text in texts], 95)
        ),
        "threshold": threshold,
        **metrics,
        "status": "complete",
    }
    checkpoint["result"] = result
    checkpoint["status"] = "complete"
    atomic_write_json(checkpoint_path, checkpoint)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    data_dir = args.data_dir or locate_data(Path.cwd())
    output_dir = args.output_dir or (data_dir.parent / "goemotions_results" / "bpe_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)
    splits, targets, _ = load_splits(data_dir)
    texts = splits["train"]["text"].reset_index(drop=True)
    y = targets["train"]
    results_path = output_dir / "results.csv"
    log_path = output_dir / "progress.log"

    results = []
    for config in CONFIGURATIONS:
        started = time.time()
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"START {config['name']}\n")
        try:
            result = run_configuration(config, texts, y, output_dir)
            results.append(result)
            pd.DataFrame(results).to_csv(results_path, index=False)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(
                    f"COMPLETE {config['name']} "
                    f"elapsed_minutes={(time.time() - started) / 60:.2f}\n"
                )
        except Exception as error:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"ERROR {config['name']}: {error!r}\n")
            raise

    print(f"Completed sweep. Results: {results_path}")


if __name__ == "__main__":
    main()
