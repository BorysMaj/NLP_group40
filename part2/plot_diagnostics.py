"""Build presentation figures and paired diagnostics from a completed run."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("results", type=Path)
root = parser.parse_args().results
runs = pd.read_csv(root / "runs.csv")
for source, group in runs.query("encoder != 'bpe'").groupby("source"):
    scores = group.pivot(index="seed", columns="run", values="dev_macro_f1")
    scores.columns = scores.columns.str.removeprefix(source + "_")
    comparisons = {
        "MLP head vs linear": ("pool_mlp", "pool_frozen"),
        "Lower pool learning rate": ("pool_lr", "pool_frozen"),
        "GRU vs mean pooling": ("gru_frozen", "pool_frozen"),
        "Fine-tuning vs freezing": ("gru_tuned", "gru_frozen"),
        "Distributional vs random initialization": ("gru_tuned", "gru_random"),
        "Full vs quarter, distributional": ("gru_tuned", "gru_tuned_quarter"),
        "Full vs quarter, random": ("gru_random", "gru_random_quarter"),
        "Higher GRU dropout": ("gru_dropout", "gru_tuned"),
    }
    deltas = pd.DataFrame({name: scores[a] - scores[b] for name, (a, b) in comparisons.items()})
    deltas.to_csv(root / f"{source}_paired_dev_deltas.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    for index, (name, points) in enumerate(deltas.items()):
        ax.scatter(points, np.full(len(points), index), color="#386b80", alpha=.75)
        ax.scatter(points.mean(), index, color="#be644d", marker="D", zorder=3)
    ax.axvline(0, color=".5", linewidth=1)
    ax.set(yticks=range(len(deltas.columns)), yticklabels=deltas.columns,
           xlabel="Paired change in validation macro-F1 (dots = seeds, diamond = mean)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(root / f"{source}_paired_dev_deltas.png", dpi=180)
    plt.close(fig)

slices = pd.read_csv(root / "test_error_slices.csv")
means = slices.groupby(["feature", "present", "run"], as_index=False).agg(
    micro_f1=("micro_f1", "mean"), n=("n", "first"))
means["group"] = means.apply(
    lambda row: f"{row.feature} · {'yes' if row.present else 'no'} (n={row.n})", axis=1)
fig, ax = plt.subplots(figsize=(11, 5))
means.pivot(index="group", columns="run", values="micro_f1").plot.barh(ax=ax)
ax.set(xlabel="Test micro-F1 (mean across seeds)", ylabel="")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(root / "test_error_slices.png", dpi=180)
plt.close(fig)

labels = pd.read_csv(root / "test_per_label.csv")
average = labels.groupby(["label", "run"], as_index=False).agg(
    f1=("f1", "mean"), support=("support", "first"))
support = average.drop_duplicates("label").set_index("label").support.sort_values()
fig, ax = plt.subplots(figsize=(10, 9))
average.pivot(index="label", columns="run", values="f1").reindex(support.index).plot.barh(ax=ax)
ax.set(yticklabels=[f"{label} ({n})" for label, n in support.items()],
       xlabel="Test F1 (mean across seeds)", ylabel="Emotion (test support)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(root / "test_per_label.png", dpi=180)
plt.close(fig)

probes = pd.read_csv(root / "emotion_probes.csv")
for source, rows in probes.groupby("source"):
    rows = rows.dropna(subset=["cosine"]).copy()
    if rows.empty:
        continue
    rows["pair"] = rows["query"] + " / " + rows["candidate"]
    colors = rows.group.map({"similar": "#386b80", "opposite": "#be644d"})
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(rows.pair, rows.cosine, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Cosine similarity (blue = related probe, red = contrasting probe)")
    fig.tight_layout()
    fig.savefig(root / f"{source}_emotion_probes.png", dpi=180)
    plt.close(fig)

support = pd.read_csv(root / "label_support.csv").query("split == 'train'").sort_values("count")
fig, ax = plt.subplots(figsize=(8, 8))
ax.barh(support.label, support["count"], color="#386b80")
ax.set(xlabel="Training comments carrying the label", xscale="log")
fig.tight_layout()
fig.savefig(root / "train_label_support.png", dpi=180)
plt.close(fig)

# Recheck headline scores on test comments with no exact text match in train or dev.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "part1"))
from goemotions_pipeline import load_splits, locate_data, multilabel_metrics

splits, targets, labels = load_splits(locate_data())
seen = set(splits["train"].text) | set(splits["dev"].text)
unseen = ~splits["test"].text.isin(seen).to_numpy()
predictions = pd.read_csv(root / "test_predictions.csv", keep_default_na=False)
if predictions.id.tolist() != splits["test"].id.tolist():
    raise ValueError("Test prediction IDs do not align with the current test split")
sensitivity = []
for name in predictions.columns.difference(["id", "gold"]):
    predicted_labels = predictions[name].str.split("|")
    binary = np.array([[label in row for label in labels] for row in predicted_labels])
    sensitivity.append(dict(run_seed=name, n=int(unseen.sum()),
                            **multilabel_metrics(targets["test"][unseen], binary[unseen], .5)))
pd.DataFrame(sensitivity).to_csv(root / "test_without_text_overlap.csv", index=False)
