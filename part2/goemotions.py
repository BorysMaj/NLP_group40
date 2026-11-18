"""Reproduce the GoEmotions Part 2 runs from the group repository root."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from nltk.tokenize import TweetTokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.trainers import BpeTrainer
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "part1"))
from goemotions_pipeline import (
    dataset_audit, load_splits, locate_data, multilabel_metrics, select_global_threshold,
)

PROBES = {
    "joy": ["happy", "glad", "excited", "sad", "upset"],
    "anger": ["angry", "furious", "mad", "calm", "relaxed"],
    "sad": ["sadness", "unhappy", "miserable", "happy", "delighted"],
    "love": ["adore", "affection", "cherish", "hate", "despise"],
}
PROBE_QUERIES = ["happy", "angry", "sad", "love", "lol", "ugh"]
THRESHOLDS = np.arange(2, 61, 2) / 100


def corpus_vectors(sequences, size, dimension):
    counts = Counter()
    for ids in sequences:
        for left, right in zip(ids, ids[1:]):
            if min(left, right) > 1:
                counts[left, right] += 1
                counts[right, left] += 1
        for left, right in zip(ids, ids[2:]):
            if min(left, right) > 1:
                counts[left, right] += 1
                counts[right, left] += 1
    rows, cols = np.array(list(counts)).T
    values = np.array(list(counts.values()), dtype=float)
    marginals = np.bincount(rows, weights=values, minlength=size)
    ppmi = np.maximum(0, np.log(values * values.sum() / (marginals[rows] * marginals[cols])))
    matrix = coo_matrix((ppmi, (rows, cols)), shape=(size, size)).tocsr()
    vectors = TruncatedSVD(dimension, random_state=0).fit_transform(matrix)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
    return vectors.astype(np.float32)


def read_vectors(path, wanted):
    opener = gzip.open if path.suffix == ".gz" else open
    found = {}
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            word, _, values = line.rstrip().partition(" ")
            if word in wanted:
                found[word] = np.fromstring(values, dtype=np.float32, sep=" ")
    if not found:
        raise ValueError(f"No vocabulary matches in {path}")
    return found


class Classifier(nn.Module):
    def __init__(self, matrix, encoder, trainable, head, dropout, labels):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(matrix.copy()),
                                                      freeze=not trainable, padding_idx=0)
        dimension = matrix.shape[1]
        self.rnn = nn.GRU(dimension, 64, batch_first=True) if encoder == "gru" else None
        width = 64 if self.rnn is not None else dimension
        self.head = (nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Dropout(dropout),
                                   nn.Linear(64, labels)) if head == "mlp" else
                     nn.Sequential(nn.Dropout(dropout), nn.Linear(width, labels)))

    def forward(self, ids):
        mask = ids.ne(0)
        lengths = mask.sum(1).clamp_min(1)
        embeddings = self.embedding(ids)
        if self.rnn is not None:
            packed = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True,
                                          enforce_sorted=False)
            _, state = self.rnn(packed)
            features = state[-1]
        else:
            features = (embeddings * mask.unsqueeze(2)).sum(1) / lengths.unsqueeze(1)
        return self.head(features)


def fit(matrix, config, arrays, targets, device, epochs, curves):
    name, encoder, source, trainable, head, dropout, lr, fraction, seed = config
    torch.manual_seed(seed)
    model = Classifier(matrix, encoder, trainable, head, dropout, targets["train"].shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    subset = rng.permutation(len(arrays["train"]))[:int(len(arrays["train"]) * fraction)]

    @torch.no_grad()
    def predict(split):
        model.eval()
        logits = torch.cat([model(batch.to(device)).cpu() for batch in arrays[split].split(256)])
        return logits.sigmoid().numpy(), criterion(logits, targets[split]).item()

    best = None
    for epoch in range(1, epochs + 1):
        model.train()
        loss_total = 0.
        for batch in np.array_split(rng.permutation(subset), int(np.ceil(len(subset) / 128))):
            optimizer.zero_grad(set_to_none=True)
            logits = model(arrays["train"][batch].to(device))
            loss = criterion(logits, targets["train"][batch].to(device))
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * len(batch)
        probabilities, dev_loss = predict("dev")
        metrics = multilabel_metrics(targets["dev"].numpy(), probabilities, .2)
        curves.append(dict(run=name, seed=seed, epoch=epoch, train_bce=loss_total / len(subset),
                           dev_bce=dev_loss, dev_macro_f1_at_020=metrics["macro_f1"]))
        if best is None or dev_loss < best[0]:
            best = (dev_loss, epoch, {key: value.detach().cpu().clone()
                                      for key, value in model.state_dict().items()})
    model.load_state_dict(best[2])
    return predict("dev")[0], predict("test")[0], best[1]


def make_plot(frame, x, y, path, ylabel):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    frame.plot.bar(x=x, y=y, ax=ax, color="#386b80")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--embedding", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 3, 5])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, help="Training rows for a smoke run")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(3)
    data = locate_data()
    splits, gold, labels = load_splits(data)
    if args.limit:
        splits["train"] = splits["train"].sample(n=args.limit, random_state=0)
        gold["train"] = gold["train"][splits["train"].index]
    audit, support, overlap = dataset_audit(splits, gold)
    audit.to_csv(args.out / "dataset_audit.csv", index=False)
    support["label"] = support.label.map(dict(enumerate(labels)))
    support.to_csv(args.out / "label_support.csv", index=False)
    overlap.to_csv(args.out / "split_overlap.csv", index=False)
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=False, strip_handles=False)
    words = {key: [tokenizer.tokenize(text) for text in frame.text] for key, frame in splits.items()}
    counts = Counter(word for sequence in words["train"] for word in sequence)
    vocabulary = {word: index for index, word in enumerate(sorted(counts), 2)}
    full_ids = {key: [[vocabulary.get(word, 1) for word in seq] for seq in rows]
                for key, rows in words.items()}
    cap = 40
    arrays = {key: torch.tensor([(ids[:cap] or [1]) + [0] * (cap - max(1, len(ids[:cap])))
                                 for ids in sequences]) for key, sequences in full_ids.items()}
    targets = {key: torch.tensor(value, dtype=torch.float32) for key, value in gold.items()}
    corpus = corpus_vectors(full_ids["train"], len(vocabulary) + 2, 50)
    matrices = {"reddit_ppmi": corpus}
    matched = {"reddit_ppmi": {word for word, idx in vocabulary.items() if corpus[idx].any()}}
    hashes = {}
    for spec in args.embedding:
        name, filename = spec.split("=", 1)
        path = Path(filename)
        with path.open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
        vectors = read_vectors(path, set(vocabulary))
        dimension = len(next(iter(vectors.values())))
        matrix = np.zeros((len(vocabulary) + 2, dimension), np.float32)
        for word, vector in vectors.items():
            matrix[vocabulary[word]] = vector
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
        matrices[name], matched[name] = matrix, set(vectors)
    args.out.joinpath("config.json").write_text(json.dumps({
        "arguments": vars(args), "torch": torch.__version__, "embedding_sha256": hashes,
        "split_sizes": {key: len(rows) for key, rows in words.items()},
        "labels": labels, "word_vocab": len(vocabulary), "token_cap": cap,
        "threshold_grid": THRESHOLDS.tolist(), "checkpoint": "minimum dev BCE",
        "selection": "mean dev macro-F1 across seeds, then dev micro-F1",
        "bpe_baseline": "Metaspace, vocab 1000, min frequency 2, uncapped, binary, OVR liblinear",
        "data_sha256": {key: hashlib.sha256((data / f"{key}.tsv").read_bytes()).hexdigest()
                        for key in splits},
    }, indent=2, default=str), encoding="utf-8")
    coverage, neighbors, probes = [], [], []
    for source, matrix in matrices.items():
        for split, sequences in words.items():
            lengths = np.array([len(seq) for seq in sequences])
            misses = np.array([sum(word not in matched[source] for word in seq) for seq in sequences])
            coverage.append(dict(source=source, split=split,
                                 token_coverage=1 - misses.sum() / lengths.sum(),
                                 comments_with_oov=(misses > 0).mean(), truncated=(lengths > cap).mean()))
        for query in PROBE_QUERIES:
            if query not in matched[source]:
                continue
            scores = matrix @ matrix[vocabulary[query]]
            candidates = sorted(matched[source] - {query}, key=lambda word: (-scores[vocabulary[word]], word))
            for rank, word in enumerate(candidates[:8], 1):
                neighbors.append(dict(source=source, query=query, rank=rank, neighbor=word,
                                      cosine=scores[vocabulary[word]]))
        for query, candidates in PROBES.items():
            for order, word in enumerate(candidates):
                score = (float(matrix[vocabulary[query]] @ matrix[vocabulary[word]])
                         if query in matched[source] and word in matched[source] else np.nan)
                probes.append(dict(source=source, query=query, candidate=word,
                                   group="similar" if order < 3 else "opposite", cosine=score))
    pd.DataFrame(coverage).to_csv(args.out / "coverage.csv", index=False)
    pd.DataFrame(neighbors).to_csv(args.out / "neighbors.csv", index=False)
    pd.DataFrame(probes).to_csv(args.out / "emotion_probes.csv", index=False)
    results, curves, sweeps, predictions = [], [], [], {}

    def record(config, dev, test, epoch):
        name, encoder, source, trainable, head, dropout, lr, fraction, seed = config
        threshold, table = select_global_threshold(gold["dev"], dev, THRESHOLDS)
        sweeps.append(table.assign(run=name, seed=seed))
        row = dict(run=name, encoder=encoder, source=source, trainable=trainable, head=head,
                   dropout=dropout, lr=lr, fraction=fraction, seed=seed, epoch=epoch, threshold=threshold)
        row.update({f"dev_{k}": v for k,v in multilabel_metrics(gold["dev"], dev, threshold).items()})
        results.append(row)
        predictions[name, seed] = test >= threshold
        pd.DataFrame(results).to_csv(args.out / "runs.csv", index=False)
        pd.DataFrame(curves).to_csv(args.out / "learning_curves.csv", index=False)
        pd.concat(sweeps).to_csv(args.out / "threshold_sweeps.csv", index=False)
        print(name, seed, round(row["dev_macro_f1"], 4), flush=True)

    bpe = Tokenizer(BPE(unk_token="[UNK]"))
    bpe.pre_tokenizer = Metaspace()
    bpe.train_from_iterator(splits["train"].text, trainer=BpeTrainer(
        vocab_size=1000, min_frequency=2, special_tokens=["[UNK]"]))
    bpe.save(str(args.out / "bpe.json"))
    vectorizer = CountVectorizer(analyzer=lambda text: bpe.encode(text).tokens,
                                 vocabulary=bpe.get_vocab(), binary=True)
    baseline = OneVsRestClassifier(LogisticRegression(solver="liblinear", max_iter=100, random_state=0))
    baseline.fit(vectorizer.transform(splits["train"].text), gold["train"])
    record(("bpe", "bpe", "bpe", False, "linear", 0., 0., 1., 0),
           *[baseline.predict_proba(vectorizer.transform(splits[key].text)) for key in ("dev", "test")], 0)
    for source, matrix in matrices.items():
        dimension = matrix.shape[1]
        settings = [
            ("pool_frozen", "pool", False, "linear", .1, .003, 1., False),
            ("pool_mlp", "pool", False, "mlp", .1, .003, 1., False),
            ("pool_lr", "pool", False, "linear", .1, .001, 1., False),
            ("gru_frozen", "gru", False, "linear", .1, .003, 1., False),
            ("gru_tuned", "gru", True, "linear", .1, .003, 1., False),
            ("gru_random", "gru", True, "linear", .1, .003, 1., True),
            ("gru_tuned_quarter", "gru", True, "linear", .1, .003, .25, False),
            ("gru_random_quarter", "gru", True, "linear", .1, .003, .25, True),
            ("gru_dropout", "gru", True, "linear", .35, .003, 1., False),
        ]
        for variant, encoder, trainable, head, dropout, lr, fraction, random in settings:
            for seed in args.seeds:
                start = matrix
                if random:
                    start = np.random.default_rng(seed).normal(0, 1 / np.sqrt(dimension),
                                                                matrix.shape).astype(np.float32)
                    start[:2] = 0
                config = (f"{source}_{variant}", encoder, source, trainable, head,
                          dropout, lr, fraction, seed)
                record(config, *fit(start, config, arrays, targets, args.device, args.epochs, curves))

    frame = pd.DataFrame(results)
    summary = frame.groupby(["encoder", "run"], as_index=False).agg(
        dev_macro_f1=("dev_macro_f1", "mean"), seed_sd=("dev_macro_f1", "std"),
        dev_micro_f1=("dev_micro_f1", "mean"), seeds=("seed", "count"))
    summary = summary.sort_values(["dev_macro_f1", "dev_micro_f1"], ascending=False)
    summary.to_csv(args.out / "dev_summary.csv", index=False)
    winners = summary.drop_duplicates("encoder")
    final, per_label, slices = [], [], []
    test = gold["test"]
    features = {
        "negation": np.array([any(w in {"no", "not", "never", "n't", "cannot"} or w.endswith("n't")
                                  for w in seq) for seq in words["test"]]),
        "long_20plus": np.array([len(seq) >= 20 for seq in words["test"]]),
        "train_vocab_oov": np.array([1 in seq for seq in full_ids["test"]]),
    }
    for winner in winners.itertuples():
        for (name, seed), pred in predictions.items():
            if name != winner.run:
                continue
            final.append(dict(run=name, seed=seed, **multilabel_metrics(test, pred, .5)))
            f1 = f1_score(test, pred, average=None, zero_division=0)
            for i, label in enumerate(labels):
                per_label.append(dict(run=name, seed=seed, label=label, f1=f1[i],
                                      support=test[:, i].sum(), fp=(pred[:, i] & (test[:, i] == 0)).sum(),
                                      fn=(~pred[:, i] & (test[:, i] == 1)).sum()))
            for feature, mask in features.items():
                for present in [False, True]:
                    rows = mask == present
                    if rows.any():
                        slices.append(dict(run=name, seed=seed, feature=feature, present=present,
                                           n=rows.sum(), **multilabel_metrics(test[rows], pred[rows], .5)))
    final = pd.DataFrame(final)
    final.to_csv(args.out / "test_scores.csv", index=False)
    pd.DataFrame(per_label).to_csv(args.out / "test_per_label.csv", index=False)
    pd.DataFrame(slices).to_csv(args.out / "test_error_slices.csv", index=False)
    pd.DataFrame({"id": splits["test"].id, "gold": ["|".join(np.array(labels)[row.astype(bool)]) for row in test],
                  **{f"{name}_seed{seed}": ["|".join(np.array(labels)[row]) for row in pred]
                     for (name, seed), pred in predictions.items() if name in set(winners.run)}
                  }).to_csv(args.out / "test_predictions.csv", index=False)
    make_plot(pd.DataFrame(coverage).query("split == 'dev'"), "source", ["token_coverage"],
              args.out / "dev_coverage.png", "Fraction of tokens covered")
    fig, ax = plt.subplots(figsize=(10, max(4, .37 * len(summary))))
    ax.barh(summary.run, summary.dev_macro_f1, xerr=summary.seed_sd.fillna(0), color="#386b80")
    ax.invert_yaxis()
    ax.set_xlabel("Dev macro-F1 (mean ± seed SD)")
    fig.tight_layout()
    fig.savefig(args.out / "dev_comparison.png", dpi=180)
    plt.close(fig)
    make_plot(final.groupby("run", as_index=False)[["macro_f1", "micro_f1"]].mean(),
              "run", ["macro_f1", "micro_f1"], args.out / "test_comparison.png", "Test F1")
    history = pd.DataFrame(curves)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for name in set(winners.run) - {"bpe"}:
        average = history[history.run == name].groupby("epoch")[["train_bce", "dev_bce"]].mean()
        for ax, column in zip(axes, average.columns):
            ax.plot(average.index, average[column], marker="o", label=name)
            ax.set(xlabel="Epoch", ylabel=column.replace("_", " "))
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out / "learning_curves.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
