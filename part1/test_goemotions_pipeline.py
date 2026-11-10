import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from goemotions_pipeline import (
    multilabel_metrics,
    plot_bpe_token_lengths,
    select_global_threshold,
)


def test_multilabel_metrics_return_all_required_measures():
    gold = np.array([[1, 0], [0, 1], [1, 1]])
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])

    metrics = multilabel_metrics(gold, probabilities, threshold=0.5)

    assert set(metrics) == {
        "macro_f1",
        "micro_f1",
        "macro_precision",
        "micro_precision",
        "macro_recall",
        "micro_recall",
        "hamming_loss",
    }
    assert metrics["micro_f1"] > 0.0
    assert metrics["hamming_loss"] < 1.0


def test_global_threshold_selection_uses_macro_f1():
    gold = np.array([[1], [1], [0], [0]])
    probabilities = np.array([[0.4], [0.45], [0.1], [0.2]])

    threshold, table = select_global_threshold(gold, probabilities, [0.3, 0.5])

    assert threshold == 0.3
    assert table.iloc[0]["macro_f1"] == 1.0


def test_bpe_token_length_plot_is_saved(tmp_path):
    class Encoded:
        def __init__(self, length):
            self.ids = list(range(length))

    class Tokenizer:
        def encode(self, text):
            return Encoded(len(text.split()))

    output_path = tmp_path / "plots" / "bpe_lengths.png"
    lengths = plot_bpe_token_lengths(
        ["one two", "one two three", "one"],
        Tokenizer(),
        output_path,
        max_length=3,
    )

    assert lengths.tolist() == [2, 3, 1]
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
