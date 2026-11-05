# NLP Group 40 - GoEmotions

This repository contains the NLP1 project experiment for multilabel emotion
classification on the supplied GoEmotions data.

## Setup

```powershell
python -m pip install -r requirements.txt
```

Open [`part1/GoEmotions_Project_Proposition.ipynb`](part1/GoEmotions_Project_Proposition.ipynb)
and select the environment containing these dependencies.

## Experiment

The proposition notebook is the self-contained experiment. It:

- loads and validates the supplied TSV files;
- preserves all 28 multilabel emotion targets;
- trains BPE once on all official training texts;
- compares BPE indicators, counts, and TF-IDF;
- compares logistic regression, a one-hidden-layer MLP, and a two-hidden-layer MLP;
- uses repeated multilabel-stratified cross-validation;
- selects one global decision threshold per system from out-of-fold predictions;
- reports macro/micro precision, recall, F1, Hamming loss, and subset accuracy;
- performs Bayesian correlated-t and ROPE analysis;
- evaluates the selected system once on the untouched test split;
- reports per-emotion results and qualitative error examples.

Results are written to `NLP1_2026/goemotions_results/` when the notebook is
run. The notebook uses the checked-in data and does not download or substitute
another GoEmotions version.

Run focused helper tests with:

```powershell
python -m pytest part1/test_goemotions_pipeline.py
```

Record AI assistance and external sources in the final report or blog post.
