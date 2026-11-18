# GoEmotions Part 2 experiment log

## Run diary

| Stage | Decision | Evidence or reason |
|---|---|---|
| Recover Part 1 | Refit BPE indicator + one-vs-rest logistic regression using the official training split. | The saved notebook uses Metaspace, vocabulary 1,000, minimum frequency 2 and an uncapped analyzer. Earlier slide scores are historical CV references. |
| Define the task | Keep all 28 labels, including neutral. Train sigmoid classifiers with binary cross-entropy; predict every label above a shared threshold, allowing an empty label set. | Some comments have more than one gold label. Macro-F1 is the main selection metric, with micro-F1 and per-label FP/FN alongside it. |
| Pick pretrained candidates | Try uncased English GloVe Twitter 50d and Wiki/Gigaword 50d. Lowercase and use TweetTokenizer, preserving punctuation, handles, and repeated letters. | Twitter is an informal English domain and Wikipedia/Gigaword is a contrasting domain. Matched width removes one source of variation. Exact lookup can miss source-specific normalized forms or Reddit spellings. |
| Handle download failure | Run 50d Reddit PPMI + truncated SVD as a corpus-derived pilot. | GitHub DNS failed from the terminal. The pilot vectors are learned without labels, using only official training comments and a symmetric two-token context window. They are unit normalized. |
| Audit tokenization | Use 40 real token positions for dense encoders, reserved padding/unknown IDs, and train-vocabulary lookup. | The cap truncates about 0.021% of training comments and 0.018% of dev comments. The BPE reference retains its original uncapped analyzer. |
| Audit coverage | Measure token coverage and comments with at least one uncovered token. | PPMI covers 97.77% of dev tokens and 26.34% of dev comments contain an uncovered token. Coverage is calculated before truncation. External-vector coverage also reflects the shared train-vocabulary restriction. |
| Probe similarities | Record eight cosine neighbors per query and compare related and contrasting emotion-word pairs. | `happy` has `:)`, `nice`, and `sweet` as its top three candidates. `angry` has useful neighbors such as `upset` and `pissed` alongside `expecting`. `love`–`hate` scores 0.722 versus `love`–`adore` at 0.299. Shared contexts can bring opposite sentiments together. Candidate neighbors are restricted to the train vocabulary. |
| Set the search budget | Use seeds 0, 3, and 5 and five epochs per dense configuration on fixed official splits. Run single-factor comparisons around the reference settings. | Seed variation captures initialization and batch order under a fixed split. The logistic baseline is one fit. Scores from the same seed can be compared in pairs. |
| Define checkpoint and threshold rules | Pick the checkpoint by dev BCE. Sweep thresholds from 0.02 to 0.60 in steps of 0.02 on dev, maximizing macro-F1 with micro-F1 as a tie-breaker. | This keeps checkpoint and decision selection on dev. A fixed 0.20 threshold is used for the epoch-by-epoch macro-F1 trace only. |
| Plan error analysis | Compare dev-selected BPE, pool and GRU systems on matched test IDs. Slice by negation, length of 20+ tokens, and training-vocabulary OOV. | Report slice sizes and label FP/FN counts. Slice differences are descriptive and can reflect class mix. The negation feature captures surface cues and misses semantic scope. |

## Controlled comparisons

- Pooling starts with frozen vectors, masked mean, linear head, dropout 0.10, and AdamW at 0.003. Compare a one-hidden-layer MLP and a learning rate of 0.001 separately.
- GRU uses a 64-dimensional hidden state and the final valid state. Compare frozen, fine-tuned, and trainable random starts, then a dropout change from 0.10 to 0.35.
- Quarter-data runs use nested random subsets of labeled training data. The vocabulary and corpus distributional vectors still use all unlabeled training text. This measures sensitivity to labeled data size under that setup.
- Compare the corpus and external embeddings in the same model settings. Matching dimensions is easiest with the 50d Twitter and 50d Wiki/Gigaword variants. Different vocabulary coverage remains part of the system comparison.
- Baseline versus dense comparisons change representation, tokenizer, and fitting procedure. The frozen pool-versus-GRU comparison holds tokenizer, vectors, and linear head type fixed, while recurrent parameter count also changes.
- Best-of-grid dev results can be optimistic. Test evaluation is for the dev-selected configurations. Fixed splits and three seeds give a limited view of uncertainty.

## Reproduce

From `NLP_group40`:

```powershell
python part2/goemotions.py --out part2/results/reproduction --epochs 5 --seeds 0 3 5
python part2/plot_diagnostics.py part2/results/reproduction
```

To add GloVe, pass `--embedding twitter50=PATH_TO_GLOVE_TWITTER_50` and `--embedding wiki50=PATH_TO_GLOVE_WIKI_GIGAWORD_50` to a fresh run. The loader accepts plain-text or gzip word-vector files, with one token followed by space-separated coordinates per row. Source and dimensions are documented on the [Stanford GloVe project page](https://nlp.stanford.edu/projects/glove/).

`config.json` fingerprints the split files and any external vectors. `runs.csv`, `threshold_sweeps.csv`, and `learning_curves.csv` retain the configuration and model-selection history. `test_predictions.csv` joins system predictions and gold labels by comment ID for error review. Results directories are ignored by the group repository; keep the run folder with the notebook when moving the experiment.

## Dataset audit

The official splits contain 43,410 train, 5,426 dev, and 5,427 test comments. About 16.36% of training comments have multiple labels. Neutral appears on 14,219 training comments, while grief appears on 77. This imbalance motivates macro-F1 and inspecting label support alongside F1 and error counts.

The split files share zero IDs. There are 41 distinct exact text matches between train and dev, 32 between train and test, and 10 between dev and test. Keep the official split for the headline comparison and report a test sensitivity check excluding any exact matches to train or dev text.

## Pilot results and interpretation

| Dev-selected system | Dev macro-F1 | Test macro-F1 | Test micro-F1 |
|---|---:|---:|---:|
| BPE indicators + logistic regression | 0.391 | 0.358 | 0.490 |
| Frozen Reddit vectors + mean pool + MLP | 0.141 | 0.141 | 0.275 |
| Trainable random word vectors + GRU | 0.324 | 0.317 | 0.487 |

Neural scores are means across seeds 0, 3, and 5. Random GRU starts won the dev macro-F1 selection, while the fine-tuned PPMI GRU led dev micro-F1 at 0.503. BPE remained the best macro-F1 system on the official test set. Excluding the 42 test comments with exact text overlap with train or dev preserved the ordering.

- **Pool design.** The MLP improved over a linear head by 0.059 macro-F1 on average across paired seeds. Lowering the linear head learning rate from 0.003 to 0.001 reduced macro-F1 by about 0.006 under the five-epoch budget.
- **Encoder and embedding updates.** A frozen GRU improved on frozen linear pooling by 0.083 validation macro-F1. Fine-tuning the GRU embeddings gained another 0.151. The random trainable GRU beat the fine-tuned PPMI GRU in macro-F1 at each seed, with an average margin of 0.008. The result applies to the tested PPMI vectors and training settings.
- **Data size.** At one quarter of the labeled data, PPMI initialization scored 0.127 versus 0.112 for random starts. The PPMI advantage appeared in two of three seeds. Full-data models had substantially better scores and received more optimizer steps at the same number of epochs.
- **Overfitting.** The random GRU's training BCE decreased from epoch 4 to 5 for every seed. Dev BCE rose over those epochs for seeds 3 and 5. The 0.35-dropout PPMI GRU averaged 0.282 dev macro-F1 versus 0.316 for 0.10 dropout, so the higher setting did not improve this pilot.
- **Errors.** BPE micro-F1 was about 0.43 on comments with a negation cue versus 0.51 without one. The GRU scored about 0.42 versus 0.51 on the same slices. Both systems also scored lower on comments with 20 or more tokens. The OOV slice showed little micro-F1 change. These are descriptive comparisons with different label mixes within each slice. Per-label diagnostics and matched prediction IDs support a follow-up focused on negation, label confusions, and low-support emotions.

## Next experimental priorities

Run the controlled 50d Twitter-versus-Wiki GloVe comparison when the vector files are available. Add trainable mean pooling and random-initialized pooling with a matched head, then compare against the GRU under the same embedding update policy. Use the saved validation threshold sweeps and label-wise errors to decide whether weighting or per-label thresholds merit a separate validation experiment. For a stronger data-size comparison, match optimizer steps and repeat the subset draws. Review errors by both negation scope and gold label before attributing the slice gap to word order.


## Extended study

The 18 November extension corrects the BPE reference, completes pretrained GloVe comparisons, broadens the parameter sweeps, and adds repeated controls and presentation evidence. See [the extended report](extended_report.md) for the full decision record and the selected results.
