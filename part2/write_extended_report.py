"""Append the extended study's evidence and decisions to the experiment record."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parent
out=root/"results/extended"
d=json.loads((out/"presentation_data.json").read_text())
runs=pd.DataFrame(d["runs"])
bpe=pd.DataFrame(d["bpe"])
final=pd.DataFrame(d["final"]).set_index("system")
lines=["# Extended GoEmotions study", "",
       "Study date 18 November 2026.", "",
       f"The extension completed {len(runs)} neural fits and {len(bpe)} BPE fits. The earlier pilot remains a separate study with 27 neural fits and one BPE fit.",
       "",
       "## Decisions and discoveries", "",
       "| Decision | Evidence and consequence |", "|---|---|",
       "| Restore the stronger Part 1 reference | The saved vocabulary sweep favored 1,500 over 1,000 tokens, with macro-F1 0.385 versus 0.362 under its original evaluation. The extension refits and retunes BPE on the official splits. |",
       "| Complete the external embedding comparison | Downloads became accessible. Compare original Twitter GloVe 25d, 50d and 100d with original Wiki/Gigaword 50d. Exact source hashes are retained in cached metadata. |",
       "| Expand the BPE boundary | The first 500–5,000 grid peaked at its largest vocabulary. Expand to 10,000 and 20,000 and C values 0.03, 0.1, 0.3, 1 and 3. |",
       "| Check minimum frequency | At requested vocabulary 20,000 and C 1, minimum frequencies 2 and 3 tie on dev, while 5 lowers macro-F1 by about 0.0022. |",
       "| Inspect uncovered tokens | Whole contractions dominate several frequent lookup misses. Dev counts include I'm 258, it's 248 and don't 169 after lowercasing. |",
       "| Compare a compatible tokenizer | Add Treebank contraction splitting after TweetTokenizer. Keep all model settings fixed for direct comparisons, then repeat the trainable pool and GRU variants across three seeds. This implements a measured vocabulary alignment and retains remaining source-specific normalization limitations. |",
       "| Sweep architectures in stages | For pooling, vary learning rate, MLP head width and dropout sequentially, then freezing. For the GRU, vary learning rate, hidden width and head dropout. A later sweep carries forward earlier selected settings. |",
       "| Choose the recurrent encoder | Use one GRU layer as the recurrent comparison within the local training budget. Sweep hidden width and training settings before comparing the selected system with pooling. |",
       "| Preserve attribution controls | Repeat frozen Twitter, trainable Twitter, trainable random and trainable Wiki at 50 dimensions with shared architecture settings. Random and pretrained runs have identical embedding matrix shapes. |",
       "| Check random-initialization scale | The raw pretrained vectors have different norms from ordinary unit-scale Gaussian starts. Add a three-seed random-direction control that preserves every pretrained row norm. This isolates direction information under the matched initialization scale. |",
       "| Match data-size optimizer budgets | Use nested 25%, 50% and 100% labeled subsets within each seed and cycle smaller subsets to receive the full-data number of update steps. Report that hyperparameters were chosen with full data. |",
       "| Recheck the weighted-loss threshold boundary | The initial weighted model favored threshold 0.50, the highest screening value. Expand the weighted threshold grid to 0.90 and repeat three seeds before selection. |",
       "| Retain uncertainty limits | Three seeds capture initialization and batch order on fixed splits. A paired comment bootstrap quantifies conditional test-sample variation. The pilot already exposed the test set, so the extension remains exploratory. |",
       "",
       "## Selected systems", "",
       "| System | Dev macro-F1 | Test macro-F1 | Test micro-F1 |",
       "|---|---:|---:|---:|"]
for system,rows in d["selected"].items():
    dev=np.mean([r["dev_macro_f1"] for r in rows])
    lines.append(f"| {system} | {dev:.4f} | {final.loc[system,'macro']:.4f} | {final.loc[system,'micro']:.4f} |")
lines+=["","Exact configurations, checkpoint epochs and thresholds are in selection.json. The presentation uses native editable charts and tables generated from the same numeric tables.",
        "","## Paired uncertainty","","| Comparison | Mean macro-F1 difference | 95% bootstrap interval |",
        "|---|---:|---|"]
for row in d["bootstrap"]:
    lines.append(f"| {row['comparison']} | {row['mean']:.4f} | [{row['lower']:.4f}, {row['upper']:.4f}] |")
lines+=["","Bootstrap resampling preserves each comment's labels and uses the same resampled IDs for all systems. Each replicate averages F1 across the available seed fits. The intervals condition on the already trained, selected models and fixed thresholds. They omit split, search and retraining uncertainty. The practical margin is ±0.005 macro-F1, carried over from Part 1.",
        "","## Requirement audit","","| Requirement | Completion and evidence |",
        "|---|---|",
        "| Appropriate word embeddings | English social and edited-text GloVe comparisons, dimensions, coverage, tokenizer controls and uncovered-token audit. See coverage.csv, uncovered_tokens.csv and stages.json. |",
        "| What embeddings capture | Neighbors and explicit emotion-word probes, including related and contrasting words. See neighbors.csv and emotion_probes.csv. Candidate neighbors are restricted to the training vocabulary. |",
        "| Embed and pool | Head, optimization, dropout and embedding-update sweeps, plus matched random and pretrained controls. See runs.csv and sweep_pool_*.png. |",
        "| Recurrent encoder | GRU width, learning rate, dropout, embedding-update and initialization comparisons, plus matched-update data-size study and epoch curves. |",
        "| Uncertainty and search | Planned staged ranges, all run settings, three-seed controls and finalists, and 1,000 paired test-comment bootstrap samples. See search_plan.json, repeated_summary.csv and paired_bootstrap.csv. |",
        "| Error analysis | All 28 per-label F1/FP/FN counts, matched negation/length/OOV/punctuation slices, approximate Bayesian logistic intervals for exact label-set correctness, and readable disagreements. |",
        "","## Error-analysis interpretation","",
        "The logistic analysis fits one model per classifier family using a single seed. Its target is exact correctness of the entire 28-label set. Independent Normal(0,2.5) coefficient priors and an inverse-Hessian Gaussian approximation give approximate posterior intervals. This is a Laplace approximation. It does not use MCMC, and it has no R-hat or chain diagnostics.",
        "",
        "Text features are standardized log length, standardized OOV rate, a negation cue and expressive punctuation. Negative coefficients lower modeled log-odds of exact correctness with other features fixed. An interval containing both signs leaves direction uncertain at that interval level. Raw slices have different label mixes and support sizes.",
        "",
        "The exact-text overlap sensitivity check removes 42 test comments that match train or dev text. See test_without_overlap.csv. Each split has unique IDs. No test labels guide fitting or configuration selection in this extension, while the earlier pilot's published test results still limit confirmatory claims.",
        "",
        "## Meaning of the selected optimum","",
        "The results establish the best tested configurations under a finite staged search. The BPE vocabulary/C grid is joint. Neural sweeps change one factor at a time and depend on earlier choices. Untested interactions, source-specific normalization, additional embeddings, alternative losses, fresh splits and longer training remain possible improvements. The largest requested BPE vocabulary still has a narrow dev advantage, so the search does not establish a global vocabulary optimum.",
        "",
        "## Reproduction","",
        "Run the following from the group repository root in this order. Each completed neural configuration is cached by its parameter identity. Keep run directories to preserve the fitting history.",
        "",
        "```powershell",
        "python part2/extended_experiments.py",
        "python part2/run_bpe_minfreq.py",
        "python part2/run_weighted_extension.py",
        "python part2/run_alignment_extension.py",
        "python part2/run_scale_control.py",
        "python part2/analyze_extended.py",
        "python part2/prepare_slides.py",
        "```",
        "",
        "The external gzip vector files must be present in part2/results/embeddings as twitter25.gz, twitter50.gz, twitter100.gz and wiki50.gz. Source releases are listed on https://nlp.stanford.edu/projects/glove/ and https://github.com/piskvorky/gensim-data . The dataset paper is https://aclanthology.org/2020.acl-main.372/ .",
        "",
        "All neural configurations use batch size 128, AdamW weight decay 0.01, gradient norm clipping at 5, a 40-token cap, and 10 epochs. The original threshold grid is 0.05–0.50 in 0.025 increments. The weighted extension uses 0.05–0.90. Model and tokenization changes are recorded by unique configuration IDs. Fixed pretrained lookup can assign IDs to held-out strings without using held-out labels, frequencies or co-occurrences."]
(root/"extended_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
old=root/"experiment_log.md"
content=old.read_text(encoding="utf-8-sig")
if "extended_report.md" not in content:
    content+="\n\n## Extended study\n\nThe 18 November extension corrects the BPE reference, completes pretrained GloVe comparisons, broadens the parameter sweeps, and adds repeated controls and presentation evidence. See [the extended report](extended_report.md) for the full decision record and the selected results.\n"
    old.write_text(content,encoding="utf-8")
print("Wrote extended_report.md")
