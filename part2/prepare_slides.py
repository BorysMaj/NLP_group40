"""Turn verified run tables into slide content and editable chart specifications."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results/extended"
D=json.loads((OUT/"presentation_data.json").read_text(encoding="utf-8"))
runs=pd.DataFrame(D["runs"])
bpe=pd.DataFrame(D["bpe"])
selected=D["selected"]
slides=[]
def f(value): return f"{value:.3f}"
def p(value): return f"{100*value:.1f}%"
def add(title,subtitle="",body=None,charts=None,table=None,note="",takeaway=""):
    slides.append(dict(title=title,subtitle=subtitle,body=body,charts=charts,table=table,
                       notes=note,takeaway=takeaway))
def chart(categories,series,title="",kind="bar",minimum=0,maximum=.6,labels=True):
    return dict(categories=[str(x) for x in categories],series=series,title=title,type=kind,
                minimum=minimum,maximum=maximum,labels=labels)
def series(name,values): return dict(name=name,values=[round(float(x),6) for x in values])
def stage(name):
    return runs[runs.id.isin(D["stages"][name])].copy()
def sweep(name,column,title):
    rows=stage(name).sort_values(column)
    return chart(rows[column], [series("Dev macro-F1",rows.dev_macro_f1)],title)
def config_string(row):
    if "vocab" in row:
        return f"Vocabulary {int(row['vocab']):,}; C {row['C']:g}; minimum frequency {int(row['min_frequency'])}"
    source=row["source"].replace("twitter","Twitter ").replace("wiki","Wiki ").replace("random","Random ")
    head=(f"GRU {row['hidden']}" if row["encoder"]=="gru" else
          (f"MLP {row['head']}" if row["head"] else "linear head"))
    train="trainable" if row["trainable"] else "frozen"
    weight="; weighted loss" if row["weight"].startswith("sqrt") else ""
    return f"{source}; {train}; {head}; LR {row['lr']:g}; dropout {row['dropout']:g}{weight}"
counts=pd.DataFrame(D["dataset"])
final=pd.DataFrame(D["final"]).set_index("system")
add("GoEmotions Project 2","Word embeddings, pooling and recurrent encoders",
    body=[f"{len(runs)} neural fits and {len(bpe)} BPE fits in the extended search",
          "English Reddit comments with 27 emotions and neutral",
          "Group 40"],note="Run sources are retained in results/extended. Dataset source https://aclanthology.org/2020.acl-main.372/ . Counts exclude the earlier pilot's 27 neural fits.")
add("Evidence for the six project requirements",
    table=[["Requirement","Evidence"],
           ["Embedding choice","Twitter and Wiki GloVe; coverage, tokenization and dimension comparisons"],
           ["What vectors capture","Nearest neighbors and related versus contrasting emotion probes"],
           ["Embed and pool","Learning rate, head size, dropout, freezing and random controls"],
           ["Recurrent encoder","GRU width, optimization, embedding updates and labeled data size"],
           ["Uncertainty","Three seeds on finalists and controls; paired comment bootstrap"],
           ["Error analysis","Per-label FP/FN, matched text slices, logistic intervals and examples"]],
    note="Sources search_plan.json, runs.csv, bpe_runs.csv, neighbors.csv, bootstrap_intervals.csv, error_logistic_intervals.csv.")
add("Dataset and evaluation targets","Each comment can receive more than one label",
    table=[["Split","Comments","Multiple labels","Mean labels"],
           *[[r["split"],f"{r['examples']:,}",p(r["multilabel_percent"]/100),f"{r['mean_labels']:.2f}"] for r in D["dataset"]]],
    body=["Train support ranges from 77 grief comments to 14,219 neutral comments.",
          "Macro-F1 weights each emotion equally. Micro-F1 pools all label decisions."],
    note="Sources dataset_audit.csv and label_support.csv. The task retains all 28 labels and permits an empty predicted label set.")
add("Model selection and evaluation","Official train and dev splits guide the search",
    body=["Fit tokenizers and supervised parameters on 43,410 training comments.",
          "Select neural checkpoints and a global threshold by dev macro-F1. Break ties with dev micro-F1.",
          "Screen configurations with seed 0. Repeat finalists and controlled comparisons with seeds 0, 3 and 5.",
          "Report the best tested settings. The search is finite and staged.",
          "The earlier pilot already exposed this test set. Further test results are exploratory."],
    note="Sources search_plan.json, stages.json and selection.json. Threshold grid 0.05 to 0.50, increment 0.025. Every neural fit receives 10 epochs. The test set has 5,427 comments. Corpus vectors in the pilot are separate from the GloVe extension.")
hist=pd.DataFrame(D["historical_bpe"]).query("min_frequency==2").sort_values("vocab_size")
add("The Part 1 sweep favored 1,500 BPE tokens","Historical sweep under its original evaluation protocol",
    charts=[chart(hist.vocab_size,[series("Macro-F1",hist.macro_f1),series("Micro-F1",hist.micro_f1)],maximum=.6)],
    takeaway="Macro-F1 increased from 0.362 at 1,000 tokens to 0.385 at 1,500.",
    note="Source ../../goemotions_results/bpe_sweep/results.csv. These are saved Part 1 sweep scores, distinct from the new official dev scores. Minimum-frequency values 2, 3 and 4 had the same scores at vocabulary 1,000.")
best=bpe.sort_values(["dev_macro_f1","dev_micro_f1"],ascending=False).iloc[0]
grid=bpe.query("min_frequency==2").pivot(index="vocab",columns="C",values="dev_macro_f1").sort_index()
add("BPE vocabulary and regularization sweep","Refit the Part 1 representation on the official training split",
    charts=[chart([f"{x/1000:g}k" for x in grid.index],
                  [series(f"C {c:g}",grid[c]) for c in grid.columns],kind="line",labels=False,maximum=.55)],
    takeaway=f"Best tested BPE setting used {int(best.vocab):,} tokens and C {best.C:g}; dev macro-F1 {f(best.dev_macro_f1)}.",
    note="Source bpe_runs.csv. Metaspace BPE, minimum frequency 2, binary features, one-vs-rest liblinear logistic regression, max_iter 300. Each configuration receives its own dev-selected threshold. Larger C weakens regularization.")
vocab_uncertainty=D["vocab_bootstrap"]
add("The 10,000 versus 20,000 BPE margin",
    "Paired bootstrap of dev comments, with fitted models and thresholds fixed",
    table=[["Comparison","Macro-F1 difference","95% bootstrap interval","Within ±0.005"],
           ["20k minus 10k",f(vocab_uncertainty["point"]),
            f"[{f(vocab_uncertainty['lower'])}, {f(vocab_uncertainty['upper'])}]",
            p(vocab_uncertainty["within_rope"])]],
    body=["The highest dev score selects the 20,000-token fit within this grid.",
          "The small margin and its interval limit how strongly we can favor that vocabulary size."],
    note="Source bpe_vocabulary_bootstrap.csv. Select each vocabulary's C by dev macro-F1, retain those two fitted systems and thresholds, then jointly bootstrap their predictions over 1,000 dev-comment resamples. This conditional analysis uses the same data that guided selection and omits search uncertainty. It does not prove a global or unique vocabulary optimum.")
add("Pretrained embedding candidates",
    table=[["Property","GloVe Twitter","GloVe Wiki + Gigaword"],
           ["Training text","2 billion tweets","Wikipedia 2014 and Gigaword 5"],
           ["Training tokens","27 billion","6 billion"],
           ["Published vocabulary","About 1.2 million","400,000"],
           ["Dimensions tested","25, 50, 100","50"],
           ["Reason to compare","Informal English usage","Different text domain at matched width"]],
    takeaway="A shared 50-dimensional model setup supports the direct source comparison.",
    note="Sources https://nlp.stanford.edu/projects/glove/ and https://github.com/piskvorky/gensim-data . Files glove-twitter-25/50/100 and glove-wiki-gigaword-50. These use the original 2014 Wiki/Gigaword release. Static vectors give each token one representation.")
coverage=pd.DataFrame(D["coverage"]).query("split=='dev'")
cov=coverage[(coverage.source.isin(["twitter50","wiki50"])) | (coverage.source.eq("twitter100"))].drop_duplicates(["source","tokenizer"])
add("Tokenizer compatibility and vocabulary coverage","Lowercasing matches the uncased embedding tables",
    table=[["Embedding","Tokenizer","Token coverage","Comments with OOV"],
           *[[r.source.replace("twitter","Twitter ").replace("wiki","Wiki "),r.tokenizer,p(r.token_coverage),p(r.comments_with_oov)] for r in cov.itertuples()]],
    body=["TweetTokenizer preserves punctuation and expressive spellings. Whitespace keeps attached punctuation.",
          "Uncovered tokens start at the mean pretrained training-word vector. Padding uses a masked zero row."],
    note="Source coverage.csv. Coverage precedes truncation and uses exact pretrained token lookup. At most 40 token positions feed the encoders. Held-out strings receive label-free IDs for pretrained lookup, without contributing frequencies or training statistics.")
emb=stage("embedding").sort_values(["source"])
add("Embedding source and dimension screening","Frozen mean pooling with a 64-unit MLP and seed 0",
    charts=[chart([r.source.replace("twitter","Twitter ").replace("wiki","Wiki ") for r in emb.itertuples()],
                  [series("Dev macro-F1",emb.dev_macro_f1)],maximum=.4)],
    takeaway="Source and dimension are system choices. The 50-dimensional repeated controls compare sources at matched width.",
    note="Source stages.json embedding and runs.csv. Dropout 0.2, learning rate 0.003, unit-normalized vectors, TweetTokenizer and 10 epochs. The 25/50/100 Twitter files have the same published vocabulary.")
pre=stage("preprocessing")
add("Vector normalization and tokenization","Same selected source and pooling architecture",
    charts=[chart([f"{r.norm} / {r.tokenizer}" for r in pre.itertuples()],
                  [series("Dev macro-F1",pre.dev_macro_f1)],maximum=.4)],
    takeaway="The following parameter sweeps carry forward the strongest preprocessing setting.",
    note="Source stage preprocessing. Unit normalization changes vector scale before initialization. Raw preserves pretrained norms. Each score is one screening fit.")
probes=pd.DataFrame(D["probes"]).query("source=='twitter100' and tokenizer=='tweet'")
probe_pairs={("joy","happy"),("joy","sad"),("sad","miserable"),("sad","happy"),("love","adore"),("love","hate")}
probe_rows=probes[[ (row["query"],row["candidate"]) in probe_pairs for _,row in probes.iterrows() ]]
alignment=stage("contraction_alignment")
alignment_rows=[]
for _,row in alignment.iterrows():
    match=runs[runs.tokenizer.eq("tweet")]
    for column in ["source","encoder","trainable","head","hidden","lr","dropout","norm","seed","weight"]:
        match=match[match[column].eq(row[column])]
    if not match.empty:
        old=match.iloc[0]
        alignment_rows.append([f"{row.encoder.upper()} / {'tuned' if row.trainable else 'frozen'}",
                               f(old.dev_macro_f1),f(row.dev_macro_f1),
                               f(row.dev_macro_f1-old.dev_macro_f1)])
add("Contractions exposed a tokenization mismatch",
    "Examples such as I'm and don't missed exact pretrained lookup",
    table=[["Same model setup","TweetTokenizer","Split contractions","Change"],*alignment_rows],
    body=["The aligned variant splits contractions into Treebank token pieces after TweetTokenizer.",
          "Matched comparisons keep the embedding source, architecture and seed fixed."],
    note="Sources uncovered_tokens.csv and stage contraction_alignment. Whole-token OOV counts on dev include I'm 258 and don't 169. After lowercasing, Treebank splits these into i / 'm and do / n't. This is a compatibility check against the existing vocabulary, rather than a claim to reproduce every historical GloVe preprocessing rule.")
add("Emotion similarities include contrasting words","Selected cosine probes in Twitter GloVe",
    charts=[chart([f"{r.query} / {r.candidate}" for r in probe_rows.itertuples()],
                  [series("Cosine",probe_rows.cosine)],maximum=1,labels=False)],
    takeaway="Static co-occurrence vectors can place opposite sentiments near one another when their contexts overlap.",
    note="Source emotion_probes.csv. Related and contrasting lists were defined before the extension. Cosine probes describe these examples and do not constitute a general emotion benchmark.")
neighbors=pd.DataFrame(D["neighbors"]).query("source=='twitter100' and tokenizer=='tweet' and rank<=3")
add("Nearest neighbors reveal strengths and limits",
    table=[["Query","Three nearest candidates in the dataset vocabulary"],
           *[[query,", ".join(group.sort_values("rank").neighbor)] for query,group in neighbors.groupby("query")]],
    body=["Candidates come from words seen in the training comments.",
          "One vector per word conflates senses and omits sentence-level negation and sarcasm."],
    note="Source neighbors.csv. Searches exclude the query itself and rank by cosine. Only these candidate words were searched, rather than all 1.2 million published entries.")
add("Pooling parameter sweeps","Sequential comparisons carry forward the best earlier setting",
    charts=[sweep("pool_lr","lr","Learning rate"),sweep("pool_head","head","Head width; 0 is linear"),
            sweep("pool_dropout","dropout","Dropout")],
    takeaway="Each bar uses seed 0 and a validation-selected checkpoint and threshold.",
    note="Sources stages pool_lr, pool_head and pool_dropout. All remaining settings are fixed within a panel. Screens are conditional on the previous selected setting, so they cannot establish a global joint optimum.")
train=stage("pool_trainable").sort_values("trainable")
add("Fine-tuning the selected pooling model",
    charts=[chart(["Trainable" if value else "Frozen" for value in train.trainable],
                  [series("Dev macro-F1",train.dev_macro_f1)],maximum=.6)],
    takeaway="Updating embeddings lets the word vectors adapt to the emotion labels.",
    note="Source stage pool_trainable. Identical initial pretrained table, tokenizer, head, optimizer, and seed. This is the screening comparison. Repeated controls follow.")
add("GRU parameter sweeps","Final valid hidden state with a linear multilabel head",
    charts=[sweep("gru_lr","lr","Learning rate"),sweep("gru_hidden","hidden","GRU hidden width"),
            sweep("gru_dropout","dropout","Head dropout")],
    takeaway="The recurrence consumes the current word vector and the state built from preceding tokens.",
    note="Source stages gru_lr, gru_hidden and gru_dropout. One GRU layer, packed sequences and gradient clipping at norm 5. Dropout affects the classification head. Each screening fit receives 10 epochs.")
rep=pd.DataFrame(D["repeated"])
paired=runs[runs.id.isin(D["stages"]["paired_controls"])]
charts=[]
for enc in ["pool","gru"]:
    rows=paired.query("encoder==@enc").groupby(["source","trainable"]).dev_macro_f1.mean()
    cats=[f"{source.replace('twitter','Twitter ').replace('random','Random ').replace('wiki','Wiki ')}\n{'tuned' if trained else 'frozen'}" for source,trained in rows.index]
    charts.append(chart(cats,[series("Mean dev macro-F1",rows.values)],enc.upper(),maximum=.6))
add("Repeated initialization and fine-tuning controls","Shared 50-dimensional inputs and architecture settings within each panel",
    charts=charts,takeaway="Each bar averages seeds 0, 3 and 5. Full settings and per-seed scores appear in the appendix.",
    note="Source stage paired_controls. Random and pretrained tables have the same size and output dimension. Frozen versus tuned Twitter isolates embedding updates. Trainable Twitter versus trainable random tests pretrained initialization. Pool versus GRU also changes recurrent parameter count.")
scale_rows=runs[runs.id.isin(D["stages"]["random_scale_control"])]
scale_charts=[]
for encoder in ["pool","gru"]:
    reference=paired[(paired.encoder==encoder)&paired.trainable&
                     paired.source.isin(["twitter50","random50"])]
    matched=scale_rows[scale_rows.encoder==encoder]
    combined=pd.concat([reference,matched])
    scores=combined.groupby("source").dev_macro_f1.mean().reindex(["twitter50","random50","randomscaled50"])
    scale_charts.append(chart(["Twitter 50","Random","Random\nmatched norms"],
                              [series("Mean dev macro-F1",scores)],encoder.upper(),maximum=.6))
add("Pretrained meaning and initialization scale",
    "Random directions with the same per-word initial vector lengths",
    charts=scale_charts,
    takeaway="The matched-norm control holds vector magnitude fixed while removing pretrained directions.",
    note="Source stage random_scale_control and paired_controls. Each randomscaled50 row starts with a random Gaussian direction, normalized and scaled to the original pretrained row norm. Architecture, token IDs, batch order, optimizer, trainability and seeds match the trainable Twitter control. The plain random control has expected norm near one and therefore changes initial scale as well as direction.")
data=runs[runs.id.isin(D["stages"]["data_size"])]
means=data.groupby(["source","fraction"]).dev_macro_f1.mean().unstack(0).sort_index()
add("Labeled data size with matched training updates","25%, 50% and 100% nested subsets for each seed",
    charts=[chart([p(x) for x in means.index],
                  [series(source.replace("twitter","Twitter ").replace("random","Random "),means[source]) for source in means.columns],
                  kind="line",labels=False,maximum=.6)],
    takeaway="Each run receives the same number of optimizer steps as its full-data counterpart.",
    note="Source stage data_size. Three seeds per cell, 10 matched-update epochs, fixed 50-dimensional architecture. Smaller subsets cycle to match full-data batch counts. Hyperparameters were selected on full data. The source embeddings remain pretrained on their original external corpora.")
histories=[]
for label in ["Pool","GRU"]:
    row=selected[label][0]
    frame=pd.read_csv(OUT/f"{row['id']}_epochs.csv")
    histories.append(chart(frame.epoch,[series("Training BCE",frame.train_bce),series("Dev BCE",frame.dev_bce)],
                           f"{label} seed {row['seed']}",kind="line",labels=False,maximum=.3))
add("Training curves and overfitting","Training and dev binary cross-entropy for selected configurations",
    charts=histories,takeaway="The checkpoint rule follows dev macro-F1. BCE curves reveal fit and generalization behavior.",
    note="Sources selected *_epochs.csv files. Train BCE is the average during an epoch while parameters change. For a weighted selected model, training BCE includes positive-class weights and dev BCE remains unweighted; compare trends within each curve rather than their absolute levels. Dev BCE is measured after the epoch with dropout disabled. A lower training loss accompanied by worsening dev loss supports an overfitting interpretation.")
add("Selected configurations",
    table=[["System","Configuration","Selected epochs","Thresholds"],
           *[[system,config_string(rows[0]),", ".join(str(int(r.get("epoch",0))) for r in rows),
              ", ".join(f"{r['threshold']:.3f}" for r in rows)] for system,rows in selected.items()]],
    body=["Neural models use BCE with logits, AdamW and 10 training epochs.",
          "Each seed retains its own checkpoint and global validation threshold."],
    note="Source selection.json. Embedding normalization, tokenizer, hidden width, parameter count, and all per-run settings appear in runs.csv. The paired controls use unweighted BCE. The selected model may use the separately repeated weighted-loss configuration. The separate weighting screen is retained in the appendix.")
devvalues=[bpe.sort_values("dev_macro_f1",ascending=False).iloc[0].dev_macro_f1]
for system in ["Pool","GRU"]:
    devvalues.append(np.mean([row["dev_macro_f1"] for row in selected[system]]))
add("Final system comparison","Configurations selected by validation performance",
    charts=[chart(["BPE","Pool","GRU"],[series("Dev macro-F1",devvalues),
                   series("Test macro-F1",[final.loc[x,"macro"] for x in ["BPE","Pool","GRU"]])],maximum=.65)],
    takeaway="Neural test scores average three independently trained seeds.",
    note="Sources selection.json, final_summary.csv and final_test_scores.csv. The earlier pilot exposed test results, so this extension is exploratory rather than a fresh confirmatory test. The BPE fit is deterministic under the fixed split and has one fit.")
add("Macro-F1 and micro-F1",
    table=[["System","Test macro-F1","Seed SD","Test micro-F1","Seed SD"],
           *[[system,f(final.loc[system,"macro"]),f(final.loc[system,"macro_sd"]) if pd.notna(final.loc[system,"macro_sd"]) else "One fit",
              f(final.loc[system,"micro"]),f(final.loc[system,"micro_sd"]) if pd.notna(final.loc[system,"micro_sd"]) else "One fit"] for system in ["BPE","Pool","GRU"]]],
    body=["Macro-F1 highlights performance across all emotions, including rare labels.",
          "Micro-F1 weights the pooled label decisions and can favor a different system."],
    note="Sources final_test_scores.csv and final_summary.csv. Seed SD uses three fits and a fixed split. It omits split uncertainty and the effect of the search.")
intervals=pd.DataFrame(D["bootstrap"])
add("Paired uncertainty and practical differences","1,000 bootstrap resamples of the same test comments",
    table=[["Difference in macro-F1","Mean","95% interval","Below −0.005","Within ±0.005","Above +0.005"],
           *[[r.comparison,f(r["mean"]),f"[{f(r.lower)}, {f(r.upper)}]",p(r.below_rope),p(r.within_rope),p(r.above_rope)]
             for _,r in intervals.iterrows()]],
    takeaway="These fractions describe the bootstrap distribution under the fitted models.",
    note="Source bootstrap_intervals.csv and paired_bootstrap.csv. Resample comments jointly across systems while retaining all labels, compute each seed's macro-F1, then average across seeds. Thresholds and checkpoints stay fixed. This measures conditional test-sample variation and omits full search and training uncertainty. ROPE ±0.005 is the Part 1 practical-margin choice. Bootstrap fractions are distinct from Bayesian posterior model probabilities.")
rope_chart=chart(intervals.comparison,[
    series("Below -0.005",intervals.below_rope),
    series("Within ±0.005",intervals.within_rope),
    series("Above +0.005",intervals.above_rope)],maximum=1)
rope_chart["stacked"]=True
rope_chart["percentage"]=True
add("Practical differences in test macro-F1",
    "Bootstrap fractions relative to the practical margin",
    charts=[rope_chart],
    takeaway="The practical margin is ±0.005 macro-F1. These fractions condition on the selected fitted models.",
    note="Source bootstrap_intervals.csv. The bars summarize 1,000 paired comment-bootstrap replicates. They are bootstrap fractions, rather than Bayesian probabilities that a model is globally optimal. All model-selection and training uncertainty limits from the preceding slide apply.")
slices=pd.DataFrame(D["slices"])
s=slices[slices.feature.isin(["negation","long_20plus","oov"])].groupby(["feature","present","system"]).micro_f1.mean()
cats=[("negation",False),("negation",True),("long_20plus",False),("long_20plus",True),("oov",False),("oov",True)]
names=["No negation","Negation","<20 tokens","20+ tokens","Covered","OOV present"]
add("Text characteristics and performance","Matched test comments with a shared feature definition",
    charts=[chart(names,[series(system,[s.loc[feature,present,system] for feature,present in cats]) for system in ["BPE","Pool","GRU"]],maximum=.7,labels=False)],
    takeaway="The groups have different label mixes. Slice gaps describe associations.",
    note="Source error_slices.csv. OOV is defined by the selected GRU's pretrained vocabulary. Negation uses surface tokens such as not, no, never, cannot and n't. It misses scope and implicit negation. Each system sees the same comment IDs within a group.")
coef=pd.DataFrame(D["coefficients"])
add("Predictors of an exactly correct label set","Bayesian logistic regression with a Laplace approximation",
    table=[["Text feature","BPE log-odds [95%]","Pool log-odds [95%]","GRU log-odds [95%]"],
           *[[{"log_length_z":"Log length (standardized)","oov_rate_z":"OOV rate (standardized)","negation":"Negation cue","punctuation":"Expressive punctuation"}[feature],*[f"{f((r:=coef[(coef.system==system)&(coef.feature==feature)].iloc[0])['mean'])} [{f(r.lower)}, {f(r.upper)}]" for system in ["BPE","Pool","GRU"]]]
              for feature in ["log_length_z","oov_rate_z","negation","punctuation"]]],
    body=["A negative coefficient lowers modeled correctness when the other features stay fixed.",
          "An interval crossing zero leaves the direction uncertain at this interval level."],
    note="Source error_logistic_intervals.csv. Fit one regression per system using one seed, avoiding repeated rows for seed variants. Response is exact agreement on all 28 labels. Independent Normal(0,2.5) priors, MAP optimization and inverse-Hessian Gaussian intervals. This uses a Laplace approximation, with no MCMC chain or R-hat diagnostics. Length is log1p transformed and standardized, OOV rate standardized. Features describe text and are observational.")
ex=pd.DataFrame(D["examples"])
chosen_examples=pd.concat([ex[ex.direction==direction].head(1) for direction in ["GRU improves","BPE improves"]])
add("Examples of disagreement","Same comments, different model decisions",
    table=[["","Example A","Example B"],
           ["Comment",*chosen_examples.text.tolist()],
           ["Gold",*chosen_examples.gold.tolist()],
           ["BPE",*chosen_examples.BPE.tolist()],
           ["Pool",*chosen_examples.Pool.tolist()],
           ["GRU",*chosen_examples.GRU.tolist()]],
    note="Source examples.csv. Examples are selected after evaluation for large comment-level F1 differences and readable length. Predictions shown use each family's first seed. The example selection illustrates behavior and does not estimate its frequency. Comment IDs "+", ".join(chosen_examples.id))
winner=final["macro"].idxmax()
add("Conclusions supported by the experiments",
    body=[f"{winner} had the highest mean test macro-F1 among the selected systems, at {f(final.loc[winner,'macro'])}.",
          f"The extended BPE search selected {int(best.vocab):,} tokens with C {best.C:g}; the older 1,000-token baseline understates BPE performance.",
          "Matched controls identify the effect of freezing, initialization and recurrence under the tested settings.",
          "The staged search establishes the strongest tested configurations. Untested combinations and further training can change the ranking."],
    note="Sources final_summary.csv, bpe_runs.csv, runs.csv and repeated_summary.csv. Use uncertainty intervals from the paired bootstrap slide when describing margins.")
add("Limitations and next experiments",
    body=["Check label confusions and low-support emotions before changing the loss or threshold policy.",
          "Repeat the most promising comparisons across fresh splits to assess split sensitivity.",
          "Test source-specific Twitter normalization, additional embeddings and per-label threshold regularization.",
          "The reused test set and single-seed screening limit confirmatory claims.",
          "Larger search ranges would be justified where the best score lies on a tested boundary."],
    note="These are priorities inferred from the search coverage and error analysis. The experiments cover a finite staged grid and do not prove a global optimum.")

# Appendix preserves the complete search grid and per-seed evidence.
add("Appendix A. Search ranges and controls",
    table=[["Choice","Tested values","Selection or control"],
           ["BPE vocabulary","500, 1k, 1.5k, 2k, 3k, 5k, 10k, 20k","Global dev threshold per fit"],
           ["BPE C","0.03, 0.1, 0.3, 1, 3","Grid across vocabulary sizes"],
           ["Twitter dimensions","25, 50, 100","Shared frozen pooling reference"],
           ["Pooling","LR .001/.003/.01; head 0/64/128; dropout 0/.2/.5","Sequential single-factor screens"],
           ["GRU","LR .001/.003/.01; hidden 32/64/128; dropout 0/.2/.5","Sequential single-factor screens"],
           ["Embedding status","Frozen pretrained, trainable pretrained, trainable random","Matched 50d controls, three seeds"],
           ["Labeled data","25%, 50%, 100%","Nested subsets and matched update counts"]],
    note="Sources search_plan.json, stages.json and runs.csv. Actual configurations are listed in these files. Word cap 40, batch size 128, AdamW weight_decay .01, gradient norm clipping 5, threshold .05 to .50 in .025 steps.")
add("Appendix B. Complete BPE grid","Validation macro-F1; all configurations have minimum frequency {int(row['min_frequency'])}",
    table=[["Vocabulary",*[f"C {value:g}" for value in grid.columns]],
           *[[f"{int(size):,}",*[f(value) for value in row]] for size,row in grid.iterrows()]],
    note="Source bpe_runs.csv. The grid is selected by macro-F1, with micro-F1 as tie-breaker. Actual tokenizers are retained as bpe_vSIZE.json.")
paired_table=paired.groupby(["encoder","source","trainable","seed"]).dev_macro_f1.mean().unstack("seed").reset_index()
frequency=bpe[(bpe.vocab==20000)&(bpe.C==1)].sort_values("min_frequency")
add("Appendix B2. BPE minimum frequency",
    "Fixed requested vocabulary 20,000 and C 1",
    table=[["Minimum frequency","Dev macro-F1","Dev micro-F1"],
           *[[str(int(r.min_frequency)),f(r.dev_macro_f1),f(r.dev_micro_f1)] for r in frequency.itertuples()]],
    note="Source bpe_runs.csv and bpe_v20000_mf*.json. Values 2 and 3 give the same validation scores; 5 gives a small decrease. This extends the earlier Part 1 check at vocabulary 1,000.")
add("Appendix C. Paired per-seed scores","Validation macro-F1 with the shared 50-dimensional architecture",
    table=[["Encoder","Initialization","Updates","Seed 0","Seed 3","Seed 5"],
           *[[r.encoder,r.source,"Trainable" if r.trainable else "Frozen",f(r[0]),f(r[3]),f(r[5])]
             for _,r in paired_table.iterrows()]],
    note="Source stage paired_controls. Repeated rows use the same fixed train/dev split and each seed controls initial weights, the data subset permutation, and batch order.")
thresholds=pd.DataFrame(D["thresholds"])
chartseries=[]
common_grid=sorted(set.intersection(*[set(thresholds[thresholds.system==system].threshold) for system in ["BPE","Pool","GRU"]]))
for system in ["BPE","Pool","GRU"]:
    table=thresholds[thresholds.system==system].groupby("threshold").macro_f1.mean().reindex(common_grid)
    chartseries.append(series(system,table))
add("Appendix D. Threshold sensitivity","Shared threshold range for the selected systems",
    charts=[chart([f"{x:.3f}" for x in common_grid],chartseries,kind="line",labels=False,maximum=.65)],
    takeaway="The same probability threshold need not give comparable precision and recall across systems.",
    note="Source final_threshold_curves.csv. Each curve uses validation predictions from the chosen checkpoints. The plotted range is shared across systems. Weighted threshold sweeps extend to 0.90 in their full CSV records. The final fit-specific threshold maximizes validation macro-F1, with micro-F1 breaking ties.")
per=pd.DataFrame(D["per_label"]).groupby(["label","system"]).agg(f1=("f1","mean"),support=("support","first")).reset_index()
pivot=per.pivot(index="label",columns="system",values="f1")
support=per.drop_duplicates("label").set_index("label").support
order=support.sort_values().index
for index,part in enumerate([order[:14],order[14:]],1):
    add(f"Appendix E{index}. Per-emotion performance","Test F1 averaged across the selected seeds",
        table=[["Emotion","Support","BPE","Pool","GRU"],
               *[[name,str(int(support[name])),*[f(pivot.loc[name,system]) for system in ["BPE","Pool","GRU"]]] for name in part]],
        note="Source per_label.csv. Full per-seed false-positive and false-negative counts are retained. Test support counts comments carrying that label.")
weighted=stage("weighting")
add("Appendix F. Loss weighting screen",
    charts=[chart(weighted.weight,[series("Dev macro-F1",weighted.dev_macro_f1)],maximum=.6)],
    body=["The weighted loss uses the square root of the negative-to-positive training ratio, capped at 10.",
          "Weighting shifts the sigmoid scores, so every fit receives a fresh dev threshold sweep."],
    note="Source stage weighting. The paired controls retain unweighted BCE. The initial weighting screen has one seed. The weighted_finalists extension expands the threshold range to 0.90 and repeats the weighted GRU across three seeds; it participates in final system selection.")
add("Appendix G. Reproducibility and scope",
    body=["extended_experiments.py retains the configurations, checkpoints, predictions and hashes.",
          "analyze_extended.py retains the selected-model scores, paired bootstrap and error analysis.",
          "The project notebook displays the tables and scientific plots.",
          "The earlier pilot's 27 neural fits used corpus PPMI vectors. These extension results use external GloVe.",
          "Exact text overlap occurs across splits. A sensitivity table excludes 42 affected test comments."],
    note="Sources search_plan.json, complete.json, selected_predictions.csv, test_without_overlap.csv and project.ipynb. External sources https://nlp.stanford.edu/projects/glove/ ; https://aclanthology.org/2020.acl-main.372/ . The bootstrap and approximate Bayesian logistic analyses have distinct uncertainty interpretations.")
(OUT/"deck_build").mkdir(exist_ok=True)
(OUT/"deck_build/slide_specs.json").write_text(json.dumps(slides,indent=2,default=str),encoding="utf-8")
print("Slides",len(slides))
