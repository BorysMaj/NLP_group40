import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,"NLP_group40/part2")
from extended_experiments import *
exp=Experiment()
path=OUT/"bpe_runs.csv"
rows=pd.read_csv(path).to_dict("records")
best=pd.DataFrame(rows).sort_values(["dev_macro_f1","dev_micro_f1"],ascending=False).iloc[0]
size=int(best.vocab)
for frequency in [3,5]:
    name=f"bpe_v{size}_mf{frequency}_c{best.C}"
    if any(row["id"]==name for row in rows):
        continue
    tok=Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer=Metaspace()
    tok.train_from_iterator(exp.frames["train"].text,trainer=BpeTrainer(
        vocab_size=size,min_frequency=frequency,special_tokens=["[UNK]"]))
    tok.save(str(OUT/f"{name}.json"))
    vectorizer=CountVectorizer(analyzer=lambda text:tok.encode(text).tokens,
                               vocabulary=tok.get_vocab(),binary=True)
    model=OneVsRestClassifier(LogisticRegression(solver="liblinear",C=float(best.C),
                                                max_iter=300,random_state=0))
    model.fit(vectorizer.transform(exp.frames["train"].text),exp.gold["train"])
    dev=model.predict_proba(vectorizer.transform(exp.frames["dev"].text))
    test=model.predict_proba(vectorizer.transform(exp.frames["test"].text))
    score,sweep=best_threshold(exp.gold["dev"],dev)
    np.savez_compressed(OUT/f"{name}.npz",dev=dev,test=test)
    sweep.to_csv(OUT/f"{name}_thresholds.csv",index=False)
    rows.append(dict(id=name,vocab=size,min_frequency=frequency,C=best.C,threshold=score.threshold,
                     actual_vocab=tok.get_vocab_size(),dev_macro_f1=score.macro_f1,dev_micro_f1=score.micro_f1))
    pd.DataFrame(rows).to_csv(path,index=False)
    print(name,score.macro_f1,flush=True)
