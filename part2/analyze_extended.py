"""Summarize the extended search, quantify variation, and export presentation data."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"part1"))
from goemotions_pipeline import load_splits,locate_data,multilabel_metrics

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results/extended"
frames,gold,labels=load_splits(locate_data())
runs=pd.read_csv(OUT/"runs.csv")
stages=json.loads((OUT/"stages.json").read_text())
bpe=pd.read_csv(OUT/"bpe_runs.csv")
bpe_best=bpe.sort_values(["dev_macro_f1","dev_micro_f1"],ascending=False).iloc[0]
group_cols=["source","encoder","trainable","hidden","head","dropout","lr","norm",
            "tokenizer","fraction","epochs","budget","weight"]
repeated=runs.groupby(group_cols,dropna=False).agg(
    n=("seed","nunique"),macro=("dev_macro_f1","mean"),sd=("dev_macro_f1","std"),
    micro=("dev_micro_f1","mean")).reset_index()
repeated.to_csv(OUT/"repeated_summary.csv",index=False)
selected={"BPE": [{key:value for key,value in bpe_best.to_dict().items() if pd.notna(value)}]}
for encoder,label in [("pool","Pool"),("gru","GRU")]:
    candidates=repeated.query("encoder==@encoder and n==3 and fraction==1 and budget=='epochs'")
    best=candidates.sort_values(["macro","micro"],ascending=False).iloc[0]
    mask=np.ones(len(runs),bool)
    for col in group_cols:
        mask &= runs[col].eq(best[col])
    selected[label]=runs[mask].to_dict("records")
# Lock the chosen configurations before computing test summary.
(OUT/"selection.json").write_text(json.dumps(selected,indent=2))
predictions={}
probabilities={}
scores=[]
threshold_curves=[]
for label,rows in selected.items():
    predictions[label]=[]
    probabilities[label]=[]
    for row in rows:
        data=np.load(OUT/f"{row['id']}.npz")
        pred=data["test"]>=row["threshold"]
        predictions[label].append(pred)
        probabilities[label].append(data["test"])
        scores.append(dict(system=label,id=row["id"],seed=row.get("seed",0),
                           **multilabel_metrics(gold["test"],data["test"],row["threshold"])))
        sweep=pd.read_csv(OUT/f"{row['id']}_thresholds.csv")
        threshold_curves.append(sweep.assign(system=label,id=row["id"]))
scores=pd.DataFrame(scores)
scores.to_csv(OUT/"final_test_scores.csv",index=False)
pd.concat(threshold_curves).to_csv(OUT/"final_threshold_curves.csv",index=False)
final=scores.groupby("system",as_index=False).agg(macro=("macro_f1","mean"),
    macro_sd=("macro_f1","std"),micro=("micro_f1","mean"),micro_sd=("micro_f1","std"))
final.to_csv(OUT/"final_summary.csv",index=False)
seen=set(frames["train"].text)|set(frames["dev"].text)
clean=~frames["test"].text.isin(seen).to_numpy()
sensitivity=[]
for label,preds in predictions.items():
    for i,pred in enumerate(preds):
        sensitivity.append(dict(system=label,seed=i,n=clean.sum(),
            **multilabel_metrics(gold["test"][clean],pred[clean],.5)))
pd.DataFrame(sensitivity).to_csv(OUT/"test_without_overlap.csv",index=False)
# Paired bootstrap over comments, preserving all label decisions and seed fits.
rng=np.random.default_rng(40)
def macro(y,p):
    tp=(y&p).sum(0)
    denominator=y.sum(0)+p.sum(0)
    return np.divide(2*tp,denominator,out=np.zeros(28),where=denominator>0).mean()
truth=gold["test"].astype(bool)
boot=[]
for iteration in range(1000):
    idx=rng.integers(0,len(truth),len(truth))
    values={system:np.mean([macro(truth[idx],p[idx]) for p in preds])
            for system,preds in predictions.items()}
    for left,right in [("GRU","BPE"),("Pool","BPE"),("GRU","Pool")]:
        boot.append(dict(iteration=iteration,comparison=f"{left} minus {right}",
                         delta=values[left]-values[right]))
boot=pd.DataFrame(boot)
boot.to_csv(OUT/"paired_bootstrap.csv",index=False)
intervals=boot.groupby("comparison").delta.agg(
    mean="mean",lower=lambda x:x.quantile(.025),upper=lambda x:x.quantile(.975),
    below_rope=lambda x:(x<-.005).mean(),within_rope=lambda x:(x.abs()<=.005).mean(),
    above_rope=lambda x:(x>.005).mean()).reset_index()
intervals.to_csv(OUT/"bootstrap_intervals.csv",index=False)

# Features based only on text and pretrained lookup.
grurow=selected["GRU"][0]
cache=torch_cache=None
import torch
cache=torch.load(OUT/f"cache_{grurow['source'].replace('randomscaled','twitter').replace('random','twitter')}_{grurow['tokenizer']}.pt",
                 weights_only=False)
features=cache[2]["test"].reset_index(drop=True)
feature_names=["log_length_z","oov_rate_z","negation","punctuation"]
length=np.log1p(features.length.to_numpy())
oov=features.oov_rate.to_numpy()
X=np.column_stack([np.ones(len(length)),(length-length.mean())/length.std(),
                  (oov-oov.mean())/max(oov.std(),1e-8),
                  features.negation.astype(float),features.punctuation.astype(float)])
coefficients=[]
slice_rows=[]
label_rows=[]
example_rows=[]
for system,preds in predictions.items():
    # One model per family prevents treating repeated-seed predictions as independent observations.
    pred=preds[0]
    correct=(pred==truth).all(1).astype(float)
    prior_precision=1/2.5**2
    def objective(beta):
        eta=X@beta
        loss=(np.logaddexp(0,eta)-correct*eta).sum()+.5*prior_precision*(beta@beta)
        gradient=X.T@(expit(eta)-correct)+prior_precision*beta
        return loss,gradient
    optimum=minimize(objective,np.zeros(X.shape[1]),jac=True,method="BFGS",
                     options={"gtol":1e-6})
    beta=optimum.x
    p=expit(X@beta)
    hessian=X.T@(X*(p*(1-p))[:,None])+prior_precision*np.eye(X.shape[1])
    covariance=np.linalg.inv(hessian)
    sd=np.sqrt(np.diag(covariance))
    for name,mean,error in zip(["intercept"]+feature_names,beta,sd):
        coefficients.append(dict(system=system,feature=name,mean=mean,
            lower=mean-1.96*error,upper=mean+1.96*error,gradient_max=np.abs(optimum.jac).max()))
    for i,pred in enumerate(preds):
        for name,mask in dict(negation=features.negation.to_numpy(),long_20plus=features.length.to_numpy()>=20,
                              oov=features.oov_rate.to_numpy()>0,punctuation=features.punctuation.to_numpy(),
                              multilabel=gold["test"].sum(1)>1).items():
            for present in [False,True]:
                keep=mask==present
                slice_rows.append(dict(system=system,seed=i,feature=name,present=present,n=keep.sum(),
                                       **multilabel_metrics(gold["test"][keep],pred[keep],.5)))
        for j,label in enumerate(labels):
            label_rows.append(dict(system=system,seed=i,label=label,support=truth[:,j].sum(),
                f1=f1_score(truth[:,j],pred[:,j],zero_division=0),
                fp=(pred[:,j]&~truth[:,j]).sum(),fn=(~pred[:,j]&truth[:,j]).sum()))
pd.DataFrame(coefficients).to_csv(OUT/"error_logistic_intervals.csv",index=False)
pd.DataFrame(slice_rows).to_csv(OUT/"error_slices.csv",index=False)
pd.DataFrame(label_rows).to_csv(OUT/"per_label.csv",index=False)
sample_f1={}
for system,preds in predictions.items():
    rows=[]
    for pred in preds:
        tp=(truth&pred).sum(1)
        rows.append(2*tp/np.maximum(1,truth.sum(1)+pred.sum(1)))
    sample_f1[system]=np.mean(rows,axis=0)
delta=sample_f1["GRU"]-sample_f1["BPE"]
first_scores={}
for system in ["GRU","BPE"]:
    pred=predictions[system][0]
    first_scores[system]=2*(truth&pred).sum(1)/np.maximum(1,truth.sum(1)+pred.sum(1))
first_delta=first_scores["GRU"]-first_scores["BPE"]
for direction,order in [("GRU improves",np.argsort(-delta)),("BPE improves",np.argsort(delta))]:
    kept=0
    for idx in order:
        if len(frames["test"].iloc[idx].text)>90 or abs(delta[idx])<.4:
            continue
        if direction=="GRU improves" and first_delta[idx]<=.1:
            continue
        if direction=="BPE improves" and first_delta[idx]>=-.1:
            continue
        row=dict(direction=direction,id=frames["test"].iloc[idx].id,
                 text=frames["test"].iloc[idx].text,delta=delta[idx],
                 gold=", ".join(np.array(labels)[truth[idx]]))
        for system,preds in predictions.items():
            row[system]=", ".join(np.array(labels)[preds[0][idx]])
        example_rows.append(row)
        kept+=1
        if kept==4:
            break
pd.DataFrame(example_rows).to_csv(OUT/"examples.csv",index=False)
pd.DataFrame({"id":frames["test"].id,"text":frames["test"].text,
              **{f"{system}_seed{i}":["|".join(np.array(labels)[row]) for row in pred]
                 for system,preds in predictions.items() for i,pred in enumerate(preds)}}).to_csv(
                     OUT/"selected_predictions.csv",index=False)

# Export every search-stage curve as a standard scientific plot.
for stage,ids in stages.items():
    rows=runs[runs.id.isin(ids)]
    if len(rows)<2:
        continue
    rows=rows.copy()
    varying=[col for col in group_cols+["seed"] if rows[col].nunique()>1]
    rows["setting"]=rows[varying].astype(str).agg(" / ".join,axis=1) if varying else rows.id
    fig,ax=plt.subplots(figsize=(10,max(4,.3*len(rows))))
    ax.barh(rows.setting,rows.dev_macro_f1,color="#326f81")
    ax.set_xlabel("Validation macro-F1")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT/f"sweep_{stage}.png",dpi=180)
    plt.close(fig)
# Pack literal data into a JSON file for editable slide charts and tables.
def records(frame):
    return json.loads(frame.to_json(orient="records"))
payload=dict(runs=records(runs),stages=stages,bpe=records(bpe),selected=selected,
    final=records(final),scores=records(scores),coverage=records(pd.read_csv(OUT/"coverage.csv")),
    repeated=records(repeated),bootstrap=records(intervals),coefficients=coefficients,
    slices=slice_rows,per_label=label_rows,examples=example_rows,
    neighbors=records(pd.read_csv(OUT/"neighbors.csv")),
    probes=records(pd.read_csv(OUT/"emotion_probes.csv")),
    historical_bpe=records(pd.read_csv(ROOT.parents[1]/"goemotions_results/bpe_sweep/results.csv")),
    dataset=records(pd.read_csv(OUT/"dataset_audit.csv")),
    thresholds=records(pd.concat(threshold_curves)),
    overlap=records(pd.read_csv(OUT/"test_without_overlap.csv")))
# JSON conversion handles numpy scalar types in manually assembled rows.
(OUT/"presentation_data.json").write_text(json.dumps(payload,indent=2,
    default=lambda value:value.item() if hasattr(value,"item") else str(value)),encoding="utf-8")
print(final.to_string(index=False))
print(intervals.to_string(index=False))
print("Neural fits",len(runs),"BPE fits",len(bpe))

# Standalone scientific figures for the presentation and later blog.
palette={"BPE":"#327587","Pool":"#B76C49","GRU":"#6B7698"}
fig,axes=plt.subplots(1,2,figsize=(10,4.5))
for ax,metric in zip(axes,["macro","micro"]):
    ax.bar(final.system,final[metric],yerr=final[f"{metric}_sd"].fillna(0),
           color=[palette[x] for x in final.system],capsize=4)
    ax.set(ylabel=f"Test {metric}-F1",ylim=(0,.7))
fig.tight_layout()
fig.savefig(OUT/"final_comparison.png",dpi=200)
plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4))
ax.axvspan(-.005,.005,color="#DCEAF1",label="Practical margin Ã‚Â±0.005")
for i,row in intervals.iterrows():
    ax.errorbar(row["mean"],i,xerr=[[row["mean"]-row.lower],[row.upper-row["mean"]]],
                fmt="o",color="#327587",capsize=4)
ax.axvline(0,color=".6",linewidth=1)
ax.set(yticks=range(len(intervals)),yticklabels=intervals.comparison,
       xlabel="Paired test macro-F1 difference with 95% bootstrap interval")
fig.tight_layout()
fig.savefig(OUT/"paired_bootstrap_intervals.png",dpi=200)
plt.close(fig)
coefs=pd.DataFrame(coefficients).query("feature!='intercept'")
fig,axes=plt.subplots(1,3,figsize=(12,4.5),sharey=True)
for ax,system in zip(axes,["BPE","Pool","GRU"]):
    rows=coefs[coefs.system==system]
    ax.errorbar(rows["mean"],range(len(rows)),
                xerr=[rows["mean"]-rows.lower,rows.upper-rows["mean"]],fmt="o",
                color=palette[system],capsize=4)
    ax.axvline(0,color=".6",linewidth=1)
    ax.set(title=system,xlabel="Log-odds coefficient",yticks=range(len(rows)),yticklabels=rows.feature)
fig.tight_layout()
fig.savefig(OUT/"error_logistic_intervals.png",dpi=200)
plt.close(fig)
slices=pd.DataFrame(slice_rows)
groups=slices[slices.feature.isin(["negation","long_20plus","oov"])].groupby(
    ["feature","present","system"]).agg(micro=("micro_f1","mean"),n=("n","first")).reset_index()
groups["slice"]=groups.apply(lambda row:f"{row.feature} = {row.present} (n={row.n})",axis=1)
fig,ax=plt.subplots(figsize=(10,5))
groups.pivot(index="slice",columns="system",values="micro").plot.barh(ax=ax,
    color=[palette[system] for system in sorted(palette)])
ax.set(xlabel="Test micro-F1, mean across seeds",ylabel="")
fig.tight_layout()
fig.savefig(OUT/"error_slices.png",dpi=200)
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4.5))
for ax,system in zip(axes,["Pool","GRU"]):
    row=selected[system][0]
    history=pd.read_csv(OUT/f"{row['id']}_epochs.csv")
    ax.plot(history.epoch,history.train_bce,label="Train loss")
    ax.plot(history.epoch,history.dev_bce,label="Dev BCE")
    ax.axvline(row["epoch"],color=".6",linestyle=":",label="Selected checkpoint")
    ax.set(title=f"{system}, {row['weight']} weighting",xlabel="Epoch",ylabel="Loss")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT/"final_learning_curves.png",dpi=200)
plt.close(fig)
per=pd.DataFrame(label_rows).groupby(["label","system"]).agg(f1=("f1","mean"),support=("support","first")).reset_index()
order=per.drop_duplicates("label").sort_values("support").label
fig,ax=plt.subplots(figsize=(10,10))
per.pivot(index="label",columns="system",values="f1").reindex(order).plot.barh(
    ax=ax,color=[palette[system] for system in sorted(palette)])
ax.set(xlabel="Test per-emotion F1, mean across seeds",ylabel="")
fig.tight_layout()
fig.savefig(OUT/"per_label.png",dpi=200)
plt.close(fig)
fig,ax=plt.subplots(figsize=(8,5))
for system,rows in pd.concat(threshold_curves).groupby("system"):
    average=rows.groupby("threshold").macro_f1.mean().sort_index()
    ax.plot(average.index,average.values,label=system,color=palette[system])
ax.set(xlabel="Global decision threshold",ylabel="Validation macro-F1",ylim=(0,.65))
ax.legend()
fig.tight_layout()
fig.savefig(OUT/"threshold_sensitivity.png",dpi=200)
plt.close(fig)

# Check the small BPE vocabulary margin on the same dev comments.
vocab_models=[]
for size in [10000,20000]:
    row=bpe.query("vocab==@size and min_frequency==2").sort_values(
        ["dev_macro_f1","dev_micro_f1"],ascending=False).iloc[0]
    probabilities=np.load(OUT/f"{row.id}.npz")["dev"]
    vocab_models.append(probabilities>=row.threshold)
rng=np.random.default_rng(41)
dev_truth=gold["dev"].astype(bool)
vocab_deltas=[]
for _ in range(1000):
    idx=rng.integers(0,len(dev_truth),len(dev_truth))
    vocab_deltas.append(macro(dev_truth[idx],vocab_models[1][idx])-macro(dev_truth[idx],vocab_models[0][idx]))
vocab_deltas=np.array(vocab_deltas)
vocab_interval=dict(comparison="20k minus 10k BPE on dev",point=float(
    macro(dev_truth,vocab_models[1])-macro(dev_truth,vocab_models[0])),
    lower=float(np.quantile(vocab_deltas,.025)),upper=float(np.quantile(vocab_deltas,.975)),
    within_rope=float((np.abs(vocab_deltas)<=.005).mean()))
pd.DataFrame([vocab_interval]).to_csv(OUT/"bpe_vocabulary_bootstrap.csv",index=False)
payload["vocab_bootstrap"]=vocab_interval
(OUT/"presentation_data.json").write_text(json.dumps(payload,indent=2,
    default=lambda value:value.item() if hasattr(value,"item") else str(value)),encoding="utf-8")
print(vocab_interval)

(OUT/"analysis_complete.json").write_text(json.dumps(dict(
    completed=True,neural_fits=len(runs),bpe_fits=len(bpe)),indent=2))
