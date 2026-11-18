"""Extended GoEmotions experiments with resumable runs and validation-only selection."""
import argparse
from dataclasses import asdict, dataclass, replace
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.tokenize import TweetTokenizer, TreebankWordTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.trainers import BpeTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "part1"))
from goemotions_pipeline import load_splits, locate_data, dataset_audit, multilabel_metrics
from goemotions import read_vectors, PROBES, PROBE_QUERIES

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "extended"
OUT.mkdir(parents=True, exist_ok=True)
THRESHOLDS = np.arange(.05, .51, .025).round(3)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(3)


def threshold_scores(gold, probabilities):
    rows = []
    for threshold in THRESHOLDS:
        predicted = probabilities >= threshold
        tp = (predicted & gold.astype(bool)).sum(0)
        fp = (predicted & ~gold.astype(bool)).sum(0)
        fn = (~predicted & gold.astype(bool)).sum(0)
        denom = 2 * tp + fp + fn
        macro = np.divide(2 * tp, denom, out=np.zeros(len(tp), float), where=denom > 0).mean()
        micro = 2 * tp.sum() / max(1, denom.sum())
        rows.append(dict(threshold=threshold, macro_f1=macro, micro_f1=micro))
    return pd.DataFrame(rows)


def best_threshold(gold, probabilities):
    table = threshold_scores(gold, probabilities)
    return table.sort_values(["macro_f1", "micro_f1", "threshold"],
                             ascending=[False, False, True]).iloc[0], table


@dataclass(frozen=True)
class Config:
    source: str = "twitter50"
    encoder: str = "pool"
    trainable: bool = False
    hidden: int = 64
    head: int = 64
    dropout: float = .2
    lr: float = .003
    norm: str = "unit"
    tokenizer: str = "tweet"
    fraction: float = 1.
    epochs: int = 10
    budget: str = "epochs"
    weight: str = "none"
    seed: int = 0

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:14]


class Encoder(nn.Module):
    def __init__(self, matrix, cfg):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(matrix.copy()),
                                                     freeze=not cfg.trainable, padding_idx=0)
        dimension = matrix.shape[1]
        self.rnn = nn.GRU(dimension, cfg.hidden, batch_first=True) if cfg.encoder == "gru" else None
        width = cfg.hidden if self.rnn else dimension
        self.head = (nn.Sequential(nn.Linear(width, cfg.head), nn.ReLU(),
                                   nn.Dropout(cfg.dropout), nn.Linear(cfg.head, 28))
                     if cfg.head else nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(width, 28)))

    def forward(self, ids):
        mask = ids.ne(0)
        lengths = mask.sum(1).clamp_min(1)
        embedding = self.embedding(ids)
        if self.rnn:
            packed = pack_padded_sequence(embedding, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, hidden = self.rnn(packed)
            features = hidden[-1]
        else:
            features = (embedding * mask.unsqueeze(-1)).sum(1) / lengths.unsqueeze(1)
        return self.head(features)


class Experiment:
    def __init__(self):
        self.data = locate_data()
        self.frames, self.gold, self.labels = load_splits(self.data)
        self.targets = {name: torch.tensor(y, dtype=torch.float32) for name, y in self.gold.items()}
        audit, support, overlap = dataset_audit(self.frames, self.gold)
        audit.to_csv(OUT / "dataset_audit.csv", index=False)
        support.assign(label=lambda x:x.label.map(dict(enumerate(self.labels)))).to_csv(
            OUT / "label_support.csv", index=False)
        overlap.to_csv(OUT / "split_overlap.csv", index=False)
        self.cache = {}
        self.rows = (pd.read_csv(OUT / "runs.csv").to_dict("records")
                     if (OUT / "runs.csv").exists() else [])
        self.stages = json.loads((OUT / "stages.json").read_text()) if (OUT / "stages.json").exists() else {}
        self.coverage = pd.read_csv(OUT / "coverage.csv").to_dict("records") if (OUT / "coverage.csv").exists() else []
        self.neighbors = pd.read_csv(OUT / "neighbors.csv").to_dict("records") if (OUT / "neighbors.csv").exists() else []
        self.probes = pd.read_csv(OUT / "emotion_probes.csv").to_dict("records") if (OUT / "emotion_probes.csv").exists() else []

    def prepare(self, source, tokenizer):
        key = source, tokenizer
        if key in self.cache:
            return self.cache[key]
        cachefile = OUT / f"cache_{source}_{tokenizer}.pt"
        if cachefile.exists():
            result = torch.load(cachefile, weights_only=False)
            self.cache[key] = result
            return result
        tweet = TweetTokenizer(preserve_case=False, reduce_len=False, strip_handles=False)
        treebank = TreebankWordTokenizer()
        if tokenizer == "aligned":
            function = lambda text: [part for token in tweet.tokenize(text)
                                     for part in treebank.tokenize(token)]
        else:
            function = tweet.tokenize if tokenizer == "tweet" else lambda text: text.lower().split()
        words = {key:[function(text) for text in frame.text] for key,frame in self.frames.items()}
        wanted = {word for sequences in words.values() for seq in sequences for word in seq}
        file = ROOT / "results" / "embeddings" / f"{source}.gz"
        found = read_vectors(file, wanted)
        # Held-out strings only receive IDs to perform fixed pretrained lookup.
        # No held-out frequency, co-occurrence, or labels affect initialization.
        vocabulary = {word:index for index,word in enumerate(sorted(wanted), 2)}
        dim = len(next(iter(found.values())))
        matrix = np.zeros((len(vocabulary) + 2, dim), np.float32)
        train_counts = Counter(word for seq in words["train"] for word in seq)
        mean = np.mean([vector for word,vector in found.items() if word in train_counts], axis=0)
        matrix[1:] = mean
        for word, vector in found.items():
            matrix[vocabulary[word]] = vector
        arrays = {}
        feature_rows = {}
        coverage = []
        for split, sequences in words.items():
            ids = [[vocabulary[word] if word in found or word in train_counts else 1 for word in seq]
                   for seq in sequences]
            arrays[split] = torch.tensor([(row[:40] or [1]) + [0] * (40-max(1,len(row[:40])))
                                          for row in ids])
            lengths = np.array([len(seq) for seq in sequences])
            oov = np.array([sum(word not in found for word in seq) for seq in sequences])
            coverage.append(dict(source=source, tokenizer=tokenizer, split=split,
                                 tokens=int(lengths.sum()), token_coverage=1-oov.sum()/lengths.sum(),
                                 type_coverage=len(set(word for seq in sequences for word in seq)&found.keys())/
                                 len(set(word for seq in sequences for word in seq)),
                                 comments_with_oov=(oov>0).mean(), truncated=(lengths>40).mean()))
            feature_rows[split] = pd.DataFrame(dict(id=self.frames[split].id, length=lengths,
                oov_rate=oov / np.maximum(1,lengths),
                negation=[any(w in {"not","no","never","cannot"} or w.endswith("n't") for w in seq)
                          for seq in sequences],
                punctuation=[any(w in {"!","?","!!","??","!!!","???"} for w in seq) for seq in sequences],
                multilabel=self.gold[split].sum(1)>1))
        for query in PROBE_QUERIES:
            if query not in found:
                continue
            candidates = sorted(found.keys() & train_counts.keys())
            vectors = np.stack([found[word] for word in candidates])
            vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
            q = found[query]/np.linalg.norm(found[query])
            scores = vectors @ q
            for rank, idx in enumerate(np.argsort(-scores), 1):
                if candidates[idx] == query:
                    continue
                self.neighbors.append(dict(source=source, tokenizer=tokenizer, query=query,
                    rank=sum(r["source"]==source and r["tokenizer"]==tokenizer and r["query"]==query
                             for r in self.neighbors)+1,
                    neighbor=candidates[idx], cosine=float(scores[idx])))
                if sum(r["source"]==source and r["tokenizer"]==tokenizer and r["query"]==query
                       for r in self.neighbors) == 6:
                    break
        for query, candidates in PROBES.items():
            for i, word in enumerate(candidates):
                score = (float(found[query] @ found[word] /
                         (np.linalg.norm(found[query])*np.linalg.norm(found[word])))
                         if query in found and word in found else np.nan)
                self.probes.append(dict(source=source, tokenizer=tokenizer, query=query,
                    candidate=word, group="related" if i<3 else "contrasting", cosine=score))
        self.coverage.extend(coverage)
        result = matrix, arrays, feature_rows, dict(dimension=dim, vocabulary=len(vocabulary),
            file=str(file), sha256=hashlib.sha256(file.read_bytes()).hexdigest())
        torch.save(result, cachefile)
        pd.DataFrame(self.coverage).to_csv(OUT / "coverage.csv",index=False)
        pd.DataFrame(self.neighbors).to_csv(OUT / "neighbors.csv",index=False)
        pd.DataFrame(self.probes).to_csv(OUT / "emotion_probes.csv",index=False)
        self.cache[key] = result
        return result

    def run(self, cfg, stage):
        identity = cfg.identity
        stage_ids = self.stages.setdefault(stage, [])
        if identity not in stage_ids:
            stage_ids.append(identity)
        (OUT / "stages.json").write_text(json.dumps(self.stages,indent=2))
        old = next((row for row in self.rows if row["id"]==identity), None)
        if old:
            return old
        source = cfg.source.replace("randomscaled", "twitter").replace("random", "twitter")
        matrix, arrays, features, metadata = self.prepare(source, cfg.tokenizer)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        if cfg.source.startswith("random"):
            reference_norms = np.linalg.norm(matrix,axis=1,keepdims=True)
            matrix = np.random.default_rng(cfg.seed).normal(0,1/np.sqrt(matrix.shape[1]),
                                                            matrix.shape).astype(np.float32)
            if cfg.source.startswith("randomscaled"):
                scale = reference_norms if cfg.norm == "raw" else np.ones_like(reference_norms)
                matrix = matrix / np.maximum(np.linalg.norm(matrix,axis=1,keepdims=True),1e-8) * scale
            matrix[0] = 0
        elif cfg.norm == "unit":
            matrix = matrix / np.maximum(np.linalg.norm(matrix,axis=1,keepdims=True),1e-8)
        model = Encoder(matrix,cfg).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=.01)
        counts = self.gold["train"].sum(0)
        weight = torch.tensor(np.minimum(10,np.sqrt((len(arrays["train"])-counts)/counts)),
                              dtype=torch.float32,device=DEVICE) if cfg.weight.startswith("sqrt") else None
        criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
        rng = np.random.default_rng(cfg.seed)
        subset = rng.permutation(len(arrays["train"]))[:round(cfg.fraction*len(arrays["train"]))]
        # Precompute frozen pooled features to make the head sweep economical.
        frozen = cfg.encoder=="pool" and not cfg.trainable
        tensors = {key:value.to(DEVICE) for key,value in arrays.items()}
        if frozen:
            with torch.no_grad():
                vectors = {}
                for key,ids in tensors.items():
                    vectors[key] = (model.embedding(ids)*ids.ne(0).unsqueeze(-1)).sum(1) / ids.ne(0).sum(1).clamp_min(1).unsqueeze(1)
            forward = lambda value:model.head(value)
            inputs = vectors
        else:
            forward = model
            inputs = tensors
        y = {key:value.to(DEVICE) for key,value in self.targets.items()}

        @torch.no_grad()
        def predict(split):
            model.eval()
            logits = torch.cat([forward(batch) for batch in inputs[split].split(512)])
            return logits.sigmoid().cpu().numpy(), nn.functional.binary_cross_entropy_with_logits(logits,y[split]).item()

        best = None
        history = []
        started = time.monotonic()
        for epoch in range(1,cfg.epochs+1):
            model.train()
            total_loss = 0
            order = rng.permutation(subset)
            if cfg.budget=="steps":
                # Cycle through the same nested subset to match full-data updates.
                order = np.resize(order,len(arrays["train"]))
            for start in range(0,len(order),128):
                idx = order[start:start+128]
                optimizer.zero_grad(set_to_none=True)
                logits = forward(inputs["train"][idx])
                loss = criterion(logits,y["train"][idx])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),5)
                optimizer.step()
                total_loss += loss.item()*len(idx)
            probabilities,dev_bce = predict("dev")
            score, _ = best_threshold(self.gold["dev"],probabilities)
            history.append(dict(epoch=epoch,train_bce=total_loss/len(order),dev_bce=dev_bce,
                                **score.to_dict()))
            if best is None or (score.macro_f1,score.micro_f1)>(best[0],best[1]):
                best = (score.macro_f1,score.micro_f1,epoch,
                        {key:value.detach().cpu().clone() for key,value in model.state_dict().items()})
        model.load_state_dict(best[3])
        dev,_ = predict("dev")
        test,_ = predict("test")
        score, sweep = best_threshold(self.gold["dev"],dev)
        row = dict(id=identity,stage=stage,**asdict(cfg),epoch=best[2],
                   threshold=score.threshold,dev_macro_f1=score.macro_f1,dev_micro_f1=score.micro_f1,
                   seconds=time.monotonic()-started,parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                   dimension=matrix.shape[1])
        np.savez_compressed(OUT/f"{identity}.npz",dev=dev,test=test)
        torch.save(dict(config=asdict(cfg),state=best[3],metadata=metadata),OUT/f"{identity}.pt")
        pd.DataFrame(history).to_csv(OUT/f"{identity}_epochs.csv",index=False)
        sweep.to_csv(OUT/f"{identity}_thresholds.csv",index=False)
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(OUT/"runs.csv",index=False)
        print(stage,cfg.source,cfg.encoder,cfg.seed,identity,round(score.macro_f1,4),flush=True)
        del model, optimizer
        torch.cuda.empty_cache()
        return row

    def winner(self, stage):
        frame=pd.DataFrame(self.rows)
        frame=frame[frame.id.isin(self.stages[stage])]
        winner=frame.sort_values(["dev_macro_f1","dev_micro_f1"],ascending=False).iloc[0]
        return Config(**{key:type(value)(winner[key]) for key,value in asdict(Config()).items()})


def baseline(exp):
    path=OUT/"bpe_runs.csv"
    rows=pd.read_csv(path).to_dict("records") if path.exists() else []
    for size in (500,1000,1500,2000,3000,5000,10000,20000):
        tok=Tokenizer(BPE(unk_token="[UNK]"))
        tok.pre_tokenizer=Metaspace()
        tok.train_from_iterator(exp.frames["train"].text,trainer=BpeTrainer(
            vocab_size=size,min_frequency=2,special_tokens=["[UNK]"]))
        tok.save(str(OUT/f"bpe_v{size}.json"))
        vec=CountVectorizer(analyzer=lambda text:tok.encode(text).tokens,
                            vocabulary=tok.get_vocab(),binary=True)
        arrays={key:vec.transform(frame.text) for key,frame in exp.frames.items()}
        for c in (.03,.1,.3,1.,3.):
            name=f"bpe_v{size}_c{c}"
            if any(row["id"]==name for row in rows):
                continue
            model=OneVsRestClassifier(LogisticRegression(solver="liblinear",C=c,max_iter=300,random_state=0))
            model.fit(arrays["train"],exp.gold["train"])
            dev=model.predict_proba(arrays["dev"])
            test=model.predict_proba(arrays["test"])
            score,sweep=best_threshold(exp.gold["dev"],dev)
            np.savez_compressed(OUT/f"{name}.npz",dev=dev,test=test)
            sweep.to_csv(OUT/f"{name}_thresholds.csv",index=False)
            rows.append(dict(id=name,vocab=size,min_frequency=2,C=c,threshold=score.threshold,
                             dev_macro_f1=score.macro_f1,dev_micro_f1=score.micro_f1))
            pd.DataFrame(rows).to_csv(path,index=False)
            print(name,round(score.macro_f1,4),flush=True)


def main():
    exp=Experiment()
    # Persist the planned ranges before looking at results.
    plan=dict(seeds=[0,3,5],selection="dev macro-F1, then dev micro-F1",
              checkpoint="best dev macro-F1 over 10 epochs and global threshold grid",
              thresholds=THRESHOLDS.tolist(),device=DEVICE,torch=torch.__version__,
              data_hashes={key:hashlib.sha256((exp.data/f"{key}.tsv").read_bytes()).hexdigest()
                           for key in exp.frames},
              pilot_test_exposure="Pilot test scores were already inspected. Further test results are exploratory replication on the same test set.",
              scope="Finite staged search. Stage winners guide subsequent one-factor sweeps.",
              random_vocabulary="Same label-free token ID mapping as the corresponding pretrained system.",
              unknown="Mean training-vocabulary pretrained vector, padding zero",
              pool=dict(lr=[.001,.003,.01],head=[0,64,128],dropout=[0,.2,.5],trainable=[False,True]),
              gru=dict(hidden=[32,64,128],lr=[.001,.003,.01],dropout=[0,.2,.5]),
              dimensions=[25,50,100],bpe_vocab=[500,1000,1500,2000,3000,5000,10000,20000],bpe_C=[.03,.1,.3,1,3])
    (OUT/"search_plan.json").write_text(json.dumps(plan,indent=2))
    # Shared reference for source, dimension, normalization and tokenizer comparisons.
    base=Config()
    for source in ["twitter25","twitter50","twitter100","wiki50"]:
        exp.run(replace(base,source=source),"embedding")
    exp.run(replace(base,source="twitter50",norm="raw"),"normalization")
    exp.run(replace(base,source="twitter50",tokenizer="whitespace"),"tokenizer")
    # Select the source before architecture tuning.
    chosen=exp.winner("embedding")
    for norm in ["unit","raw"]:
        for tokenizer in ["tweet","whitespace"]:
            exp.run(replace(chosen,norm=norm,tokenizer=tokenizer),"preprocessing")
    chosen=exp.winner("preprocessing")
    for lr in [.001,.003,.01]:
        exp.run(replace(chosen,lr=lr),"pool_lr")
    chosen=exp.winner("pool_lr")
    for head in [0,64,128]:
        exp.run(replace(chosen,head=head),"pool_head")
    chosen=exp.winner("pool_head")
    for dropout in [0.,.2,.5]:
        exp.run(replace(chosen,dropout=dropout),"pool_dropout")
    chosen=exp.winner("pool_dropout")
    for trainable in [False,True]:
        exp.run(replace(chosen,trainable=trainable),"pool_trainable")
    pool=exp.winner("pool_trainable")
    # Fixed frozen and trainable comparisons also support attributing gains.
    for source in ["twitter50","wiki50"]:
        for encoder in ["pool","gru"]:
            for trainable in [False,True]:
                exp.run(replace(base,source=source,encoder=encoder,trainable=trainable,head=0),"controls")
    gru=replace(pool,encoder="gru",head=0,hidden=64)
    for lr in [.001,.003,.01]:
        exp.run(replace(gru,lr=lr),"gru_lr")
    gru=exp.winner("gru_lr")
    for hidden in [32,64,128]:
        exp.run(replace(gru,hidden=hidden),"gru_hidden")
    gru=exp.winner("gru_hidden")
    for dropout in [0.,.2,.5]:
        exp.run(replace(gru,dropout=dropout),"gru_dropout")
    gru=exp.winner("gru_dropout")
    for weight in ["none","sqrt"]:
        exp.run(replace(gru,weight=weight),"weighting")
    # Repeated finalists plus shared-architecture comparisons.
    for cfg in [pool,gru]:
        for seed in [0,3,5]:
            exp.run(replace(cfg,seed=seed),"finalists")
    for encoder in ["pool","gru"]:
        reference=replace(gru,encoder=encoder,source="twitter50",weight="none")
        for source,trainable in [("twitter50",False),("twitter50",True),("random50",True),("wiki50",True)]:
            for seed in [0,3,5]:
                exp.run(replace(reference,source=source,trainable=trainable,seed=seed),"paired_controls")
    # Matched optimizer-step data-size comparison.
    for source in ["twitter50","random50"]:
        for fraction in [.25,.5,1.]:
            for seed in [0,3,5]:
                exp.run(replace(gru,source=source,fraction=fraction,budget="steps",
                                seed=seed,weight="none"),"data_size")
    baseline(exp)
    (OUT/"complete.json").write_text(json.dumps(dict(completed=True,neural_fits=len(exp.rows)),indent=2))


if __name__=="__main__":
    main()
