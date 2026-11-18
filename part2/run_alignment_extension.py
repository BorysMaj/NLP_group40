import sys
from dataclasses import replace
sys.path.insert(0,"NLP_group40/part2")
from extended_experiments import *
exp=Experiment()
pool=exp.winner("pool_trainable")
gru=exp.winner("gru_dropout")
for cfg in [replace(pool,trainable=False),pool,gru]:
    exp.run(replace(cfg,tokenizer="aligned"),"contraction_alignment")
# Repeat the aligned trainable systems to avoid selecting from a single seed.
for cfg in [pool,gru]:
    for seed in [0,3,5]:
        exp.run(replace(cfg,tokenizer="aligned",seed=seed),"aligned_finalists")
