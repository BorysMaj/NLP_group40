import sys
from dataclasses import replace
sys.path.insert(0,"NLP_group40/part2")
from extended_experiments import *
exp=Experiment()
gru=exp.winner("gru_dropout")
for encoder in ["pool","gru"]:
    for seed in [0,3,5]:
        exp.run(replace(gru,encoder=encoder,source="randomscaled50",seed=seed,weight="none"),
                "random_scale_control")
