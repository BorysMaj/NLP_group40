import sys
from dataclasses import replace
import numpy as np
sys.path.insert(0,"NLP_group40/part2")
import extended_experiments as ext
ext.THRESHOLDS=np.arange(.05,.901,.025).round(3)
exp=ext.Experiment()
cfg=replace(exp.winner("gru_dropout"),weight="sqrt_wide")
for seed in [0,3,5]:
    exp.run(replace(cfg,seed=seed),"weighted_finalists")
