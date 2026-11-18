"""Refresh the project notebook with the completed extension's evidence."""
import base64
import json
from pathlib import Path
import pandas as pd

root=Path(__file__).resolve().parent
path=root/"project.ipynb"
notebook=json.loads(path.read_text(encoding="utf-8"))
cells=notebook["cells"]
cells=[cell for cell in cells if not cell.get("id","").startswith("extended-")]
def add(kind,source,outputs=None):
    cell=dict(cell_type=kind,id=f"extended-{len(cells)}",metadata={},
              source=source.splitlines(keepends=True))
    if kind=="code":
        cell.update(execution_count=None,outputs=outputs or [])
    cells.append(cell)
def evidence(title,tables,figures):
    add("markdown","## "+title)
    code=[]
    outputs=[]
    for name in tables:
        file=root/"results/extended"/f"{name}.csv"
        code.append(f"display(pd.read_csv(extended_results / '{name}.csv'))")
        if file.exists():
            frame=pd.read_csv(file)
            outputs.append(dict(output_type="display_data",metadata={},
                data={"text/html":[frame.to_html(index=False)],"text/plain":[frame.to_string(index=False)]}))
    for name in figures:
        file=root/"results/extended"/f"{name}.png"
        code.append(f"display(Image(filename=str(extended_results / '{name}.png')))")
        if file.exists():
            outputs.append(dict(output_type="display_data",metadata={},
                data={"image/png":base64.b64encode(file.read_bytes()).decode("ascii"),
                      "text/plain":["Scientific figure"]}))
    add("code","\n".join(code),outputs)
add("markdown","""# Extended study

The extension completes the pretrained embedding experiments, restores the stronger BPE reference, and broadens the parameter search. See [the extended report](extended_report.md) for decisions and their evidence. The earlier pilot above remains a separate stage of the study.

The six requirements have linked evidence below. Model selection uses dev macro-F1. Three seeds and paired comment bootstrap samples describe different sources of uncertainty. The test set was already inspected during the pilot, so these results are exploratory.""")
add("code","""from pathlib import Path
import pandas as pd
from IPython.display import display, Image
extended_results = root / 'results' / 'extended'""")
evidence("Dataset, tokenization and coverage",
         ["dataset_audit","split_overlap","coverage","uncovered_tokens"],[])
evidence("What the pretrained vectors capture",["neighbors","emotion_probes"],[])
evidence("Search history and selected configurations",["runs","bpe_runs","repeated_summary"],
         ["sweep_pool_lr","sweep_pool_head","sweep_pool_dropout","sweep_gru_lr","sweep_gru_hidden","sweep_gru_dropout"])
evidence("Data size and initialization",[],["sweep_data_size","sweep_paired_controls","sweep_random_scale_control"])
evidence("Final comparison and uncertainty",["final_summary","bootstrap_intervals","test_without_overlap"],
         ["final_comparison","paired_bootstrap_intervals","threshold_sensitivity","final_learning_curves"])
evidence("Errors and examples",["error_logistic_intervals","error_slices","per_label","examples"],
         ["error_logistic_intervals","error_slices","per_label"])
for cell in cells:
    if cell["cell_type"]=="code":
        compile("".join(cell["source"]),str(path),"exec")
path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
print("Updated notebook with stored result tables and plots")
