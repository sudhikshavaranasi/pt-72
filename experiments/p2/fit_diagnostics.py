"""Refit the recorded configurations and check fit-sample vs grouped CV Macro F1."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

from .p2 import Dataset, SEED, fit, sample, write_csv
from src.eval.metrics import candidate_metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1",type=Path,required=True)
    parser.add_argument("--raw",type=Path,required=True)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[2])
    args=parser.parse_args()
    config=json.loads((args.root/"configs/model.yaml").read_text())
    configs=json.loads((args.root/"results/model_configurations.json").read_text())
    with (args.root/"results/model_comparison.csv").open() as stream:
        summary={r["model"]:r for r in csv.DictReader(stream)}
    data=Dataset(args.p1,args.raw,config["feature_cols"])
    ix,w=sample(data,data.train,SEED)
    report=[]
    for tag,params in configs.items():
        start=perf_counter()
        model=fit(data,tag.removesuffix("_tuned"),config["feature_cols"],params,ix,w,SEED)
        prob=model.probability(data.x[ix])
        result=candidate_metrics(data.labels[ix],prob)
        cv=float(summary[tag]["cv_macro_f1"])
        report.append(dict(model=tag,fit_rows=len(ix),fit_sample_macro_f1=result["macro_f1"],
                           cv_macro_f1=cv,fit_minus_cv_f1=result["macro_f1"]-cv,
                           student_test_macro_f1=float(summary[tag]["macro_f1"]),
                           diagnostic_s=perf_counter()-start))
    write_csv(args.root/"results/fit_diagnostics.csv",report)


if __name__=="__main__":
    main()

