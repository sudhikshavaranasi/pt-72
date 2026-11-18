"""Grouped-CV, Bayesian tuning, and contract-compatible candidate score export."""
import argparse
from collections import defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import pickle
from time import perf_counter

import numpy as np

from src.eval.metrics import candidate_metrics, observation_metrics, top_arcs
from src.models import CandidateModel, CONFIGS, SPACES

ROOT = Path(__file__).resolve().parents[2]
SEED = 72
N_FOLDS = 4
TRIP_CAP = 4500


def records(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path,"rt",encoding="utf-8-sig",newline="") as stream:
        yield from csv.DictReader(stream)


def write_csv(path, rows):
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer = csv.DictWriter(stream,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


class Dataset:
    def __init__(self,p1,raw,feature_cols):
        point_map = {}
        point_data = []
        names, splits, networks = [], [], []
        trip_map = {}
        for row in records(p1/"point_references.csv"):
            split, name = row["split"], row["trip_id"]
            key = (split,name)
            if key not in trip_map:
                trip_map[key] = len(names)
                names.append(name)
                splits.append(split)
                networks.append(np.loadtxt(raw/split/name/f"{name}.arcs", dtype=np.int32, ndmin=2))
            trip = trip_map[key]
            obs = int(row["observation_index"])
            point_map[trip,obs] = len(point_data)
            point_data.append((trip,obs,float(row["timestamp_s"]),int(row["reference_arc_id"]),
                               not int(row["candidate_miss"])))
        array = np.asarray(point_data)
        self.trips = array[:,0].astype(np.int32)
        self.obs = array[:,1].astype(np.int32)
        self.time = array[:,2]
        self.truth = array[:,3].astype(np.int32)
        self.in_cands = array[:,4].astype(bool)
        self.names = np.array(names)
        self.splits = np.array(splits)
        self.networks = networks
        self.point_split = self.splits[self.trips]
        pieces_start = np.r_[True,(np.diff(self.trips)!=0) | (np.diff(self.obs)!=1) |
                             (np.diff(self.time)>30) | (np.diff(self.time)<=0)]
        self.piece = np.cumsum(pieces_start)
        features, keys = [], []
        for row in records(p1/"features.csv.gz"):
            trip = trip_map[row["split"],row["trip_id"]]
            point = point_map[trip,int(row["observation_index"])]
            keys.append((point,int(row["arc_id"]),int(row["label"])))
            features.append(tuple(float(row[c]) if row[c] != "" else np.nan for c in feature_cols))
        self.x = np.array(features,dtype=np.float32)
        ids = np.array(keys,dtype=np.int32)
        self.points, self.arcs, self.labels = ids.T
        self.row_trips = self.trips[self.points]
        if np.any(np.diff(self.points)<0) or np.any(self.labels != (self.arcs == self.truth[self.points])):
            raise ValueError("Candidate rows must be sorted by observation and match the references")
        self.train = np.flatnonzero(self.splits[self.row_trips]=="train")
        self.test = np.flatnonzero(self.splits[self.row_trips]=="student_test")
        self.test_points = np.flatnonzero(self.point_split=="student_test")

    def measure(self, rows, scores, point_idx):
        cp = self.points[rows]
        pred = top_arcs(np.searchsorted(point_idx,cp),self.arcs[rows],scores,len(point_idx))
        source = np.full(len(pred),-1,dtype=np.int32)
        target = source.copy()
        trips = self.trips[point_idx]
        for trip in np.unique(trips):
            positions = np.flatnonzero((trips == trip) & (pred >= 0))
            source[positions],target[positions] = self.networks[trip][pred[positions]].T
        return {**candidate_metrics(self.labels[rows],scores),
                **observation_metrics(self.truth[point_idx],self.in_cands[point_idx],
                                      pred,source,target,self.piece[point_idx])}

    def write_scores(self, directory, name, part, rows, score, parquet):
        path=directory/f"{name}_{part}.csv.gz"
        with gzip.open(path,"wt",newline="",encoding="utf-8",compresslevel=3) as stream:
            writer=csv.writer(stream)
            writer.writerow(("trip_id","obs_idx","t","arc_id","score"))
            writer.writerows(
                (self.names[self.trips[p]],self.obs[p],self.time[p],a,format(float(s),".8g"))
                for p,a,s in zip(self.points[rows],self.arcs[rows],score)
            )
        if parquet:
            import pyarrow as pa
            import pyarrow.csv as cc
            import pyarrow.parquet as pq
            schema=dict(trip_id=pa.string(),obs_idx=pa.int32(),t=pa.float64(),arc_id=pa.int32(),score=pa.float32())
            with gzip.open(path,"rb") as stream:
                table=cc.read_csv(stream,convert_options=cc.ConvertOptions(column_types=schema))
            pq.write_table(table,path.with_suffix("").with_suffix(".parquet"),compression="zstd")


def fold_groups(data):
    train=np.flatnonzero(data.splits=="train")
    count=np.bincount(data.trips,minlength=len(data.splits))
    buckets=[[] for _ in range(N_FOLDS)]
    totals=np.zeros(N_FOLDS)
    for trip in sorted(train,key=lambda t:(-count[t],data.names[t])):
        fold=int(totals.argmin())
        buckets[fold].append(trip)
        totals[fold]+=count[trip]
    return [np.array(b) for b in buckets]


def sample(data,eligible,seed):
    rng=np.random.default_rng(seed)
    rows, weights=[],[]
    for t in np.unique(data.row_trips[eligible]):
        idx=eligible[data.row_trips[eligible]==t]
        if len(idx)>TRIP_CAP:
            idx=np.sort(rng.choice(idx,TRIP_CAP,replace=False))
        rows.append(idx)
        weights.append(np.full(len(idx),1/math.sqrt(len(idx)),dtype=np.float32))
    w=np.concatenate(weights)
    return np.concatenate(rows),w/w.mean()


def fit(data,name,feature_cols,config,rows,weights,seed):
    labels=data.labels[rows].astype(np.float32)
    weights=weights*np.where(labels==1,np.float32(1.5),np.float32(1))
    return CandidateModel(name,feature_cols,config,seed).fit(data.x[rows],labels,weights)


def grouped_cv(data,name,feature_cols,config,folds,cv_log,tag):
    scores=np.empty(len(data.train),np.float32)
    metrics=[]
    for fold,(sample_rows,weights,valid,points) in enumerate(folds):
        start=perf_counter()
        model=fit(data,name,feature_cols,config,sample_rows,weights,SEED+fold)
        fit_time=perf_counter()-start
        start=perf_counter()
        score=model.probability(data.x[data.train[valid]])
        infer_time=perf_counter()-start
        scores[valid]=score
        row=data.measure(data.train[valid],score,points)
        cv_log.append(dict(model=tag,fold=fold,fit_s=fit_time,inference_s=infer_time,**row))
        metrics.append(row)
    return float(np.mean([r["macro_f1"] for r in metrics])),float(np.mean([r["top1_accuracy"] for r in metrics])),scores


def tune(name, evaluate, trials):
    # Optuna TPE when installed. GP expected improvement otherwise.
    try:
        import optuna
    except ModuleNotFoundError as error:
        if error.name!="optuna":
            raise
        optuna=None
    bounds=SPACES[name]
    ran=[]
    def run(params):
        started=perf_counter()
        f1,top1,scores=evaluate(params,len(ran))
        ran.append(dict(trial=len(ran),params=params,cv_macro_f1=f1,cv_top1=top1,
                        elapsed_s=perf_counter()-started,scores=scores))
        print(f"Tuning {name} [{len(ran)}/{trials}]: Macro F1 {f1:.4f}, Top-1 {top1:.4f}",flush=True)
        return f1

    if optuna is not None:
        strategy="Optuna TPE"
        study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(
            seed=SEED,n_startup_trials=3))
        def objective(trial):
            return run({key:(trial.suggest_int(key,lo,hi) if integer else trial.suggest_float(key,lo,hi))
                        for key,lo,hi,integer in bounds})
        study.optimize(objective,n_trials=trials)
    else:
        strategy="Gaussian-process Bayesian search with expected improvement"
        rng=np.random.default_rng(SEED)
        tested=[]
        def kernel(a,b):
            return np.exp(-.5*np.sum((a[:,None,:]-b[None,:,:])**2,axis=2)/.28**2)
        for i in range(trials):
            if i<3:
                option=rng.random(len(bounds))
            else:
                x=np.asarray(tested)
                y=np.array([r["cv_macro_f1"] for r in ran])
                points=rng.random((768,len(bounds)))
                covariance=kernel(x,x)+np.eye(i)*1e-5
                cross=kernel(x,points)
                centered=(y-y.mean())/max(y.std(),1e-5)
                mean=cross.T @ np.linalg.solve(covariance,centered)
                sd=np.sqrt(np.maximum(1-(cross*np.linalg.solve(covariance,cross)).sum(axis=0),1e-10))
                gain=mean-centered.max()-.01
                z=gain/sd
                cdf=np.array([.5*(1+math.erf(float(v)/math.sqrt(2))) for v in z])
                ei=gain*cdf+sd*np.exp(-.5*z*z)/math.sqrt(2*math.pi)
                option=points[ei.argmax()]
            params={}
            actual=[]
            for value,(key,lo,hi,integer) in zip(option,bounds):
                mapped=lo+value*(hi-lo)
                params[key]=int(round(mapped)) if integer else float(mapped)
                actual.append((params[key]-lo)/(hi-lo))
            tested.append(actual)
            run(params)
    return strategy,ran


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--p1",type=Path,required=True)
    p.add_argument("--raw",type=Path,required=True)
    p.add_argument("--out",type=Path,default=ROOT)
    p.add_argument("--trials",type=int,default=10)
    args=p.parse_args()
    if args.trials<4:
        p.error("use at least four tuning trials")
    feature_cols=json.loads((ROOT/"configs"/"features.yaml").read_text(encoding="utf-8-sig"))["FEATURE_COLS"]
    out=args.out
    for folder in ("configs","models","results/scores","results/cv","results/optuna"):
        (out/folder).mkdir(parents=True,exist_ok=True)
    np.seterr(over="raise",invalid="raise")
    data=Dataset(args.p1,args.raw,feature_cols)
    folds=[]
    assignments=[]
    for fold,trips in enumerate(fold_groups(data)):
        valid=np.isin(data.row_trips[data.train],trips)
        tr,w=sample(data,data.train[~valid],SEED+fold)
        points=np.flatnonzero(np.isin(data.trips,trips))
        folds.append((tr,w,np.flatnonzero(valid),points))
        assignments.extend(dict(trip_id=data.names[t],fold=fold) for t in trips)
    write_csv(out/"results/cv/fold_assignment.csv",assignments)
    cv_log=[]
    base={}
    for name,config in CONFIGS.items():
        result=grouped_cv(data,name,feature_cols,config,folds,cv_log,name)
        base[name]=result
        print(f"CV {name}: Macro F1 {result[0]:.4f}, Top-1 {result[1]:.4f}",flush=True)
    winner=max(base,key=lambda n:base[n][:2])
    def objective(params,index):
        return grouped_cv(data,winner,feature_cols,{**CONFIGS[winner],**params},folds,cv_log,f"trial_{index:02}")
    strategy,trials=tune(winner,objective,args.trials)
    best=max(trials,key=lambda r:(r["cv_macro_f1"],r["cv_top1"]))
    tuned=winner+"_tuned"
    configs={**CONFIGS,tuned:{**CONFIGS[winner],**best["params"]}}
    cv={**base,tuned:(best["cv_macro_f1"],best["cv_top1"],best["scores"])}
    final=max((winner,tuned),key=lambda n:cv[n][:2])
    write_csv(out/"results/cv/fold_metrics.csv",cv_log)
    tuning={"strategy":strategy,"objective":"mean grouped CV candidate Macro F1 at threshold 0.5",
            "search_space":SPACES[winner],"trial_count":len(trials),"best_trial":best["trial"],
            "trials":[{k:v for k,v in r.items() if k!="scores"} for r in trials]}
    (out/"results/optuna/trials.json").write_text(json.dumps(tuning,indent=2),encoding="utf-8")
    fit_rows,w=sample(data,data.train,SEED)
    summary=[]
    try:
        import pyarrow
    except ModuleNotFoundError:
        parquet=False
    else:
        parquet=True
    for tag,config in configs.items():
        name=winner if tag==tuned else tag
        start=perf_counter()
        model=fit(data,name,feature_cols,config,fit_rows,w,SEED)
        fit_time=perf_counter()-start
        start=perf_counter()
        values=model.probability(data.x[data.test])
        infer_time=perf_counter()-start
        metrics=data.measure(data.test,values,data.test_points)
        summary.append(dict(model=tag,cv_macro_f1=cv[tag][0],cv_top1=cv[tag][1],
                            fit_s=fit_time,inference_s=infer_time,**metrics))
        data.write_scores(out/"results/scores",tag,"train",data.train,cv[tag][2],parquet)
        data.write_scores(out/"results/scores",tag,"student_test",data.test,values,parquet)
        if tag==final:
            with (out/"models/final_model.joblib").open("wb") as stream:
                pickle.dump(model,stream,protocol=4)
        print(f"Test {tag}: Macro F1 {metrics['macro_f1']:.4f}, Top-1 {metrics['top1_accuracy']:.4f}",flush=True)
    write_csv(out/"results/model_comparison.csv",summary)
    model_config=dict(model_name=final,algorithm=winner,hyperparameters=configs[final],seed=SEED,
                      feature_cols=feature_cols,selection_metric=tuning["objective"],probability_threshold=.5,
                      scores_are_calibrated=False,train_scores="out-of-fold",parquet=parquet)
    (out/"configs/model.yaml").write_text(json.dumps(model_config,indent=2),encoding="utf-8")
    (out/"results/model_configurations.json").write_text(json.dumps(configs,indent=2),encoding="utf-8")
    print(f"Selected {final} by grouped CV. Score output: {'Parquet + CSV' if parquet else 'gzip CSV'}",flush=True)


if __name__=="__main__":
    main()

