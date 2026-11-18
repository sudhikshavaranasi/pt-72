"""Render the measured model comparison and Bayesian trial history."""
import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


NAMES = {
    "logistic": "Logistic regression", "gaussian_nb": "Gaussian naive Bayes",
    "rbf_prototypes": "RBF prototypes", "boosted_stumps": "Boosted stumps",
    "shallow_mlp": "Shallow MLP",
}


def name(identifier):
    base = identifier.removesuffix("_tuned")
    return NAMES[base] + (" (tuned)" if identifier != base else "")


def text(draw, xy, content, size, color="#172741"):
    draw.text(xy, content, font=ImageFont.load_default(size=size), fill=color)


def draw_comparison(rows, final, output):
    canvas = Image.new("RGB", (1340, 660), "#f4f7fb")
    draw = ImageDraw.Draw(canvas)
    text(draw, (30, 23), "PathSnap model comparison", 31)
    text(draw, (30, 66), "Four trip-grouped folds  |  10 student-test trips  |  All observations used for Top-1", 18)
    metrics = [
        ("cv_macro_f1","CV macro F1"), ("macro_f1","Test macro F1"),
        ("top1_accuracy","Test Top-1"), ("route_continuity","Test continuity"),
        ("log_loss","Test log loss"),
    ]
    max_loss = max(float(r["log_loss"]) for r in rows)
    draw.rounded_rectangle((30, 108, 1310, 590), radius=12, fill="white", outline="#d6dfeb")
    text(draw,(54,137),"Classifier",18)
    for j,(_,label) in enumerate(metrics):
        text(draw,(315+196*j,137),label,18)
    for i,row in enumerate(rows):
        y=187+i*62
        color="#007d72" if row["model"]==final else "#285aa3"
        if row["model"]==final:
            draw.rectangle((41,y-8,1294,y+45),fill="#e6f6f1")
        text(draw,(54,y),name(row["model"]),17)
        for j,(key,_) in enumerate(metrics):
            x=315+196*j
            value=float(row[key])
            capacity=max_loss if key=="log_loss" else 1
            draw.rounded_rectangle((x,y+25,x+165,y+32),radius=3,fill="#e8edf4")
            draw.rounded_rectangle((x,y+25,x+max(5,round(165*value/capacity)),y+32),radius=3,fill=color)
            label=f"{value:.3f}" if key=="log_loss" else f"{value:.1%}"
            text(draw,(x,y),label,16)
    text(draw,(31,615),f"Selected by CV macro F1: {name(final)}   |   Lower log loss is better.",17)
    canvas.save(output)


def draw_trials(log, output, baseline):
    rows=log["trials"]
    canvas=Image.new("RGB",(1120,525),"#f4f7fb")
    draw=ImageDraw.Draw(canvas)
    text(draw,(30,22),"Bayesian tuning results",30)
    text(draw,(30,64),log["strategy"],17)
    left,right,top,bottom=125,1070,116,447
    draw.rectangle((left,top,right,bottom),fill="white",outline="#c9d4e5")
    values=[r["cv_macro_f1"] for r in rows]+[baseline]
    ymin,ymax=min(values)-.001,max(values)+.001
    def pos(i,v):
        return left+20+(right-left-40)*i/(len(rows)-1), bottom-(bottom-top)*(v-ymin)/(ymax-ymin)
    for tick in range(5):
        val=ymin+(ymax-ymin)*tick/4
        y=pos(0,val)[1]
        draw.line((left,y,right,y),fill="#e1e7ef")
        text(draw,(42,y-9),f"{val:.3f}",16)
    baseline_y=pos(0,baseline)[1]
    draw.line((left,baseline_y,right,baseline_y),fill="#a63b3b",width=2)
    coords=[pos(i,r["cv_macro_f1"]) for i,r in enumerate(rows)]
    draw.line(coords,fill="#285aa3",width=3)
    for i,(x,y) in enumerate(coords):
        color="#007d72" if i==log["best_trial"] else "#285aa3"
        draw.ellipse((x-6,y-6,x+6,y+6),fill=color)
        text(draw,(x-6,461),str(i+1),15)
    text(draw,(30,493),"Y: grouped CV macro F1    |    X: trial number    |    Red line: untuned",17)
    canvas.save(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[2])
    args=parser.parse_args()
    results=args.root/"results"
    with (results/"model_comparison.csv").open() as stream:
        rows=list(csv.DictReader(stream))
    model=json.loads((args.root/"configs/model.yaml").read_text())
    trials=json.loads((results/"optuna/trials.json").read_text())
    plot=results/"plots"
    plot.mkdir(exist_ok=True)
    draw_comparison(rows,model["model_name"],plot/"model_comparison.png")
    baseline=next(float(row["cv_macro_f1"]) for row in rows if row["model"]==model["algorithm"])
    draw_trials(trials,plot/"bayesian_trials.png",baseline)


if __name__=="__main__":
    main()

