"""Validate all ten routes in process or against a deployed UI relay."""
import argparse
import csv
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from urllib.error import HTTPError
from urllib.request import Request,urlopen

from api.service import PathSnap

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trips",type=Path)
    source.add_argument("--ui",help="Evaluator UI origin, for a real HTTP rehearsal")
    parser.add_argument("--out",type=Path,default=ROOT/"results/p4")
    parser.add_argument("--preview-payload",type=Path)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    mode="http" if args.ui else "direct"
    if args.ui:
        endpoint=args.ui.rstrip("/")
        with urlopen(endpoint+"/api/trips",timeout=10) as dialog:
            trip_ids=json.load(dialog)
        def predict(trip):
            with urlopen(Request(endpoint+"/api/predict/"+trip,method="POST"),timeout=60) as response:
                return json.load(response)
    else:
        service=PathSnap(args.trips)
        trip_ids=list(service.trip_dirs)
        predict=service.predict
    assert len(trip_ids)==len(set(trip_ids))==10
    with (ROOT/"results/p3/student_test_metrics.csv").open() as f:
        expected={(r["trip_id"],r["method"]):r for r in csv.DictReader(f)}
    compare=all((trip,"hmm") in expected for trip in trip_ids)
    rows=[]
    for trip in trip_ids:
        start=perf_counter()
        result=predict(trip)
        elapsed_ms=1000*(perf_counter()-start)
        assert set(result)=={"trip_id","n_points","gps","predicted_route","metrics"}
        assert result["trip_id"]==trip and 0<len(result["gps"])<=result["n_points"]
        assert all(set(seg)=={"arc_id","coords"} and len(seg["coords"])==2
                   for seg in result["predicted_route"])
        values=result["metrics"]
        assert set(values)=={"top1_acc_clf","top1_acc_hmm","route_continuity","candidate_miss_rate","inference_ms"}
        if compare:
            for key,method,column in (
                ("top1_acc_clf","classifier","top1_accuracy"),("top1_acc_hmm","hmm","top1_accuracy"),
                ("candidate_miss_rate","hmm","candidate_miss_rate"),
                ("route_continuity","hmm","decoder_piece_continuity"),
            ):
                assert abs(values[key]-float(expected[trip,method][column]))<1e-9,(trip,key,values[key])
        rows.append(dict(trip_id=trip,n_points=result["n_points"],route_segments=len(result["predicted_route"]),
                         elapsed_ms=round(elapsed_ms,2),**values))
        if args.preview_payload and trip=="00000076":
            args.preview_payload.write_text(json.dumps({"trips":trip_ids,"prediction":result},allow_nan=False))
        print(f"{mode.upper()} /predict/{trip} | {result['n_points']} points | {elapsed_ms:.1f} ms",flush=True)
    if args.ui:
        try:
            predict("12345678")
        except HTTPError as error:
            assert error.code==404
        else:
            raise AssertionError("Unknown trip accepted")
    with (args.out/f"{mode}_rehearsal.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Validated {len(rows)} trips. Median {median(r['elapsed_ms'] for r in rows):.1f} ms; "
          f"max {max(r['elapsed_ms'] for r in rows):.1f} ms. P3 parity: {compare}.")


if __name__=="__main__":
    main()
