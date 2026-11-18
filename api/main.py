"""FastAPI deployment entry point and local reference runner."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
from time import perf_counter
from urllib.parse import parse_qs, urlsplit

from .service import PathSnap, logger


def create_app(dataset_dir=None):
    from fastapi import FastAPI, HTTPException

    service = PathSnap(dataset_dir or os.environ.get("PATHSNAP_TRIPS","data/raw/evaluator_test"))
    app = FastAPI(title="PathSnap", version="1.0")

    @app.get("/trips")
    def trips():
        return list(service.trip_dirs)

    @app.post("/predict/{trip_id}")
    def predict(trip_id: str, comparison: bool = False):
        if trip_id not in service.trip_dirs:
            raise HTTPException(status_code=404, detail="Trip not found")
        start = perf_counter()
        result = service.predict(trip_id, comparison=comparison)
        logger.info("POST /predict/%s 200 | %d GPS points | %.1f ms request",
                    trip_id,result["n_points"],(perf_counter()-start)*1000)
        return result

    return app


def run_reference(service,host,port):
    """Same route contract for local testing when FastAPI is unavailable."""
    class Handler(BaseHTTPRequestHandler):
        def send_json(self,status,payload):
            raw=json.dumps(payload,allow_nan=False,separators=(",",":")).encode()
            self.send_response(status)
            self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if urlsplit(self.path).path == "/trips":
                self.send_json(200,list(service.trip_dirs))
            else:
                self.send_json(404,{"detail":"Not found"})

        def do_POST(self):
            request = urlsplit(self.path)
            path = request.path
            prefix="/predict/"
            trip=path[len(prefix):] if path.startswith(prefix) else None
            if trip not in service.trip_dirs:
                self.send_json(404,{"detail":"Trip not found"})
                return
            start=perf_counter()
            result=service.predict(trip, comparison=parse_qs(request.query).get("comparison") == ["1"])
            self.send_json(200,result)
            logger.info("POST /predict/%s 200 | %d GPS points | %.1f ms request",
                        trip,result["n_points"],(perf_counter()-start)*1000)

    with ThreadingHTTPServer((host,port),Handler) as server:
        logger.info("Local reference backend listening on %s:%d",host,port)
        server.serve_forever()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trips",required=True,help="Directory of ten trip folders")
    parser.add_argument("--host",default="127.0.0.1")
    parser.add_argument("--port",type=int,default=8000)
    parser.add_argument("--reference-http",action="store_true",help="Run local HTTP without FastAPI dependencies")
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(levelname)s %(message)s")
    if args.reference_http:
        run_reference(PathSnap(args.trips),args.host,args.port)
    else:
        import uvicorn
        uvicorn.run(create_app(args.trips),host=args.host,port=args.port)


if __name__ == "__main__":
    main()
