"""Serve the evaluator UI and forward its two requests to Laptop 1."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

STATIC = Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api",required=True,help="Laptop 1 HTTP address, e.g. http://192.168.1.20:8000")
    parser.add_argument("--host",default="127.0.0.1")
    parser.add_argument("--port",default=8501,type=int)
    args=parser.parse_args()
    remote=urlsplit(args.api)
    if remote.scheme not in ("http","https") or not remote.netloc or remote.path not in ("","/") or remote.query or remote.fragment:
        parser.error("--api must contain only an HTTP(S) scheme and host, with an optional port")
    endpoint=args.api.rstrip("/")
    files={"/":("index.html","text/html; charset=utf-8"),
           "/style.css":("style.css","text/css; charset=utf-8"),
           "/app.js":("app.js","text/javascript; charset=utf-8")}

    class Handler(BaseHTTPRequestHandler):
        def send_data(self,status,mime,body):
            self.send_response(status)
            self.send_header("Content-Type",mime)
            self.send_header("Content-Length",str(len(body)))
            self.send_header("Cache-Control","no-store")
            self.end_headers()
            self.wfile.write(body)

        def relay(self,route):
            request=Request(endpoint+route,method=self.command)
            try:
                with urlopen(request,timeout=60) as r:
                    self.send_data(r.status,"application/json",r.read())
            except HTTPError as e:
                self.send_data(e.code,"application/json",e.read())
            except (URLError,TimeoutError):
                self.send_data(502,"application/json",b'{"detail":"Could not reach the PathSnap API on Laptop 1."}')

        def do_GET(self):
            if self.path in files:
                name,mime=files[self.path]
                self.send_data(200,mime,(STATIC/name).read_bytes())
            elif self.path=="/api/trips":
                self.relay("/trips")
            else:
                self.send_error(404)

        def do_POST(self):
            if re.fullmatch(r"/api/predict/[0-9]{8}(?:\?comparison=1)?",self.path):
                self.relay(self.path[4:])
            else:
                self.send_error(404)

    with ThreadingHTTPServer((args.host,args.port),Handler) as server:
        print(f"Evaluator UI http://{args.host}:{args.port} -> {endpoint}",flush=True)
        server.serve_forever()


if __name__=="__main__":
    main()
