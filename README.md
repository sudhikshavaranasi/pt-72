# PathSnap

PathSnap matches GPS observations to road segments using a trained classifier and a Viterbi decoder.

## Backend laptop

From the repository root, install the dependencies and start the API. Replace the example path with the folder containing the trip directories.

    python -m pip install -r requirements.txt
    python -m api.main --trips "C:\path\to\trips" --host 0.0.0.0 --port 8000

## UI laptop

On the same network, run the UI with the backend laptop's IPv4 address in place of the example.

    python -m ui.app --api http://192.168.1.20:8000 --port 8501

Open http://127.0.0.1:8501, select a trip, and click Reconstruct route. Use Route result to compare the classifier and Viterbi routes.

## Validation

With the UI running and the backend dependencies installed, run:

    python -m scripts.validate --ui http://127.0.0.1:8501 --out data/validation

Keep raw trips and evaluator files outside the public repository.
