# PathSnap

PathSnap reconstructs a route from noisy GPS observations and a directed road network. It scores nearby road segments with a trained classifier, then uses Viterbi decoding to find a sequence consistent with the road graph and the observed movement. A FastAPI backend runs inference and a browser UI displays the GPS trace, both route predictions, and the evaluation metrics.

Machine Learning Hackathon · UE24CS352A · Section 5E · Team 72

## Results

On 10 held-out student-test trips with 19,160 aligned GPS observations, Viterbi increased **point-level Top-1 accuracy from 80.27% to 86.95%**, a gain of **6.69 percentage points** calculated before rounding.

| Student-test result | Tuned classifier | Classifier + Viterbi |
| --- | ---: | ---: |
| Correct road-segment matches | 15,379 | 16,660 |
| Point-level Top-1 accuracy | 80.27% | 86.95% |

Top-1 is the share of aligned GPS observations assigned to the reference road segment. The reported score pools observations across trips. See the [per-trip results](results/p3/student_test_metrics.csv) and [direct API service rehearsal](results/p4/direct_rehearsal.csv).

Five classifier families were evaluated using four-fold cross-validation grouped by trip. The tuned shallow MLP led with a grouped CV macro F1 of **0.8601**. See the [model comparison](results/p2/model_comparison.csv), [fold assignments](results/p2/cv/fold_assignment.csv), and [Bayesian tuning trials](results/p2/optuna/trials.json).

## Run locally

Use **Python 3.11+** and a copy of the trip dataset. The commands below are for Windows PowerShell.

### 1. Start the backend

Open PowerShell and run:

~~~powershell
git clone https://github.com/sudhikshavaranasi/pt-72.git
Set-Location .\pt-72
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
$trips = 'C:\path\to\student_test'
& .\.venv\Scripts\python.exe -m api.main --trips $trips --host 127.0.0.1 --port 8000
~~~

Set **$trips** to the folder directly containing the trip directories. Each trip directory has an eight-digit ID and matching **.track**, **.nodes**, and **.arcs** files. An optional **.route** file supplies the reference for accuracy and candidate-miss metrics. The API calculates those metrics after inference. Use the student-test split for rehearsal and keep raw datasets out of Git.

### 2. Start the browser UI

Open another PowerShell window in the repository root:

~~~powershell
python -m ui.app --api http://127.0.0.1:8000 --port 8501
~~~

Open [http://127.0.0.1:8501](http://127.0.0.1:8501), select a trip, and click **Reconstruct route**. Use **Route result** to switch between the classifier, Viterbi, and comparison views. The map needs internet access to load Leaflet and OpenStreetMap tiles.

### Run on two laptops

Keep both laptops on the same network. On the backend laptop, use the command in step 1 with **--host 0.0.0.0** so the second laptop can connect. On the UI laptop, use step 2 from its own checkout and set **--api** to the backend laptop's IPv4 address and port, such as **http://192.168.1.20:8000**. Open **http://127.0.0.1:8501** on the UI laptop.

If the trip list stays empty, test the backend address from the UI laptop with **Invoke-RestMethod http://192.168.1.20:8000/trips**, using the actual address.

## API and validation

| Request | Response |
| --- | --- |
| **GET /trips** | Available trip IDs |
| **POST /predict/{trip_id}** | GPS points, reconstructed road segments, and metrics |
| **POST /predict/{trip_id}?comparison=1** | The same result plus the classifier-only route |

With the local backend and UI running, run the smoke check from the backend checkout. It checks all ten trip responses, confirms that an unknown trip returns HTTP 404, and writes **http_rehearsal.csv** under **data/validation/**. When the loaded trips match the recorded student-test set, it also checks metric parity.

~~~powershell
& .\.venv\Scripts\python.exe -m scripts.validate --ui http://127.0.0.1:8501 --out data/validation
~~~

To run the decoder unit tests without the trip dataset:

~~~powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests
~~~

## How the pipeline works

1. **Generate candidates.** Project coordinates into local metres and keep up to 30 road segments within 25 m of each GPS point.
2. **Score candidates.** The tuned shallow MLP uses eight features, including distance to the road, heading agreement, and GPS speed. Classifier-only prediction selects the highest-scoring candidate at each point.
3. **Decode the route.** Viterbi combines candidate scores with transition scores based on movement and directed road connectivity. When a GPS gap or an impossible transition breaks the sequence, decoding resumes in a new piece.
4. **Serve the result.** FastAPI returns the GPS points and reconstructed route. The browser displays the route and any available metrics.

| Location | Contents |
| --- | --- |
| [src/](src/) and [configs/](configs/) | GPS and road loading, feature extraction, scoring, and HMM decoding |
| [models/](models/) | Saved tuned classifier |
| [experiments/](experiments/) | Dataset analysis, classifier training, and tuning |
| [api/](api/) and [ui/](ui/) | FastAPI backend and browser interface |
| [results/](results/) | Recorded model comparisons, evaluation results, and rehearsal timings |
| [scripts/](scripts/) and [tests/](tests/) | Smoke validation and decoder unit tests |

## Team

- Sriviswaja Mudumba
- Sudhiksha Varanasi
- Suhas Mohan Kumar
- Suhas Sreenath Iyengar
