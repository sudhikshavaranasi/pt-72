\# PathSnap – GPS Map Matching and Route Reconstruction



\## 1. Project Overview



PathSnap is a GPS map-matching system that reconstructs the most likely road route followed by a vehicle from noisy GPS observations.



GPS recordings are often inaccurate because of signal noise, urban interference, irregular sampling and positioning errors. PathSnap addresses this by matching GPS observations to candidate road segments and using road-network connectivity and probabilistic sequence decoding to identify a plausible route.



The system combines:



\- GPS preprocessing and geometric projection.

\- Candidate road-segment generation.

\- Machine-learning-based candidate scoring.

\- Hidden Markov Model (HMM) decoding using the Viterbi algorithm.

\- Route evaluation and visualization.



\## 2. System Architecture



The project is divided into four major components.



\### P1 – Data Foundation and Candidate Generation



\- Loads GPS tracks, road-network nodes and directed arcs.

\- Projects geographic coordinates into a local metric coordinate system.

\- Generates candidate road segments for each GPS observation.

\- Calculates candidate features such as distance, direction and position along a segment.

\- Aligns reference routes for evaluation.



\### P2 – Candidate Classification



\- Trains candidate classification models.

\- Uses grouped cross-validation to evaluate model performance.

\- Performs Bayesian hyperparameter optimization.

\- Exports candidate-level prediction scores for downstream inference.



\### P3 – HMM Map Matching and Route Evaluation



\- Constructs a directed road graph.

\- Calculates road-network distances using Dijkstra's algorithm.

\- Uses classifier probabilities as HMM emission information.

\- Models transitions using GPS displacement and road-network distance.

\- Applies Viterbi decoding to find the most likely sequence of road segments.

\- Handles GPS gaps, unreachable transitions and stationary observations.

\- Evaluates route accuracy and continuity.



\### P4 – Application and Visualization



\- Provides the application interface and API integration.

\- Displays GPS observations and predicted routes.

\- Supports comparison of route predictions.



\## 3. Technologies Used



| Technology | Purpose |

|---|---|

| Python 3.11 | Core implementation |

| NumPy | Numerical computation |

| Pandas | Data processing |

| Scikit-learn | Machine learning utilities |

| Matplotlib | Evaluation plots |

| Folium | Interactive route maps |

| Streamlit | User interface |

| FastAPI | Backend API |

| Git and GitHub | Version control and collaboration |



\## 4. Repository Structure



```text

PathSnap/

│

├── api/                    # Backend API

├── configs/                # Feature, model and HMM configurations

│   ├── features.yaml

│   ├── model.yaml

│   └── hmm.yaml

│

├── experiments/

│   ├── p1/                 # Data foundation experiments

│   ├── p2/                 # Classifier experiments

│   └── p3/                 # HMM evaluation and map generation

│

├── maps/                   # Route visualization outputs

├── models/                 # Exported trained model

├── report/                 # Project report material

├── results/

│   ├── p2/                 # Classifier results

│   ├── p3/                 # HMM evaluation results

│   └── p4/                 # Application results

│

├── scripts/                # Utility scripts

├── src/

│   ├── eval/               # Evaluation metrics

│   ├── features.py         # Candidate feature generation

│   ├── geo.py              # Geographic geometry utilities

│   ├── hmm.py              # HMM and Viterbi decoder

│   ├── models.py           # Model implementations

│   ├── scoring.py          # Candidate scoring

│   └── \_p1\_data.py         # Shared trip and geometry utilities

│

├── tests/

│   └── test\_hmm.py         # HMM unit tests

│

├── ui/                     # Frontend interface

│

├── requirements.txt

├── requirements-experiments.txt

└── requirements-optional.txt

```



\## 5. Installation and Setup



\### Prerequisites



\- Python 3.11 or compatible Python version.

\- Git.

\- pip.

\- Access to the project dataset.

\- Windows, Linux or macOS.



\### Step 1 – Clone the repository



```bash

git clone https://github.com/sudhikshavaranasi/pt-72.git

cd pt-72

```



\### Step 2 – Create a virtual environment



Windows PowerShell:



```powershell

python -m venv .venv

.venv\\Scripts\\Activate.ps1

```



Linux/macOS:



```bash

python3 -m venv .venv

source .venv/bin/activate

```



\### Step 3 – Install dependencies



Install the core dependencies:



```bash

python -m pip install -r requirements.txt

```



For experiment-related dependencies:



```bash

python -m pip install -r requirements-experiments.txt

```



For optional visualization and application dependencies, install:



```bash

python -m pip install -r requirements-optional.txt

```



\### Step 4 – Verify the installation



```bash

python -c "import numpy, pandas; print('Dependencies installed successfully')"

```



\## 6. Dataset Structure



The dataset contains training trips and student-test trips.



Expected directory structure:



```text

Student folder/

└── Student folder/

&#x20;   ├── train/

&#x20;   │   ├── 00000001/

&#x20;   │   │   ├── 00000001.track

&#x20;   │   │   ├── 00000001.nodes

&#x20;   │   │   ├── 00000001.arcs

&#x20;   │   │   └── 00000001.route

&#x20;   │   └── ...

&#x20;   │

&#x20;   └── student\_test/

&#x20;       ├── 00000007/

&#x20;       │   ├── 00000007.track

&#x20;       │   ├── 00000007.nodes

&#x20;       │   └── 00000007.arcs

&#x20;       └── ...

```



\### Input files



| File | Description |

|---|---|

| `.track` | GPS observations containing longitude, latitude and timestamp |

| `.nodes` | Geographic coordinates of road-network nodes |

| `.arcs` | Directed connections between road-network nodes |

| `.route` | Reference road-segment sequence used for training/evaluation where available |



Reference routes must not be used to generate predictions.



\## 7. Running the Project



\### A. Run unit tests



Execute the HMM test suite:



```bash

python -m unittest discover -s tests -v

```



The tests cover:



\- Directed road graph connectivity.

\- Viterbi decoding and transition costs.

\- Empty candidate handling.

\- Unavailable transitions and sequence segmentation.

\- Stationary transition penalties.



\### B. Run P1 experiments



The P1 experiment script accepts the dataset and output directory:



```bash

python -m experiments.p1.p1 --data PATH\_TO\_DATASET --out results/p1

```



\### C. Run P2 experiments



P2 accepts the P1 output, raw dataset and optional experiment count:



```bash

python -m experiments.p2.p2 --p1 results/p1 --raw PATH\_TO\_DATASET --out results/p2 --trials 30

```



\### D. Run P3 experiments



P3 performs decoder tuning on training out-of-fold scores and evaluates the selected configuration on student-test trips.



```bash

python -m experiments.p3.p3 --raw PATH\_TO\_DATASET --p1 results/p1 --p2 PATH\_TO\_P2\_SCORE\_EXPORTS --out results/p3 --model shallow\_mlp\_tuned

```



The P3 experiment generates:



\- `selected\_config.yaml`

\- `training\_decoder\_sweep.csv`

\- `student\_test\_metrics.csv`

\- `student\_test\_summary.json`

\- `stress\_test\_metrics.csv`

\- Per-observation prediction files under `results/p3/points/`



The P3 experiment requires P1 reference alignment outputs and P2 candidate-score exports. The score-export directory must contain the corresponding training and student-test score files.



\## 8. HMM Map-Matching Methodology



PathSnap uses a Hidden Markov Model to estimate the most likely road-segment sequence for a GPS trajectory.



\### Emission probabilities



The candidate classifier provides a probability score for each candidate road segment.



These scores represent how likely a GPS observation is to correspond to a particular road segment.



\### Transition probabilities



The transition model evaluates whether movement between two candidate road segments is plausible.



It considers:



\- Road-network distance.

\- Straight-line GPS displacement.

\- Time elapsed between observations.

\- Directed road connectivity.

\- Stationary movement.

\- Abnormally high GPS speeds.



\### Viterbi decoding



The Viterbi algorithm combines emission and transition information to determine the most likely sequence of road segments.



The output includes classifier-only predictions and HMM-decoded predictions.



\### Handling difficult trajectories



The decoder handles:

\- Large temporal gaps.

\- Unreachable road transitions.

\- Missing candidate sets.

\- Stationary GPS observations.

\- GPS speed anomalies.



\## 9. Evaluation Metrics



The project evaluates map-matching performance using:



| Metric | Description |

|---|---|

| Top-1 Accuracy | Percentage of aligned GPS observations assigned the correct road segment |

| Candidate Miss Rate | Percentage of aligned observations where the correct segment was absent from the candidate set |

| Route Continuity | Percentage of consecutive distinct predicted road transitions that are valid in the directed road graph |

| Route Edit Distance | Difference between predicted and reference road-segment sequences |

| Normalized Route Edit Distance | Edit distance normalized by the larger sequence length |



The classifier-only predictions are compared with HMM-decoded predictions to measure the benefit of sequence-level inference.



\## 10. Outputs and Visualizations



Results are stored under the `results/` directory.



\- P1: Candidate generation and alignment outputs.

\- P2: Model comparison, grouped validation and optimization outputs.

\- P3: HMM tuning, student-test evaluation and stress-test metrics.

\- P4: Application-related outputs.



Interactive route visualizations and comparison outputs are stored under `maps/`.



\## 11. Troubleshooting



\### ModuleNotFoundError



Run Python modules from the repository root:



```bash

python -m experiments.p3.p3 --help

```



Use package-relative imports for modules within packages.



\### Missing dataset



Verify that the dataset directory contains both `train` and `student\_test`.



\### Missing model or score files



Verify that the trained model exists under `models/` and that the required P2 score exports are available before running P3 evaluation.



\### Dependency errors



Activate the virtual environment and reinstall the relevant requirements file.



\## 12. Contributors



PathSnap was developed collaboratively with responsibilities divided across:



\- P1 – Data foundation and candidate generation.

\- P2 – Candidate classification and model optimization.

\- P3 – HMM map matching, route evaluation and visualization.

\- P4 – Application development and integration.



\## 13. License



Add the applicable project license before distributing or reusing this repository.



\---



\*\*PathSnap – Turning noisy GPS observations into plausible road journeys.\*\*

