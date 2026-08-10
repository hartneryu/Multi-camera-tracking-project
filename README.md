# Multi-Camera Spatial Tracking on EPFL Terrace

This repository implements multi-camera pedestrian tracking on the EPFL
Terrace dataset. It processes four synchronized cameras, projects detections
onto a shared ground plane, fuses cross-camera observations, and assigns global
track identities over time.

The system uses separate detection, projection, fusion, and tracking modules:

```text
4 synchronized cameras
        ↓
YOLO11s person detection
        ↓
OSNet appearance embeddings
        ↓
Foot-point projection to the ground plane
        ↓
Cross-camera spatial fusion
        ↓
Kalman prediction + Hungarian assignment
        ↓
Global tracks in bird's-eye view
```


## How the pipeline works

### 1. Person detection

YOLO11s detects the `person` class independently in every camera. For a box
$(x_1,y_1,x_2,y_2)$, the bottom centre is used as the approximate point where
the person touches the ground:

$$
u_f = \frac{x_1+x_2}{2}, \qquad v_f = y_2.
$$

The bottom centre is used as the estimated ground-contact point.

### 2. Projection to a common coordinate system

The Tsai camera calibration maps each foot point $(u_f,v_f)$ to a metric
ground position $(X,Y)$. Detections from different images can then be compared
in metres instead of pixels.


### 3. Multi-camera fusion

Projected detections from different cameras are grouped when they are within
`1.2 m` of one another. A strong observation requires support from at least
two cameras. Its position and normalized OSNet embedding are obtained by
confidence-weighted averaging.

Single-camera observations cannot create a new global track. They may only
continue an already confirmed active track, which helps during short
occlusions without introducing many false identities.

### 4. Temporal association

Each track has a constant-velocity Kalman state:

$$
\mathbf{x} = [X,\ Y,\ V_X,\ V_Y]^T.
$$

The Kalman filter predicts the next ground position and its uncertainty.
Candidate track-observation pairs are gated using Mahalanobis distance. For
valid pairs, the assignment cost combines motion and OSNet cosine distance:

$$
C = 0.5\frac{d_M^2}{9.21} + 0.5\frac{d_A}{0.5}.
$$

The Hungarian algorithm finds the minimum-cost one-to-one assignment.

Active tracks are matched first. Lost tracks receive only the remaining
strong observations and use a stricter appearance rule. A track is confirmed
after three matches and retained for up to five seconds after disappearing.

## Selected configuration

Pipeline parameters are fixed in the code and are not exposed through the
command line.

| Component | Selected value |
| --- | --- |
| Detector | YOLO11s |
| Detection confidence | 0.25 |
| Detector image size | 640 |
| Processing rate | 5 FPS |
| Fusion distance | 1.2 m |
| Track initialization | At least 2 cameras |
| Appearance model | OSNet x0.25 pretrained on MSMT17 |
| Active association | 0.5 motion + 0.5 appearance |
| Track confirmation | 3 matches |
| Lost-track retention | 5 seconds |
| Evaluation match distance | 0.5 m |

## Installation

This project was done on a local nvidia rtx pro 500 blackwell generation laptop gpu. Thus
an NVIDIA GPU with CUDA support is required by the current implementation.

```powershell
git clone https://github.com/hartneryu/Multi-camera-tracking-project.git
cd Multi-camera-tracking-project

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dataset and model files

Download the [EPFL Terrace dataset](https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/)
and arrange the required files as follows:

```text
data/EPFL-Terrace/
├── annotations/
│   └── gt_terrace1.txt
├── calibration/
│   └── tsai/
│       ├── terrace-tsai-c0.xml
│       ├── terrace-tsai-c1.xml
│       ├── terrace-tsai-c2.xml
│       └── terrace-tsai-c3.xml
└── videos/
    ├── terrace1-c0.avi
    ├── terrace1-c1.avi
    ├── terrace1-c2.avi
    └── terrace1-c3.avi
```

Place the MSMT17-pretrained OSNet weights at:

```text
models/osnet_x0_25_msmt17.pth
```

The code expects the YOLO weights as `yolo11s.pt`. Ultralytics downloads them
on first use when they are not already present. Dataset files, generated
outputs, and model weights are excluded from Git.

## Usage

Commands write generated images, videos, and CSV files to `outputs/`. The
folder is created automatically.

### Inspect the synchronized cameras and annotations

```powershell
python main.py --mode view --frame 2000 --output outputs\terrace_view.png
```

### Validate the camera calibration

```powershell
python main.py --mode projection --frame 2000 --output outputs\terrace_projection.png
```

This draws the projected ground locations in every camera and reports the
image-to-ground-to-image round-trip error.

### Inspect detection, projection, and fusion

```powershell
python main.py --mode yolo --frame 2000 --output outputs\terrace_yolo.png
```

The resulting image shows camera detections, individual projected points,
fused observations, and official ground truth in separate layers.

### Generate a tracking video

```powershell
python main.py --mode track --start-frame 2500 --seconds 60 --output outputs\terrace_tracking.mp4
```

In the bird's-eye view, each confirmed track is drawn with its ID, trajectory,
and a covariance ellipse. Under the Kalman filter's Gaussian assumption, the
ellipse contains 95% of the predicted two-dimensional position distribution. If
$P_{xy}=P[0:2,0:2]$ is the position covariance matrix and $\lambda_1,
\lambda_2$ are its eigenvalues, the ellipse semi-axes are:

$$
a_i=\sqrt{5.991\lambda_i}.
$$

The value `5.991` is the 95% chi-square boundary for two dimensions. The
eigenvectors determine the ellipse orientation. The ellipse grows while a
track is predicted without a matching observation and usually contracts after
a Kalman measurement update. It is not a person footprint or a ground-truth
region.

The displayed 95% ellipse is also distinct from the association gate. Track
matching uses a 99% Mahalanobis gate, $d_M^2\leq9.21$, which is not drawn.

### Evaluate the complete annotated sequence

```powershell
python main.py --mode evaluate --start-frame 0 --seconds 200.2 --output outputs\terrace_evaluation.csv
```

The evaluator also creates `terrace_evaluation_frames.csv` with per-frame
counts and errors.

## Evaluation

The complete evaluation covers all **201 labeled timestamps** and **1,023
ground-truth person instances**. Terrace is annotated once every 25 source
frames; the pipeline processes intermediate timestamps at 5 FPS.

| Metric | Result |
| --- | ---: |
| MOTA | **93.16%** |
| MOTP / mean ground error | **16.00 cm** |
| IDF1 | **77.39%** |
| Ground-plane HOTA | **50.21%** |
| DetA | **57.87%** |
| AssA | **43.58%** |
| Identity switches | **6** |
| Fragmentations | **22** |
| False positives | **22** |
| False negatives | **42** |

### Ground-plane HOTA definition

Standard HOTA for pedestrian tracking normally uses bounding-box IoU as its
localization similarity. Terrace supplies ground-plane points rather than
ground-truth image boxes, so this evaluator replaces IoU with a distance-based
similarity:

$$
s(g,p)=\max\left(0,1-\frac{d(g,p)}{0.5}\right),
$$

where $d(g,p)$ is the Euclidean ground distance in metres. This gives
similarity `1.0` at zero error, `0.5` at `0.25 m`, and `0.0` at `0.5 m` or
more.

The evaluator follows the HOTA threshold sweep:

$$
\alpha\in\{0.05,0.10,\ldots,0.95\}.
$$

At threshold $\alpha$, a match is valid when $s(g,p)\geq\alpha$. Therefore
each threshold corresponds to a different maximum ground error:

$$
d(g,p)\leq0.5(1-\alpha)\ \text{metres}.
$$

For example, $\alpha=0.5$ permits `0.25 m`, while $\alpha=0.95$ permits only
`0.025 m`. At every threshold, the evaluator calculates:

$$
\mathrm{DetA}_{\alpha}
=\frac{TP}{TP+FN+FP},
\qquad
\mathrm{HOTA}_{\alpha}
=\sqrt{\mathrm{DetA}_{\alpha}\mathrm{AssA}_{\alpha}}.
$$

`AssA` measures whether each matched ground-truth identity remains associated
with the same tracker identity over time. The reported HOTA, DetA, and AssA
values are the means over all 19 thresholds. Global identity-pair alignment
and per-frame Hungarian matching follow the TrackEval calculation; only the
localization similarity is replaced.

For this reason, the reported **ground-plane HOTA** is valid for comparing
versions evaluated by this repository, but it is not directly comparable to
box-IoU HOTA reported by MOTChallenge or by another implementation.



## Repository structure

| File | Purpose |
| --- | --- |
| `main.py` | CLI and end-to-end pipeline orchestration |
| `terrace.py` | Dataset paths, synchronization, and ground truth |
| `detector.py` | Batched YOLO person detection |
| `appearance.py` | Person crops and OSNet embeddings |
| `projection.py` | Tsai calibration and image/world transformations |
| `fusion.py` | Cross-camera spatial clustering |
| `tracker.py` | Kalman filter, association, and track lifecycle |
| `evaluation.py` | CLEAR MOT, IDF1, and ground-plane HOTA |
| `visualization.py` | Camera panels and bird's-eye-view rendering |


## Acknowledgements

- [EPFL CVLab Terrace dataset](https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Torchreid / OSNet](https://github.com/KaiyangZhou/deep-person-reid)
- [HOTA and TrackEval](https://github.com/JonathonLuiten/TrackEval)

The dataset and pretrained models remain subject to their respective licenses
and terms of use.
