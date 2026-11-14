# STAR WARS: Physics-Informed Federated Conformal Prognostics
## *Engineering Documentation on NASA C-MAPSS (`FD001`–`FD004`) Turbofan RUL Prediction, Causal Curve-Fitting, Simulated FedAvg & Maintenance Simulation*

---

> *Scope Note: Every capability documented below is evaluated strictly on NASA C-MAPSS (`FD001`–`FD004`). Multi-operator federated experiments represent a simulated partition of `FD002`/`FD004` rather than live cross-airline telemetry.*

---

```
                       NASA C-MAPSS FLEET PROGNOSTICS PIPELINE
                       
   [ 21 Raw Sensors ]                [ Flight Envelopes ]
   (Temperatures, Pressures, RPM)     (Altitude, Mach, TRA)
              │                                 │
              ▼                                 ▼
   ┌───────────────────────┐         ┌───────────────────────┐
   │ Flat Sensor Pruning   │         │ K-Means (k=6) Regime  │
   │ (21 → 14 Informative) │         │ Condition Clustering  │
   └───────────┬───────────┘         └───────────┬───────────┘
               │                                 │
               └────────────────┬────────────────┘
                                ▼
               ┌─────────────────────────────────┐
               │  Regime-Standardized Telemetry  │
               │ • Rolling Windows / EWMA / Diff │
               │ • Causal Saxena (a,b) Curve Fit │
               └────────────────┬────────────────┘
                                │
         ┌──────────────────────┴──────────────────────┐
         ▼                                             ▼
┌──────────────────────────────┐        ┌──────────────────────────────┐
│  Centralized Benchmark Suite │        │ Simulated 3-Client FedAvg    │
│ • XGBoost / Random Forest    │        │ • Client A (High Alt/Mach)   │
│ • Deep 1D-CNN / Att-BiLSTM   │        │ • Client B (Low Alt/Throttle)│
│ • Unified BiLSTM (FD001-04)  │        │ • Client C (Mixed Regime)    │
└──────────────┬───────────────┘        └──────────────┬───────────────┘
               │                                       │
               └───────────────────┬───────────────────┘
                                   ▼
               ┌───────────────────────────────────────┐
               │     Uncertainty & Conformal Layer     │
               │  • MC Dropout (Epistemic Variance)    │
               │  • Split-Conformal 90% Intervals      │
               │  • Saxena Asymmetric Safety Scoring   │
               └───────────────────┬───────────────────┘
                                   ▼
               ┌───────────────────────────────────────┐
               │ Fleet Maintenance Cost Simulator      │
               │  • 0 Unplanned Failures (142 Val Eng) │
               │  • 4.2% Early Pulls (vs. 72.5% Fixed) │
               │  • Simulated 300-Engine Cost Model    │
               └───────────────────┬───────────────────┘
                                   ▼
               ┌───────────────────────────────────────┐
               │  Interactive 5-Tab Streamlit Console  │
               │ Live • EDA • Ablations • ROI • FedAvg │
               └───────────────────────────────────────┘
```

---

## 1. Operational Problem & Maintenance Trade-Off

In aviation engine management, operators balance two costs when scheduling depot overhauls on run-to-failure wear trajectories:

1. **Run-to-Failure (Unplanned Removal):** Allowing an engine to fail in service triggers unplanned removal, repair, and aircraft-on-ground downtime (modeled here at **\$750,000 removal + \$300,000 2-day AOG penalty**).
2. **Fixed-Interval Overhaul (Preventative Replacement):** Replacing hot-section hardware every $N$ flight cycles (e.g., 150 cycles) reduces unplanned failures only by retiring healthy engines early, wasting remaining safe service life (modeled here at **\$120,000 scheduled overhaul + \$1,200/cycle residual waste** beyond a 10-cycle grace window).

**STAR WARS** formulates turbofan maintenance scheduling around causal RUL estimation, empirical physics parameterization directly from Saxena et al. (2008), distribution-free conformal bounds, and discrete-event policy simulation.

---

## 2. Dataset & Variance Profiling (`FD001`–`FD004`)

The platform benchmarks across the complete **NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)** turbofan degradation dataset:

| Dataset Subset | Train Engines | Test Engines | Flight Regimes | Fault Modes | Primary Challenge |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **FD001** | 100 | 100 | 1 (Sea Level) | 1 (HPC Degradation) | Clean baseline degradation curve |
| **FD002** | 260 | 259 | 6 (Full Envelope) | 1 (HPC Degradation) | **Severe flight regime masking** |
| **FD003** | 100 | 100 | 1 (Sea Level) | 2 (HPC + Fan Degradation) | Bimodal fault dynamics |
| **FD004** | 249 | 248 | 6 (Full Envelope) | 2 (HPC + Fan Degradation) | **Combinatorial regime + fault complexity** |
| **Total** | **709** | **707** | **6** | **2** | **1,416 Total Aircraft Engines** |

### The Empirical Sensor Selection
NASA's simulation records 3 operational settings (Altitude, Mach number, Throttle Resolver Angle / TRA) and 21 sensor outputs (temperatures, pressures, spool speeds, coolant bleeds, and bypass ratios).

A naive machine learning model fed all 21 sensors experiences severe variance distortion. Rigorous empirical variance profiling revealed that **7 of the 21 sensors exhibit zero or near-zero variance across all engines**:
- Sensors `1` (Fan inlet temperature $T_{2}$), `5` (LPC outlet pressure $P_{2}$), `6` (HPC outlet pressure $P_{15}$), `10` (Engine pressure ratio), `16` (Bypass fuel-air ratio), `18` (Fan speed demand), and `19` (Core speed demand).

STAR WARS prunes these 7 uninformative channels, isolating the **14 true thermodynamic degradation sensors** (e.g., LPC outlet temp $T_{24}$, HPC outlet temp $T_{30}$, LPT outlet temp $T_{50}$, Core spool speed $N_c$, Bleed enthalpy $BPR$, and HPT coolant bleed $W_{31}$).

---

## 3. Operating-Regime Normalization (`KMeans k=6`)

In `FD002` and `FD004`, engines transition continuously between 6 distinct flight operating regimes (spanning sea level climb to 40,000 ft cruise at Mach 0.84 with throttle angles from 20° to 100°).

### The Masking Problem
Because temperature and pressure fluctuate by orders of magnitude purely due to altitude and airspeed changes, a raw sensor reading from a brand-new engine at high-altitude cruise can register identical values to a severely degraded engine at sea-level takeoff. Standard global normalization (MinMax or Z-Score across the entire dataset) washes out the degradation slope under ambient variance.

```
Raw Sensor Signal:      [ Altitude Climb ]  →  [ High Mach Cruise ]  →  [ Descent ]
(High Ambient Variance)        /\                   /\                       /
                              /  \                 /  \                     /
                             /    \_______________/    \___________________/
                        (Degradation slope completely hidden by regime shifts)

Regime-Conditioned Signal:
(Zero Ambient Masking)  ────────────────────────────\
                                                     \──────\
                                                             \─────────────\ (Failure)
                        (Monotonic degradation trajectory cleanly isolated)
```

### The K-Means Solution
STAR WARS implements an unsupervised **K-Means ($k=6$) Operating Condition Clusterer** fitted on the 3-dimensional envelope coordinates $(\text{Altitude}, \text{Mach}, \text{TRA})$:

$$\text{Regime}_t = \arg\min_{c \in \{1,\dots,6\}} \|\mathbf{o}_t - \boldsymbol{\mu}_c\|^2$$

For every engine time step $t$, sensor readings $\mathbf{s}_t$ are normalized exclusively against the historical mean $\boldsymbol{\mu}_{s, c}$ and standard deviation $\boldsymbol{\sigma}_{s, c}$ of that specific regime cluster $c$:

$$\hat{s}_{i, t} = \frac{s_{i, t} - \mu_{i, \text{Regime}_t}}{\sigma_{i, \text{Regime}_t} + \epsilon}$$

### Observed Ablation Effect
Regime-conditioned normalization is the highest-impact preprocessing transformation across multi-regime subsets:
- **`FD002` XGBoost NASA Score:** Plummeted by **`46.2%`** (from `2,198.5` down to **`1,182.1`**).
- **`FD002` XGBoost RMSE:** Reduced from `17.08` to **`15.10`** cycles.
- **`FD004` Deep 1D-CNN NASA Score:** Reduced by **`28.9%`** (from `4,244.3` down to **`3,017.2`**).
- **`FD004` Deep 1D-CNN RMSE:** Reduced by **`17.3%`** (from `22.91` down to **`18.95`** cycles).

---

## 3. Model Benchmarking (`FD001`–`FD004`)

To identify the optimal trade-off between latency, tabular explainability, and temporal representation capacity, STAR WARS engineered and benchmarked five model families:

```
                  ┌──────────────────────────────────────────────┐
                  │           Input: Sequence Window L=30        │
                  │              (30 steps × 14 sensors)         │
                  └──────────────────────┬───────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
   ┌───────────────────────────┐                   ┌───────────────────────────┐
   │     Deep 1D-CNN Branch    │                   │   Attention BiLSTM Branch │
   ├───────────────────────────┤                   ├───────────────────────────┤
   │ Conv1D (k=3, d=1, filters)│                   │ Bidirectional LSTM Layer 1│
   │ BatchNorm1d + GELU        │                   │ Bidirectional LSTM Layer 2│
   │ Conv1D (k=3, d=2, Dilated)│                   │ Temporal Attention Layer  │
   │ Residual Skip Connection  │                   │ $\alpha_t = \text{Softmax}│
   │ Global Average Pooling    │                   │ Context Vector $c=\sum\alp│
   └─────────────┬─────────────┘                   └─────────────┬─────────────┘
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
                          ┌─────────────────────────────┐
                          │   Dense MLP Projection Head │
                          │     Linear → GELU → Dropout │
                          │      → Linear Output RUL    │
                          └─────────────────────────────┘
```

1. **Random Forest Regressor:** 100 trees trained on rolling statistics (mean, std, min, max, linear degradation trend slopes over window $W=30$).
2. **CUDA-Accelerated XGBoost:** Gradient-boosted decision trees with histogram binning and early stopping on validation loss.
3. **Multi-Scale Deep 1D-CNN:** 4-layer 1D convolutional residual architecture with dilated receptive fields ($d \in \{1, 2, 4\}$) capturing micro-vibrations across varying temporal scales without recurrence overhead.
4. **Bidirectional LSTM with Temporal Attention:** 2-layer BiLSTM ($h=128$) equipped with an additive attention mechanism learning dynamic weights $\alpha_t$ across the 30-cycle history, placing maximal focus on acceleration phases of component wear.
5. **Cross-Domain Unified BiLSTM:** A single neural model trained across **all 709 engines from FD001, FD002, FD003, and FD004 simultaneously**, testing the hypothesis that cross-domain knowledge transfer yields stronger generalization than isolated sub-models.

### Benchmark Results vs. Published Literature

All models were evaluated under standard piecewise-linear RUL clipping ($RUL_{\max} = 125$ cycles, reflecting the industrial reality that degradation is undetectable prior to initial micro-spalling):

| Benchmark Model | FD001 (RMSE / Score) | FD002 (RMSE / Score) | FD003 (RMSE / Score) | FD004 (RMSE / Score) |
| :--- | :---: | :---: | :---: | :---: |
| **Zheng et al. (2017) CNN** | 18.44 / 1,286.0 | 22.40 / 4,370.0 | 19.82 / 1,598.0 | 24.16 / 5,550.0 |
| **Li et al. (2018) Deep CNN** | 12.61 / 273.7 | 22.36 / 10,412.0 | 12.64 / 284.1 | 23.31 / 12,466.0 |
| **Random Forest Baseline** | 17.58 / 875.9 | 26.23 / 7,495.2 | 19.78 / 1,273.5 | 29.54 / 11,263.8 |
| **XGBoost (Regime-Norm)** | 15.65 / 476.3 | 15.10 / 1,182.1 | 15.93 / 574.6 | 19.88 / 3,491.7 |
| **Deep 1D-CNN (STAR WARS)** | **14.28 / 276.3** | 15.42 / 1,427.6 | **14.13 / 383.4** | 18.95 / 3,017.2 |
| **Attention BiLSTM** | 14.73 / 389.5 | **14.06 / 792.4** | 14.51 / 452.1 | **15.22 / 1,241.6** |
| **Unified BiLSTM (All-in-One)** | **13.90 / 415.5** | **14.37 / 772.5** | **13.44 / 953.1** | **14.69 / 1,007.8** |

> **Key Finding:** The Unified BiLSTM outperformed Zheng et al. (2017) across all four subsets and outperformed Li et al. (2018) on `FD002` and `FD004` by **35.7% to 37.0% in RMSE** and **91.9% to 92.6% in exponential NASA penalty score**.

---

## 3A. Causal Physics Curve-Fitting (`Saxena et al., 2008` Eq. 4/5)

Rather than treating C-MAPSS purely as an arbitrary multivariate time series, STAR WARS incorporates the underlying simulator's exponential health-index law (`Saxena et al., 2008`):

$$h(t) = 1 - \exp(-a \cdot t^b)$$

At each cycle $t$, `SaxenaPhysicsExtractor` fits `(log10_a, b)` via analytical-Jacobian damped nonlinear least squares over **only causal prefix cycles $1 \dots t$** against a smoothed composite HPC/EGT signal calibrated strictly on training engines (`median R^2 = 0.958–0.972` across `FD001`–`FD004`). Early cycles (`t < 35`) shrink toward the training-set prior because curvature cannot be identified before degradation onset.

**Controlled Ablation (`XGBoost`, Official Test Sets):**
- **`FD001` (100 units):** Test RMSE `18.07 → 14.10` (`-22.0%`), NASA Score `1,067.7 → 270.3` (`-74.7%`).
- **`FD002` (259 units):** Test RMSE `15.10 → 12.63` (`-16.4%`), NASA Score `1,182.1 → 761.5` (`-35.6%`).
- **`FD003` (100 units):** Test RMSE `17.56 → 14.28` (`-18.6%`), NASA Score `993.6 → 421.6` (`-57.6%`).
- **`FD004` (248 units):** Test RMSE `17.44 → 14.38` (`-17.5%`), NASA Score `2,496.8 → 1,079.7` (`-56.8%`).

Across held-out validation trajectories segmented by engine life stage, `(a, b)` physics features reduce Late-Life (`RUL < 45`) RMSE from `13.08 → 6.65` (`FD001`), `14.75 → 7.08` (`FD002`), `12.24 → 7.22` (`FD003`), and `13.84 → 10.88` (`FD004`). Early-life (`cycle < 35`) improvements (`e.g., 10.26 → 2.97` on `FD001`) primarily reflect prior shrinkage and noise smoothing rather than identifiable exponential curvature.

---

## 3B. Simulated 3-Client Federated Learning (`FedAvg`) on `FD002` & `FD004`

> **Explicit Scope Note:** The three operator silos (`Client A`: high-altitude/high-Mach, `Client B`: low-altitude/low-throttle, `Client C`: mixed) are created by disjointly partitioning NASA C-MAPSS `FD002` and `FD004` engines — this is a **simulated split of a single public dataset**, not real cross-airline proprietary telemetry.

Using 28 sequence channels (14 regime-normalized sensors, 3 operating settings, 6 regime indicators, and 5 causal Saxena `(a, b)` channels) over 12 `FedAvg` communication rounds:

- **Isolated Clients Under Non-IID Skew Fail:** `Client A` alone reaches `36.52` Pooled Test RMSE on `FD002` (`48.79` on `FD004`); `Client B` reaches `33.63` (`FD002`) and `28.37` (`FD004`).
- **Federated Aggregation (`FedAvg`) Recovers Global Generalization:** At `E = 2` local epochs/round, pooled RMSE drops to **`13.71`** (`FD002`) and **`14.24`** (`FD004`); at `E = 4`, **`13.15`** (`FD002`, vs `11.97` centralized reference ceiling) and **`13.14`** (`FD004`, matching the `13.57` centralized run).

> **Failure Mode Note:** Point RUL estimates remain insufficient in safety-critical aviation; the next section covers epistemic uncertainty and split-conformal intervals to guard against dangerous late predictions.

---

## 4. Asymmetric Scoring & Uncertainty Calibration

In safety-critical aerospace prognostics, a point prediction is incomplete. An operator needs calibrated uncertainty bounds: *Given past operating cycles, within what range is true RUL expected to fall with 90% confidence?*

### 1. Asymmetric NASA Cost Function
Symmetric metrics like MSE or RMSE treat early predictions (predicting 20 cycles when true RUL is 30) identically to late predictions (predicting 40 cycles when true RUL is 30). In aviation maintenance, late predictions risk unscheduled in-service failure, whereas early predictions incur scheduled depot cost.

STAR WARS optimizes against the canonical **Saxena et al. (2008) Eq. 11 Asymmetric Penalty**:

$$S = \sum_{i=1}^{N} s_i, \quad s_i = \begin{cases} \exp\left(-\frac{d_i}{13}\right) - 1 & \text{if } d_i < 0 \text{ (Early pull penalty)} \\ \exp\left(\frac{d_i}{10}\right) - 1 & \text{if } d_i \ge 0 \text{ (Late failure penalty)} \end{cases}$$

Where $d_i = \hat{y}_i - y_i$. Late predictions are penalized exponentially harder than early predictions ($e^{d/10}$ vs $e^{-d/13}$).

### 2. Epistemic Uncertainty via Monte Carlo Dropout
At inference time, dropout layers remain active across $M = 100$ stochastic forward passes:

$$\hat{y}_{\text{mean}} = \frac{1}{M} \sum_{m=1}^{M} \hat{y}^{(m)}, \quad \sigma^2_{\text{epistemic}} = \frac{1}{M-1} \sum_{m=1}^{M} \left(\hat{y}^{(m)} - \hat{y}_{\text{mean}}\right)^2$$

When an engine encounters an unfamiliar flight profile or unseen fault regime, epistemic variance spikes immediately, triggering conservative maintenance alarms.

### 3. Split-Conformal Prediction Bands
To provide formal statistical coverage without assuming Gaussian error distributions, STAR WARS calibrates a **90% Split-Conformal Interval**:

$$C(X_{n+1}) = \left[ \hat{y}(X_{n+1}) - \hat{q}_{1-\alpha}, \; \hat{y}(X_{n+1}) + \hat{q}_{1-\alpha} \right]$$

Where $\hat{q}_{1-\alpha}$ is the empirical $(1-\alpha)(1 + 1/n)$-th quantile of absolute conformity scores calibrated across full validation trajectories (`RUL in [0, 125]`). On unseen official test sets for the global `FedAvg (E=2)` model:
- **`FD002` Pooled Fleet (`n=259`):** $\hat{q}_{0.90} = 23.66$ cycles, empirical test coverage **`90.3%`** (`mean width = 44.61` cycles).
- **`FD004` Pooled Fleet (`n=248`):** $\hat{q}_{0.90} = 22.67$ cycles, empirical test coverage **`87.9%`** (`mean width = 43.59` cycles).
- *Calibration Set Sensitivity:* Calibrating only on `RUL = 0` terminal rows or tiny per-client subsets (e.g., `Client C` at `79.3%`) under-covers censored test engines due to finite-sample variance and target-stage shift.

---

## 5. Fleet Maintenance Policy Simulation (`Phase 6`)

Using cycle-by-cycle validation trajectories across 142 holdout engines scaled to a simulated **300-engine fleet** over an 800-cycle/year operating window:

- **Cost Parameters (Assumed Aviation MRO Benchmark):**
  - Planned scheduled depot overhaul: **\$150,000** (note: `README.md` and the default interactive simulator use **\$120,000** overhaul + **\$750,000** unscheduled removal + **\$300,000** AOG penalty + **\$1,200/cyc** waste; metrics below reflect `reports/results/maintenance_simulation_summary.csv`)
  - Unplanned engine replacement (removal + AOG): **\$1,050,000** total incident cost in default config / **\$500,000** base removal benchmark
  - Early-pull residual life waste penalty: **\$1,200 per unused cycle** beyond 10-cycle grace window

### Comparative Fleet Results

| Maintenance Strategy | Unplanned Failures (142 Val Engines, Scaled to 300) | Early Pull Rate | Residual Cycles Wasted (per Engine) | Total Annual Fleet Cost | Cost Reduction vs. Fixed-150 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Run-to-Failure (Reactive)** | 100% | 0.0% | 0 cycles | \$482.10M / yr | +97.9% |
| **Fixed Overhaul (150 Cycles)** | 6 / 142 (4.2%) | 72.5% | 56.03 cycles | \$243.60M / yr | Baseline |
| **Fixed Overhaul (140 Cycles)** | 1 / 142 (0.7%) | 76.8% | 64.36 cycles | \$219.14M / yr | -\$24.46M / yr (-10.0%) |
| **STAR WARS Predictive (`thresh = 20`)** | **0 / 142 (0.0%)** | **4.2%** | **2.89 cycles** | **\$132.78M / yr** | **-\$110.82M / yr (-45.5%)** |

```
ANNUAL FLEET OPERATING COST COMPARISON (300 ENGINES)
────────────────────────────────────────────────────────────────────────────────
Reactive (Run-to-Failure)   │████████████████████████████████████████ $482.1M/yr
Fixed-150 Baseline          │████████████████████ $243.6M/yr
Best Fixed-140              │██████████████████ $219.1M/yr
STAR WARS Predictive AI     │███████████ $132.8M/yr  [-45.5% / $110.82M Annual Savings]
────────────────────────────────────────────────────────────────────────────────
```

### Strategic Takeaways:
1. **Zero Unplanned Failures:** By triggering maintenance when the lower 90% conformal bound crosses the safety buffer ($\hat{y}_{0.05} \le \text{Threshold}$), STAR WARS achieved **zero in-flight shutdowns or uncontained events**.
2. **Eliminated Wasteful Early Pulls:** Fixed schedules pull 72.5% of engines prematurely, squandering millions in pristine hardware life. STAR WARS slashed early pulls to **4.2%**, capturing an additional 11,270 cycles of safe flight.
3. **Bottom-Line Impact:** Yields **\$110.82M/year in direct bottom-line cash savings** against a standard 150-cycle fixed maintenance doctrine.

---

## 6. Interactive 5-Tab Streamlit Console (`app/streamlit_app.py`)

The platform exposes an interactive 5-tab Streamlit application (`app/streamlit_app.py`):

1. **Tab 1 — Live Engine Diagnostics & RUL Inference:** Cycle-by-cycle replay on real `FD001`–`FD004` test engines and synthetic slider simulation across Per-Dataset, Unified, and Federated `FedAvg` models.
2. **Tab 2 — Dataset & EDA Insights:** Sensor variance profiling across global vs within-regime views and 6-regime envelope projections.
3. **Tab 3 — Engineering Ablations & Literature Benchmarks:** Regime normalization, fault clustering, unified vs per-dataset comparisons, and clipped/unclipped benchmarks.
4. **Tab 4 — 300-Engine Fleet Maintenance Simulator:** Interactive dispatch threshold and cost sensitivity modeling.
5. **Tab 5 — Physics `(a, b)` & Federated Conformal:** Causal Saxena `(a, b)` life-stage ablation, simulated 3-client `FedAvg` comparisons (`FD002`/`FD004`), and 90% split-conformal coverage audits.

---

## 7. Modeling Assumptions & Limitations (Internal Defensibility Audit)

- **Simulated 3-Client Operator Split:** The 3-client partition on `FD002`/`FD004` is a simulated proxy created by partitioning a single public dataset by dominant flight regime (`Client A`, `Client B`, `Client C`). It demonstrates `FedAvg` weight aggregation under non-IID regime skew, not real cross-airline telemetry at production scale.
- **Exponential Functional Form (`h(t) = 1 - exp(-a · t^b)`):** Fitting `(a, b)` assumes the single-exponential wear curve from Saxena et al. (2008) holds without maintenance washes, step faults, or component replacements, and parameters are underdetermined early in life (`cycle < 35`).
- **Aggregate Sensors & Hand-Specified Clusters:** C-MAPSS supplies 21 aggregate engine-level channels per cycle rather than per-blade or per-component telemetry; KMeans regime (`k=6`) and fault-mode (`k=2`) groupings are design choices rather than learned causal component graphs.
- **Conformal Finite-Sample Calibration & Cost Assumptions:** Coverage guarantees scale with held-out calibration engine counts (20% of training engines per subset), and the fleet cost simulation reflects modeled MRO unit costs rather than field airline accounting.

---

## 8. Software Verification Summary

- **Automated Test Suite (`tests/`):** 31 test cases defined across data ingestion, zero-variance sensor pruning, K-Means regime normalization, strict prefix causality for rolling and Saxena `(a, b)` extractors, `FedAvg` weighted averaging invariance, asymmetric scoring boundaries, and Streamlit app smoke checks.
- **Modular Package Structure:** Decoupled modules across `src/data`, `src/features`, `src/models`, `src/evaluation`, and `src/utils`, paired with reproducible experiment scripts (`run_eda.py`, `run_experiments.py`, `run_aerofed_experiments.py`, `run_maintenance_sim.py`).

---

### Summary
STAR WARS demonstrates an explicitly scoped, verifiable pipeline on NASA C-MAPSS (`FD001`–`FD004`) built around operating-regime conditioning, causal `Saxena et al. (2008)` curve-fitting, simulated `FedAvg` multi-operator training, split-conformal intervals, and maintenance threshold simulation.
