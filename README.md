# Turbofan Engine RUL Prediction & Fleet Maintenance Intelligence

**End-to-End Predictive Maintenance on NASA C-MAPSS (`FD001`–`FD004`) | Pure Python, PyTorch, XGBoost & Streamlit**

---

## Business Framing & Financial Impact

Unscheduled jet engine removals cost **$500K–$1M+ per incident**, and Aircraft-on-Ground (AOG) delays can reach **$150K/day** ($300K over a 2-day outage). Shifting from static fixed-interval overhauls to Remaining Useful Life (RUL)-driven predictive maintenance is estimated across aviation MRO benchmarks to cut fleet maintenance spend by **10–40%** — tens of millions of dollars annually for a commercial fleet.

Evaluating our trained sequence models across **142 held-out run-to-failure validation engines** scaled to a **300-engine fleet** (800 flight cycles/engine/year; $120K scheduled overhaul, $750K unscheduled removal + $300K AOG penalty, $1,200/cycle wasted-life cost beyond a 10-cycle grace window):

| Maintenance Policy | Unplanned Failures (142 Engines) | Failures Avoided vs Fixed-150 | Engines Pulled Early | Avg Wasted Cycles | Annual 300-Engine Fleet Cost | Annual Savings vs Baseline | Cost Reduction |
|---|---|---|---|---|---|---|---|
| **Fixed-Interval (150 cycles)** | 6 (4.2%) | — | 103 (72.5%) | 56.03 | **$243.60M/yr** | — | — |
| **Best Fixed-Interval (140 cycles)** | 1 (0.7%) | 5 | 109 (76.8%) | 64.36 | **$219.14M/yr** | — | — |
| **Predictive RUL (`threshold = 20` cyc)** | **0 (0.0%)** | **6 (100% eliminated)** | **6 (4.2%)** | **2.89** | **$132.78M/yr** | **$110.82M/yr** *(vs Fixed-150)* / **$86.36M/yr** *(vs Best-Fixed 140)* | **45.5%** *(vs Fixed-150)* / **39.4%** *(vs Best-Fixed 140)* |

---

## Engineering Highlights

1. **All 4 NASA C-MAPSS Subsets (`FD001`–`FD004`)**: Built and benchmarked across single-condition, multi-condition (6 flight regimes), single-fault, and dual-fault (HPC + Fan) fleets — rather than cherry-picking `FD001` alone.
2. **Leakage-Free Within-Regime Normalization (`KMeans k=6`)**: Exposed and solved the multi-operating-condition masking problem on `FD002`/`FD004`, cutting `FD004` Deep1DCNN RMSE by **17.3%** (`22.91 → 18.95`) and NASA Score by **28.9%** (`4,244.3 → 3,017.2`), and `FD002` XGBoost NASA Score by **46.2%** (`2,198.5 → 1,182.1`).
3. **Asymmetric Safety Scoring (Saxena et al., 2008 Eq. 11)**: Evaluated every model on both RMSE and the NASA asymmetric exponential score alongside **MC Dropout + split-conformal 90% prediction intervals**.
4. **Unified Multi-Dataset Modeling**: Trained a single `Unified BiLSTM` across all 709 training engines (`FD001`–`FD004` with dataset & regime one-hot features) that **outperforms Zheng et al. (2017) on all 4 subsets** (`13.90`, `14.37`, `13.44`, `14.69` RMSE) and **beats Li et al. (2018) on `FD002`/`FD004` by 35.7%–37.0% RMSE and 91.9%–92.6% NASA Score**.

---

## Dataset Structure & Difficulty Progression

The NASA C-MAPSS turbofan degradation benchmark consists of 26-column run-to-failure multivariate sensor trajectories (`unit_id`, `cycle`, 3 operational settings, 21 sensors). Difficulty increases strictly as **`FD001 < FD003 < FD002 < FD004`**:

| Subset | Train Engines | Test Engines | Train / Test Rows | Operating Conditions | Fault Modes | Train Cycles (Min–Max / Mean) | Test Cycles (Min–Max) | Test True RUL (Min–Max / Mean) |
|---|---|---|---|---|---|---|---|---|
| **FD001** | 100 | 100 | 20,631 / 13,096 | 1 (Sea Level) | 1 (HPC) | 128–362 (206.31) | 31–303 | 7–145 (75.52) |
| **FD003** | 100 | 100 | 24,720 / 16,596 | 1 (Sea Level) | 2 (HPC + Fan) | 145–525 (247.20) | 38–475 | 6–145 (75.32) |
| **FD002** | 260 | 259 | 53,759 / 33,991 | 6 (Altitude / Mach / TRA) | 1 (HPC) | 128–378 (206.77) | **21**–367 | 6–194 (81.19) |
| **FD004** | **249** | **248** | 61,249 / 41,214 | 6 (Altitude / Mach / TRA) | 2 (HPC + Fan) | 128–543 (245.98) | **19**–486 | 6–195 (86.55) |

> **Data Engineering Notes:**
> - **Actual FD004 Unit Counts:** NASA's original `readme.txt` contains a well-known typo swapping 248/249 on `FD004`; the actual raw files contain **249 training engines** and **248 test engines**.
> - **Short Test Sequences:** `FD002` (`min = 21` cycles) and `FD004` (`min = 19` cycles) contain test engines shorter than `WINDOW_SIZE = 30`, handled cleanly via causal left-edge repeat padding so every test engine receives a valid prediction without dropping units.
> - **Piecewise-Linear RUL Clipping:** Training targets are clipped at `RUL_CLIP = 125` cycles (standard C-MAPSS practice since healthy engines operate flatly before fault onset); test metrics are reported on **both** clipped-at-125 RUL (for direct literature comparison) and unclipped raw `RUL_FD00X.txt`.

---

## Phase 1 EDA: Constant Sensors & Why `FD002`/`FD004` Mask Them

Across all four subsets, **7 sensors (`1, 5, 6, 10, 16, 18, 19`)** carry zero or near-zero degradation signal (`within_regime_std < 0.02`) and are dropped, leaving **14 informative sensors** (`2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21`).

On `FD002` and `FD004`, naive global variance analysis **falsely flags these constant sensors as informative** because changing flight conditions (altitude, Mach number, throttle resolver angle) shift baseline sensor levels dramatically across the 6 operating regimes — masking zero within-regime degradation signal:

| Dropped Sensor | FD002 Global Std | FD002 Within-Regime Std | FD004 Global Std | FD004 Within-Regime Std | Finding |
|---|---|---|---|---|---|
| `sensor_1` | 26.3895 | **0.0000** | 26.4366 | **0.0000** | Regime-stepped constant |
| `sensor_5` | 3.6138 | **0.0000** | 3.6228 | **0.0000** | Regime-stepped constant |
| `sensor_6` | 5.4318 | **0.0036** | 5.4440 | **0.0148** | Regime-stepped near-constant |
| `sensor_10` | 0.1275 | **0.0015** | 0.1277 | **0.0047** | Regime-stepped near-constant |
| `sensor_16` | 0.0047 | **0.0007** | 0.0047 | **0.0006** | Discrete / near-constant |
| `sensor_18` | 145.3266 | **0.0000** | 145.4713 | **0.0000** | Regime-stepped constant |
| `sensor_19` | 5.3640 | **0.0000** | 5.3694 | **0.0000** | Regime-stepped constant |

| Sensor Variance (Global vs Within-Regime) | FD002 & FD004 6-Regime Operating Clusters |
|---|---|
| ![Sensor Variance Comparison](reports/figures/eda_sensor_variance.png) | ![Operating Regime Clusters](reports/figures/eda_regimes_FD002_FD004.png) |

| Sample Degradation Traces (`FD001`) | Sample Degradation Traces (`FD004`) |
|---|---|
| ![FD001 Sensor Degradation Traces](reports/figures/eda_sensors_FD001.png) | ![FD004 Sensor Degradation Traces](reports/figures/eda_sensors_FD004.png) |

---

## Engineering Ablations (`Phases 2–4`)

All preprocessing (`RegimeNormalizer`, `FaultModeClusterer`), engine-level train/val splits (`80/20`, zero `unit_id` overlap), causal rolling features (`windows = 5, 10, 20`, `diff(1)`), and conformal residual calibration are fit **strictly on training engines**.

### 1. Operating Regime Normalization (`FD002` & `FD004`)

Clustering `op_setting_1..3` with `KMeans(k=6)` and normalizing each sensor **within its operating regime** isolates true degradation from flight-envelope jumps (evaluated on official test sets, clipped@125):

| Dataset | Model | Global Z-Score RMSE (Before) | Within-Regime KMeans(k=6) RMSE (After) | RMSE Change | Global NASA Score (Before) | Within-Regime NASA Score (After) | NASA Score Change |
|---|---|---|---|---|---|---|---|
| **FD002** | XGBoost | 17.08 | **15.10** | **-11.6%** | 2,198.5 | **1,182.1** | **-46.2%** |
| **FD002** | Deep1DCNN | 18.27 | **17.22** | **-5.8%** | 1,575.9 | **1,267.8** | **-19.5%** |
| **FD004** | XGBoost | 20.07 | **17.44** | **-13.1%** | 3,256.9 | **2,496.8** | **-23.3%** |
| **FD004** | Deep1DCNN | 22.91 | **18.95** | **-17.3%** | 4,244.3 | **3,017.2** | **-28.9%** |

![Regime Normalization Impact](reports/figures/ablation_regime_norm.png)

### 2. Latent Fault-Mode Clustering vs Implicit Learning (`FD003` & `FD004`)

`FaultModeClusterer(k=2)` fits KMeans on training end-of-life sensor signatures (last 25 cycles) and computes causal trailing soft fault-mode probabilities (`fault_mode_prob_0..1`) without target leakage:

| Dataset | Model | Implicit RMSE | Explicit FaultCluster(k=2) RMSE | RMSE Δ | Implicit NASA Score | Explicit FaultCluster(k=2) NASA Score | NASA Score Δ |
|---|---|---|---|---|---|---|---|
| **FD003** | XGBoost | **17.56** | 17.96 | +2.3% | **993.6** | 1,123.8 | +13.1% |
| **FD003** | Deep1DCNN | **14.13** | 15.90 | +12.5% | **383.4** | 644.4 | +68.1% |
| **FD004** | XGBoost | **17.44** | 17.63 | +1.1% | **2,496.8** | 2,621.2 | +5.0% |
| **FD004** | Deep1DCNN | 18.95 | **18.25** | **-3.7%** | 3,017.2 | **2,167.0** | **-28.2%** |

> **Engineering Takeaway:** On single-regime `FD003` and tree models (which already partition multi-window rolling features), appending soft `k=2` cluster probabilities adds slight early-cycle variance. Conversely, on the hardest 6-regime, 2-fault `FD004` sequence setting (`Deep1DCNN`), explicit fault-mode probabilities cut RMSE by **3.7%** and NASA Score by **28.2%**. Model selection chooses the validation-best configuration per dataset transparently.

### 3. Per-Dataset vs Unified Multi-Dataset Models

Pooling all **709 training engines** across `FD001`–`FD004` with `dataset_FD001..FD004` and 6-regime indicators lets a single unified model share degradation physics across fleets:

| Dataset | Architecture | Per-Dataset RMSE (Clipped) | Unified RMSE (Clipped) | RMSE Δ | Per-Dataset NASA Score | Unified NASA Score | Score Δ |
|---|---|---|---|---|---|---|---|
| **FD001** | XGBoost | 18.07 | **15.24** | -2.83 | 1,067.7 | **569.9** | -46.6% |
| **FD001** | Deep1DCNN | 14.67 | **14.11** | -0.56 | **276.3** | 413.7 | +49.7% |
| **FD001** | BiLSTM | 14.58 | **13.90** | -0.68 | **405.1** | 415.5 | +2.6% |
| **FD002** | XGBoost | **15.10** | 15.26 | +0.16 | **1,182.1** | 1,257.0 | +6.3% |
| **FD002** | Deep1DCNN | 17.22 | **15.32** | -1.90 | 1,267.8 | **1,043.9** | -17.7% |
| **FD002** | BiLSTM | **14.06** | 14.37 | +0.32 | 1,041.5 | **772.5** | -25.8% |
| **FD003** | XGBoost | 17.56 | **16.82** | -0.74 | **993.6** | 1,084.4 | +9.1% |
| **FD003** | Deep1DCNN | **14.13** | 14.53 | +0.40 | **383.4** | 501.0 | +30.7% |
| **FD003** | BiLSTM | 16.01 | **13.44** | -2.57 | 1,047.2 | **953.1** | -9.0% |
| **FD004** | XGBoost | 17.44 | **17.19** | -0.25 | 2,496.8 | **2,039.5** | -18.3% |
| **FD004** | Deep1DCNN | 18.95 | **16.42** | -2.53 | 3,017.2 | **1,251.0** | -58.5% |
| **FD004** | BiLSTM | 16.02 | **14.69** | -1.33 | 1,474.5 | **1,007.8** | -31.7% |

---

## Literature Benchmark Comparison (`Phase 5`)

We evaluate with both **RMSE** and the **NASA Asymmetric Scoring Function** (Saxena et al., 2008 Eq. 11), where $d_i = \hat{y}_i - y_i$:

$$s_i = \begin{cases} e^{-d_i / 13} - 1, & d_i < 0 \text{ (early prediction)} \\ e^{d_i / 10} - 1, & d_i \ge 0 \text{ (late overestimation)} \end{cases}$$

Using denominator `10` for late predictions ($d_i \ge 0$) vs `13` for early predictions penalizes dangerous late failure overestimation exponentially harder. Negative `% diff` below denotes lower error than the published benchmark.

| Dataset | Model / Reference | Scope | RMSE (Clipped@125) | NASA Score (Clipped@125) | Unclipped RMSE / NASA | 90% CI Coverage | vs Zheng (2017) RMSE / Score | vs Li (2018) RMSE / Score |
|---|---|---|---|---|---|---|---|---|
| **FD001** | Babu et al. (2016) CNN | Literature | 18.45 | — | — | — | — | — |
| **FD001** | Zheng et al. (2017) Deep LSTM | Literature | 16.14 | 338.0 | — | — | — | — |
| **FD001** | Li et al. (2018) DCNN | Literature | 12.61 | 273.7 | — | — | — | — |
| **FD001** | RandomForest | Per-Dataset | 18.31 | 1,184.3 | 19.31 / 1,220.9 | 90.0% | +13.4% / +250.4% | +45.2% / +332.7% |
| **FD001** | XGBoost | Per-Dataset | 18.07 | 1,067.7 | 19.06 / 1,102.5 | 90.0% | +12.0% / +215.9% | +43.3% / +290.1% |
| **FD001** | Deep1DCNN | Per-Dataset | 14.67 | **276.3** | 16.28 / 331.8 | 93.0% | -9.1% / **-18.3%** | +16.3% / **+0.9%** |
| **FD001** | BiLSTM | Per-Dataset | 14.58 | 405.1 | 15.73 / 437.1 | 89.0% | -9.7% / +19.8% | +15.6% / +48.0% |
| **FD001** | **Unified BiLSTM** | Unified | **13.90** | 415.5 | 15.19 / 455.5 | 88.0% | **-13.9%** / +22.9% | +10.2% / +51.8% |
| **FD002** | Babu et al. (2016) CNN | Literature | 30.29 | — | — | — | — | — |
| **FD002** | Zheng et al. (2017) Deep LSTM | Literature | 24.49 | 4,450.0 | — | — | — | — |
| **FD002** | Li et al. (2018) DCNN | Literature | 22.36 | 10,412.0 | — | — | — | — |
| **FD002** | RandomForest | Per-Dataset | 15.90 | 1,599.0 | 28.26 / 11,653.1 | 93.1% | -35.1% / -64.1% | -28.9% / -84.6% |
| **FD002** | XGBoost | Per-Dataset | 15.10 | 1,182.1 | 27.62 / 10,088.3 | 93.1% | -38.3% / -73.4% | -32.5% / -88.6% |
| **FD002** | Deep1DCNN | Per-Dataset | 17.22 | 1,267.8 | 29.99 / 16,361.4 | 88.4% | -29.7% / -71.5% | -23.0% / -87.8% |
| **FD002** | **BiLSTM** | Per-Dataset | **14.06** | 1,041.5 | 26.24 / 7,759.4 | 91.9% | **-42.6%** / -76.6% | **-37.1%** / -90.0% |
| **FD002** | **Unified BiLSTM** | Unified | 14.37 | **772.5** | 27.39 / 10,705.6 | 87.3% | -41.3% / **-82.6%** | -35.7% / **-92.6%** |
| **FD003** | Babu et al. (2016) CNN | Literature | 19.82 | — | — | — | — | — |
| **FD003** | Zheng et al. (2017) Deep LSTM | Literature | 16.18 | 852.0 | — | — | — | — |
| **FD003** | Li et al. (2018) DCNN | Literature | 12.64 | 284.1 | — | — | — | — |
| **FD003** | RandomForest | Per-Dataset | 17.63 | 1,109.9 | 18.69 / 1,143.8 | 81.0% | +9.0% / +30.3% | +39.5% / +290.7% |
| **FD003** | XGBoost | Per-Dataset | 17.56 | 993.6 | 18.49 / 1,025.6 | 83.0% | +8.5% / +16.6% | +38.9% / +249.8% |
| **FD003** | BiLSTM | Per-Dataset | 16.01 | 1,047.2 | 17.29 / 1,089.2 | 79.0% | -1.0% / +22.9% | +26.7% / +268.6% |
| **FD003** | Deep1DCNN | Per-Dataset | 14.13 | **383.4** | 15.76 / 442.8 | 86.0% | -12.7% / **-55.0%** | +11.8% / +34.9% |
| **FD003** | **Unified BiLSTM** | Unified | **13.44** | 953.1 | 14.88 / 991.5 | 94.0% | **-16.9%** / +11.9% | +6.3% / +235.5% |
| **FD004** | Babu et al. (2016) CNN | Literature | 29.16 | — | — | — | — | — |
| **FD004** | Zheng et al. (2017) Deep LSTM | Literature | 28.17 | 5,550.0 | — | — | — | — |
| **FD004** | Li et al. (2018) DCNN | Literature | 23.31 | 12,466.0 | — | — | — | — |
| **FD004** | RandomForest | Per-Dataset | 18.06 | 2,895.9 | 29.03 / 6,496.1 | 86.3% | -35.9% / -47.8% | -22.5% / -76.8% |
| **FD004** | XGBoost | Per-Dataset | 17.44 | 2,496.8 | 28.38 / 5,780.2 | 88.7% | -38.1% / -55.0% | -25.2% / -80.0% |
| **FD004** | Deep1DCNN | Per-Dataset | 18.95 | 3,017.2 | 28.91 / 7,347.9 | 88.3% | -32.7% / -45.6% | -18.7% / -75.8% |
| **FD004** | BiLSTM | Per-Dataset | 16.02 | 1,474.5 | 27.61 / 5,501.2 | 86.7% | -43.1% / -73.4% | -31.3% / -88.2% |
| **FD004** | **Unified BiLSTM** | Unified | **14.69** | **1,007.8** | 27.17 / 5,571.7 | 87.9% | **-47.9% / -81.8%** | **-37.0% / -91.9%** |

| Benchmark Comparison Across FD001–FD004 | Test RUL Predictions & 90% Confidence Bands |
|---|---|
| ![Benchmark Comparison](reports/figures/model_benchmark_comparison.png) | ![RUL Predictions Across FD001-FD004](reports/figures/model_rul_predictions_FD001_FD004.png) |

---

## Fleet Maintenance Policy Simulation & Dollar Impact (`Phase 6`)

Using real cycle-by-cycle validation trajectories from the per-dataset BiLSTM models across **142 held-out engines** (`FD001=20`, `FD002=52`, `FD003=20`, `FD004=50`), we compare a predictive pull threshold (`pred_rul <= threshold`) against fixed-interval overhauls scaled to a **300-engine fleet** (800 cycles/engine/year):

### Per-Dataset & Combined Fleet Breakdown (`threshold = 20` cycles)

| Fleet Scope | Val Engines | Fixed-150 Failures / Early | Predictive-20 Failures / Early | Fixed-150 Annual Cost | Best-Fixed Annual Cost (Interval) | Predictive-20 Annual Cost | Savings vs Fixed-150 | Savings vs Best-Fixed |
|---|---|---|---|---|---|---|---|---|
| **FD001** | 20 | 1 / 12 | **0 / 0** | $247.18M/yr | $228.41M/yr (120 cyc) | **$149.53M/yr** | **$97.64M/yr (39.5%)** | **$78.88M/yr (34.5%)** |
| **FD002** | 52 | 3 / 34 | **0 / 3** | $258.19M/yr | $206.13M/yr (140 cyc) | **$141.84M/yr** | **$116.35M/yr (45.1%)** | **$64.28M/yr (31.2%)** |
| **FD003** | 20 | 0 / 17 | **0 / 0** | $202.70M/yr | $202.70M/yr (150 cyc) | **$123.98M/yr** | **$78.73M/yr (38.8%)** | **$78.73M/yr (38.8%)** |
| **FD004** | 50 | 2 / 40 | **0 / 3** | $245.23M/yr | $219.90M/yr (140 cyc) | **$123.16M/yr** | **$122.07M/yr (49.8%)** | **$96.74M/yr (44.0%)** |
| **Combined Fleet** | **142** | **6 / 103** | **0 / 6** | **$243.60M/yr** | **$219.14M/yr (140 cyc)** | **$132.78M/yr** | **$110.82M/yr (45.5%)** | **$86.36M/yr (39.4%)** |

### Predictive Threshold Sensitivity Sweep (`Combined 142-Engine Validation Fleet`)

| Pull Threshold (cycles) | Unplanned Failures | Failures Avoided vs Fixed-150 | Early Pulls | Avg Wasted Cycles | Cost / Engine Life | Annual 300-Fleet Cost | Annual Savings vs Fixed-150 | Savings (%) |
|---|---|---|---|---|---|---|---|---|
| 10 | 4 | 2 | 5 | 2.75 | $149,501.41 | $160.77M | $82.83M | 34.0% |
| 15 | 0 | 6 | 7 | 2.96 | $123,557.75 | $132.87M | $110.73M | 45.5% |
| **20 (Optimal)** | **0** | **6** | **6** | **2.89** | **$123,473.24** | **$132.78M** | **$110.82M** | **45.5%** |
| 25 | 0 | 6 | 7 | 2.95 | $123,540.85 | $132.85M | $110.75M | 45.5% |
| 30 | 0 | 6 | 12 | 3.46 | $124,157.75 | $133.51M | $110.09M | 45.2% |
| 35 | 0 | 6 | 11 | 3.37 | $124,039.44 | $133.39M | $110.21M | 45.2% |

![Fleet Maintenance Cost Comparison](reports/figures/maintenance_cost_comparison.png)

---

## Interactive Streamlit Dashboard (`app/streamlit_app.py`)

Run the 4-tab diagnostic and fleet simulator app locally:

- **Tab 1 — Live Engine Diagnostics & RUL Inference**
  - *Mode A (Test Engine Replay)*: Scrub any real `FD001`–`FD004` test engine cycle-by-cycle using per-dataset or unified trained checkpoints (`BiLSTM`, `Deep1DCNN`, `XGBoost`, `RandomForest`) to view live RUL + 90% conformal band, operating regime (`KMeans k=6`), latent fault-mode probabilities (`FD003`/`FD004`), and Maintenance Status Badge (`HEALTHY > 40`, `MONITOR 20–40`, `CRITICAL <= 20`).
  - *Mode B (Interactive Sensor & Operating Sliders)*: Adjust normalized Z-scores or raw physical sensor values and operating settings to observe real-time RUL, uncertainty interval, and top contributing sensors.
- **Tab 2 — Dataset & EDA Insights**: Difficulty progression table, constant vs informative sensors comparison, and EDA visualizations.
- **Tab 3 — Engineering Ablations & Literature Benchmarks**: Regime normalization, fault-mode clustering, unified vs per-dataset tables, and full benchmark comparison (Clipped@125 & Unclipped).
- **Tab 4 — 300-Engine Fleet Maintenance Simulator**: Interactive financial sliders (unscheduled removal, AOG/day, scheduled overhaul, threshold, fixed interval) computing live unplanned failures avoided, early pulls, and annual `$M` savings.

---

## Project Structure

```text
├── app/
│   └── streamlit_app.py              # 4-tab interactive diagnostics & fleet cost simulator
├── data/raw/                         # NASA C-MAPSS FD001-FD004 text files
├── models/
│   ├── FD001..FD004/                 # Saved normalizers, fault clusterers, RF/XGB/CNN/BiLSTM checkpoints
│   └── unified/                      # Multi-dataset normalizer, Unified XGBoost/1D-CNN/BiLSTM
├── reports/
│   ├── figures/                      # EDA, ablation, benchmark, and maintenance cost plots
│   └── results/                      # Verified JSON/CSV metrics & validation trajectories
├── scripts/
│   ├── download_data.py              # Raw C-MAPSS downloader
│   ├── run_eda.py                    # Phase 1 EDA & variance summary generator
│   ├── run_experiments.py            # Phases 2-5 ablations, model training & test evaluation
│   └── run_maintenance_sim.py        # Phase 6 fleet cost simulation & plot generator
├── src/
│   ├── config.py                     # Paths, constants, canonical drop/informative sensors
│   ├── data/                         # Loader, engine-level split, RegimeNormalizer (KMeans k=6)
│   ├── features/                     # Causal rolling stats, FaultModeClusterer, sequence windows
│   ├── models/                       # Tabular (RF/XGB + conformal) & PyTorch (1D-CNN/BiLSTM + MC Dropout)
│   ├── evaluation/                   # RMSE, MAE, NASA asymmetric score, benchmarks, maintenance sim
│   └── utils/                        # Plotting utilities
├── tests/                            # Pytest suite (data, causality, metrics, models, Streamlit smoke)
├── pyproject.toml
└── requirements.txt
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Windows) Avoid slow temp Matplotlib cache creation
set MPLCONFIGDIR=.cache/matplotlib   # PowerShell: $env:MPLCONFIGDIR=".cache/matplotlib"

# 3. Run unit & integration tests
pytest -v

# 4. Reproduce EDA, model training/ablations, and maintenance simulation
python scripts/run_eda.py
python scripts/run_experiments.py
python scripts/run_maintenance_sim.py

# 5. Launch the interactive Streamlit app
streamlit run app/streamlit_app.py
```
