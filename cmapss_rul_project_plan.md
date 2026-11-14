# Turbofan Engine RUL Prediction — Project Plan
### NASA C-MAPSS Dataset (FD001–FD004) | Pure Python

## Business framing
Predictive maintenance for jet engines. Unscheduled engine removal costs
$500K–$1M+ per incident; AOG (aircraft-on-ground) costs can hit $150K/day.
Accurate RUL (Remaining Useful Life) prediction shifts maintenance from
fixed-interval to predictive — industry estimates put savings at 10–40% of
fleet maintenance cost. For a 300-engine fleet, that's tens of millions
annually. This is the pitch to open your README/portfolio writeup with.

## Dataset structure
Each `train_FD00X.txt` / `test_FD00X.txt`: 26 whitespace-separated columns,
no header.

```
unit_id, cycle, op_setting_1, op_setting_2, op_setting_3, sensor_1 ... sensor_21
```

| Dataset | Engines (train) | Operating conditions | Fault modes |
|---|---|---|---|
| FD001 | 100 | 1 | 1 (HPC degradation) |
| FD002 | 260 | 6 | 1 |
| FD003 | 100 | 1 | 2 (HPC + Fan) |
| FD004 | 248 | 6 | 2 |

Difficulty order: FD001 < FD003 < FD002 < FD004.
Build and validate the pipeline on FD001 first; FD002/FD004 need an extra
normalization step (see Phase 2) before anything works well.

Matching `RUL_FD00X.txt` files give ground-truth RUL for the *last* cycle
of each engine in the test set — use these to score your test predictions.

---

## Phase 1 — Setup & EDA
- [ ] Load all four train/test sets with the column names above
- [ ] Load matching `RUL_FD00X.txt` as test-set ground truth
- [ ] Compute training RUL labels: `RUL = max_cycle(unit) - cycle`
- [ ] Clip RUL at 125 (standard practice — engines run healthy/flat for a
      long time before degrading; capping avoids penalizing the model for
      not predicting decline too early)
- [ ] Plot sensor readings vs. cycle for 3–4 engines per dataset
- [ ] Identify and drop near-constant sensors (expect sensors 1, 5, 6, 10,
      16, 18, 19 to be flat/uninformative in C-MAPSS) — state this finding
      explicitly in your writeup, it signals real EDA

## Phase 2 — Handle operating conditions (FD002/FD004)
This is the hard part and what makes FD002/FD004 worth including.
- [ ] Cluster the 3 operational settings into regimes with KMeans (k=6)
- [ ] Z-score normalize each sensor **within its regime**, not globally
- [ ] Document before/after RMSE — without this step FD002/FD004 models
      perform dramatically worse; the improvement is a result in itself

## Phase 3 — Feature engineering
- [ ] Rolling mean/std per sensor over windows of 5, 10, 20 cycles
- [ ] Rate-of-change (first difference) per sensor
- [ ] For FD003/FD004 (2 fault modes): try modeling fault mode as a latent
      cluster vs. letting the model learn it implicitly — compare both

## Phase 4 — Modeling (two tracks per dataset)
- [ ] **Tabular baseline**: Random Forest / XGBoost on engineered features
      at each cycle snapshot
- [ ] **Sequence model**: LSTM or 1D-CNN on sliding windows (e.g. 30 cycles
      × 14 sensors) — this is what gets near published benchmark RMSEs
      (~12–20 cycles on FD001, worse on FD002/FD004)
- [ ] Try one model per dataset AND one unified model with dataset/regime
      as a feature — compare the two approaches

## Phase 5 — Evaluation (don't skip this)
- [ ] Report RMSE
- [ ] Implement the NASA asymmetric scoring function (Eq. 11 from the
      Saxena et al. "Damage Propagation Modeling" paper):
      penalizes late predictions exponentially harder than early ones
      (a1=10 for early, a2=13 for late)
- [ ] Compare your scores against published C-MAPSS benchmarks (search
      "C-MAPSS RMSE benchmark FD001-FD004", e.g. Zheng et al. 2017 LSTM
      paper) — be able to say "within X% of published SOTA"

## Phase 6 — Business framing + deployment
- [ ] Simulate a maintenance policy: flag engine for maintenance when
      predicted RUL < threshold (e.g. 20 cycles)
- [ ] Compare simulated cost vs. fixed-interval maintenance: count (a)
      unplanned failures avoided, (b) engines pulled unnecessarily early
- [ ] Wrap the trained model in a Streamlit app: input sensor readings,
      get RUL + confidence band
- [ ] Package as a GitHub repo with a README stating: the business
      problem, the four-dataset difficulty progression, and your
      benchmark comparison table

---

## Stack
pandas, numpy, scikit-learn, PyTorch (or Keras), matplotlib/seaborn,
Streamlit. All pure Python — no TypeScript/JS needed anywhere in this
project.

## What makes this stand out vs. generic Kaggle notebooks
1. Using the paper's own asymmetric scoring metric (most public notebooks
   only report RMSE)
2. Testing on all four sub-datasets, not just FD001
3. Explicitly solving the multi-operating-condition normalization problem
4. Quantifying dollar impact via the maintenance policy simulation
