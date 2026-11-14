# AeroFed: Physics-Informed Federated Conformal Prognostics
### Research memo — joint proposal

**Authors:**
Dr. Elena Vasquez — Associate Professor, Aerospace Propulsion & Learning Systems Lab (fictional, for project framing)
Marcus Okafor, M.Eng — Propulsion Systems Engineer, 9 years turbine health monitoring (fictional, for project framing)

**Status:** Draft proposal, open for critique.
**Scope:** One semester, single contributor, NASA C-MAPSS FD001–FD004.

---

## 1. The problem we're actually solving

Remaining Useful Life (RUL) estimation for turbofan engines has two separate
gaps that are usually addressed in isolation:

1. **Statistical gap** — operators (airlines, MRO shops) want a RUL number
   they can act on, with a defensible uncertainty bound, not a point
   estimate they have to trust blindly.
2. **Data-sharing gap** — different operators see different degradation
   regimes (climate, route profile, duty cycle) but cannot pool raw
   telemetry across companies for competitive and contractual reasons.

Most published work addresses one or the other. We haven't found anything
in the accessible literature that combines (a) distribution-free uncertainty
quantification with (b) federated training *and* (c) a feature set that is
explicitly tied to the physical degradation model the data was generated
from, rather than raw/engineered sensor features alone. That combination,
done honestly and at small scale, is what we're proposing.

This is not claimed as a publishable research contribution. It is claimed
as a technically serious, fully implementable combination of three
established methods, applied to a dataset whose generative physics
(Saxena et al., 2008) is fully documented, which lets us ground one piece
of the feature engineering in that physics rather than inventing it.

## 2. The physics we're actually using (not inventing)

The original C-MAPSS paper defines a generalized health index:

```
h(t) = 1 - exp{ -a · t^b }        (Eq. 4/5, Saxena et al. 2008)
```

where `a` and `b` are degradation-rate parameters drawn per unit from a
constrained random distribution, and `t` is operating cycle. This is not
a black-box feature — it is the literal equation NASA used to generate
the ground-truth degradation trajectories.

**Our one genuine technical contribution**: fit `a` and `b` per engine by
nonlinear least squares against a smoothed, normalized composite sensor
signal (the sensors most correlated with HPC/EGT margins, per Phase 1 EDA),
then use the fitted `(a, b)` pair as two additional engineered features
alongside rolling statistics.

This is modest, defensible, and traceable line-for-line to the source
paper — which is exactly the property the earlier "HoloTwin" concept
lacked. We are not claiming our model discovers physics. We are claiming
we extracted two interpretable parameters *from* the known physics and
let the model use them.

**Honest limitation**: curve-fitting `a, b` per engine requires enough
cycles of data to converge, so this feature is unreliable very early in
an engine's life — we will report where it helps and where it doesn't,
not just where it helps.

## 3. The federated layer

Split FD002/FD004 (the multi-regime datasets) into three simulated
operator clients by dominant operating-condition cluster:

- **Client A — high-altitude/high-Mach duty cycle** (proxy: harsher thermal cycling)
- **Client B — low-altitude/low-throttle duty cycle** (proxy: slower, steadier wear)
- **Client C — mixed regime** (the general case)

This is a simulated split of a single public dataset, not real
multi-airline data — the memo should say this plainly every time it's
mentioned, including in any presentation. It is a credible stand-in for
the real-world federated scenario (airlines cannot share telemetry) but
it is not real cross-organization data, and we will not describe it as
such.

Implementation: `FedAvg` (McMahan et al., 2017) using PyTorch, 3 clients,
local epochs tunable, compare:
- Global federated model
- Each client's locally-trained-only model
- A fully centralized model trained on pooled data (upper bound, not
  achievable in a real multi-operator setting — included only as a
  reference ceiling)

**What we report**: RMSE/NASA-score per client under each regime, plus
convergence curves over federated rounds. If federated underperforms
centralized by a wide margin, we report that honestly as the expected
cost of data privacy, not hide it.

## 4. The uncertainty layer

Point predictions without error bars are not actionable for a maintenance
decision. Use split conformal prediction (Vovk et al.) on top of the
trained regressor to produce a distribution-free prediction interval
with a stated coverage guarantee (e.g., 90%).

This is a well-established, lightweight method — not a research risk —
and it is what turns "predicted RUL: 14 cycles" into something a
maintenance planner could actually use: "predicted RUL: 14 cycles,
90% confidence the true RUL is between 9 and 19."

## 5. Architecture (text form)

```
Raw C-MAPSS sensors (FD001-FD004)
        │
        ├── Phase 1: EDA, drop near-constant sensors
        ├── Phase 2: per-regime normalization (KMeans, k=6)
        ├── Phase 3: rolling-window features + fitted (a, b) physics features
        │
        ├── Centralized baseline model (RF / LSTM)
        │
        └── Federated split (3 simulated clients)
                 │
                 ├── Local training per client
                 ├── FedAvg aggregation → global model
                 │
                 └── Conformal calibration layer
                          │
                          └── RUL + prediction interval per engine
```

## 6. What we would NOT be able to defend if pushed

Written for internal use, not for the presentation:

- The 3-client split is a simulated proxy, not evidence that this works
  on real heterogeneous airline data at scale.
- `(a, b)` curve fitting assumes the single-exponential form from the
  2008 paper is still the right functional form for degradation — real
  engines may not follow it cleanly, and we have not validated this
  assumption independently.
- Federated learning with only 3 clients and a few hundred total engines
  is a toy-scale demonstration of the *mechanism*, not evidence of
  benefit at production fleet scale.
- Conformal intervals are only as good as the calibration set — with a
  small held-out calibration split, coverage guarantees will be loose.

## 7. Why this version is worth presenting

Every component here is either (a) a standard, citable method used
correctly, or (b) a modest, explicitly-scoped extension derived directly
from the dataset's own source paper. Nothing here requires a HOD or
recruiter to take an unverifiable claim on faith — every number on the
final slide traces back to a method that can be explained in two
sentences and defended under a follow-up question.
