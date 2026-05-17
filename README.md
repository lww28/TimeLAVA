# TimeLAVA

An PyTorch implementation of **TimeLAVA** from

> W.Liu, W.Quan, A.Zuo, E.Gao, V.Nguyen, D.Sejdinovic, H.Bondell, M.Gong.
> *"TimeLAVA: Learning-Agnostic Valuation for Time Series Data."*
> **ICML 2026.**

TimeLAVA assigns a value to each temporal segment of a time series by its
marginal contribution to minimising the **Selective Wavelet-based Wasserstein
(WSW)** discrepancy between the evaluated series and a reference series — with
**no model training**.

```
timelava-pkg/
├── README.md
├── LICENSE
├── src/
│   └── timelava/
│       ├── __init__.py            # public API
│       ├── config.py              # TimeLAVAConfig (paper defaults)
│       ├── segmentation.py        # sliding window (§3.2)
│       ├── wavelet.py             # DWT features + ground metric (Def. 4.1/4.2)
│       ├── label_consistency.py   # conditional Wasserstein term (Eq. 6)
│       ├── uot.py                 # log-domain unbalanced Sinkhorn (Alg. 1 L13)
│       ├── core.py                # TimeLAVA estimator (Alg. 1 + Alg. 2)
│       └── datasets.py            # synthetic generators + metrics
├── examples/
│   ├── demo.ipynb                 # executed notebook with visualisations
│   └── reproduce_paper.py         # CLI reproduction of the 3 checks
└── tests/
    └── test_timelava.py           # pytest suite (7 tests)
```

## Install

```bash
cd timelava-pkg
pip install -e .                 # core
pip install -e ".[notebook]"     # + matplotlib / jupyter for demo.ipynb
pip install -e ".[dev]"          # + pytest / scipy for the test suite
```

## Quick start

```python
import numpy as np
from timelava import TimeLAVA, TimeLAVAConfig

X_eval = ...   # (T_eval, d) series to value
X_ref  = ...   # (T_ref,  d) clean reference series

# Anomaly detection / forecasting pruning (unsupervised, c = 0)
cfg = TimeLAVAConfig(L=64, S=8, kappa=2.0, reg=0.01, c=0.0)
res = TimeLAVA(cfg).fit(X_eval, X_ref)

res.segment_values      # v_eps(x_i)  — one value per segment
res.point_values        # v(t)        — one value per time step
res.anomaly_scores()    # -v(t)       — higher = more anomalous
res.corruption_scores() # -v(x_i)     — higher = more likely corrupted
res.rank_segments()     # segment indices, best first

# Label-noise detection (supervised, c = 1)
cfg = TimeLAVAConfig(L=64, S=64, kappa=2.0, reg=0.01, c=1.0)
res = TimeLAVA(cfg).fit(X_eval, X_ref, y_eval=ye, y_ref=yr)
```

## Reproduce the paper's synthetic validations

```bash
python examples/reproduce_paper.py     # prints the 3 checks
jupyter notebook examples/demo.ipynb   # same, with plots
pytest                                 # the same claims as assertions
```

`examples/demo.ipynb` is already executed, so the figures are visible on
GitHub without running anything.

## Exact mapping to the paper

| Paper | Module / symbol |
|---|---|
| Sliding window, length `L`, stride `S` (§3.2) | `segmentation.sliding_window_segments` |
| DWT features, `db4`, level 2 (Def. 4.1, App. C.1.3) | `wavelet.wavelet_features` (Mallat `wavedec`) |
| Wavelet distance `‖Ψ(xᵢ)−Ψ(x′ⱼ)‖₁` (Def. 4.2, Eq. 4) | `wavelet.wavelet_cost_matrix` |
| Label-consistency `c·W_{d_wav}(μ(·\|yᵢ),μ(·\|y′ⱼ))` (Eq. 6) | `label_consistency.conditional_wasserstein_matrix` |
| WSW = entropy-reg. UOT, KL marginals (Def. 4.4, Eq. 5; Alg. 1 L13) | `uot.unbalanced_sinkhorn_dual` |
| `ψ_κ(u)=κ(1−e^{−u/κ})` (Thm. 4.5) | `uot.psi_kappa` |
| `v(xᵢ)=−(φᵢ−(S−φᵢ)/(n−1))` (Thm. 4.5, Eq. 7; Alg. 1 L14–20) | `core.TimeLAVA.fit` |
| Point-wise `v(t)=mean over covering segments` (Eq. 8; Alg. 2) | `core.TimeLAVA._pointwise` |
| Defaults κ=2.0, ε=0.01, db4, level 2 (App. C.1.3) | `config.TimeLAVAConfig` |

## Note on the UOT solver

Algorithm 1 (line 13) solves the entropy-regularised unbalanced OT problem and
reads off the **optimal dual potentials** (Appendix B.4). Off-the-shelf
stabilised Sinkhorn becomes numerically unstable at the very small `ε` used in
the paper's convergence study (Theorem B.2, down to `1e-4`). This
implementation uses a **log-domain unbalanced Sinkhorn** (the Séjourné et al.,
2019 formulation the paper cites), which is stable across the entire `ε` range
and returns `(f*, g*)` directly in the cost's own units — exactly as defined by
the KKT system in Appendix B.4. Contraction factor `ρ = κ/(κ+ε)`.

The entropic scaling iteration contracts more slowly as `ε → 0`, so the solver
checks the change of *both* potentials each sweep against an absolute tolerance
and lets the iteration ceiling scale with `1/ε` (Sinkhorn's known `O(1/ε)`
iteration complexity). This is required for the values to converge
**monotonically** with the empirical rate `~O(ε^0.9)` the paper reports in
Figure 8; a fixed small iteration cap silently returns unconverged potentials
at tiny `ε`. The regression is locked down by
`test_theorem_b_2_monotone_convergence`.
