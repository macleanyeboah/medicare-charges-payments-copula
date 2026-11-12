"""Small helpers around pyvinecopulib for bivariate copula selection and tail measures."""
import numpy as np
import pandas as pd
import pyvinecopulib as pv
from scipy import stats

F = pv.BicopFamily
# Candidate families with the rotations that allow positive dependence.
CANDIDATES = [
    (F.indep, 0), (F.gaussian, 0), (F.student, 0), (F.frank, 0),
    (F.clayton, 0), (F.clayton, 180),
    (F.gumbel, 0), (F.gumbel, 180),
    (F.joe, 0), (F.joe, 180),
    (F.bb1, 0), (F.bb1, 180),
    (F.bb7, 0), (F.bb7, 180),
]


def pseudo_obs(x: np.ndarray) -> np.ndarray:
    """Column-wise ranks scaled to (0, 1): the standard rank transform for copula data."""
    x = np.asarray(x, dtype=float)
    return np.column_stack([stats.rankdata(c) / (len(c) + 1) for c in x.T])


def label(family, rotation) -> str:
    name = str(family).split(".")[-1]
    return name if rotation == 0 else f"{name} (survival)"


def fit_one(u: np.ndarray, family, rotation) -> pv.Bicop:
    cop = pv.Bicop(family=family, rotation=rotation)
    cop.fit(u)
    return cop


def fit_candidates(u: np.ndarray) -> tuple[pd.DataFrame, dict]:
    rows, fitted = [], {}
    for fam, rot in CANDIDATES:
        cop = fit_one(u, fam, rot)
        lam_l, lam_u = tail_dependence(cop)
        lab = label(fam, rot)
        fitted[lab] = cop
        rows.append({
            "copula": lab,
            "n_par": cop.npars,
            "loglik": cop.loglik(u),
            "AIC": cop.aic(u),
            "BIC": cop.bic(u),
            "kendall_tau": cop.tau,
            "lambda_lower": lam_l,
            "lambda_upper": lam_u,
            "parameters": np.round(np.asarray(cop.parameters).ravel(), 4).tolist(),
        })
    tab = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    tab["delta_AIC"] = tab["AIC"] - tab["AIC"].min()
    return tab, fitted


def tail_dependence(cop: pv.Bicop) -> tuple[float, float]:
    """(lower, upper) tail dependence coefficients of a fitted copula."""
    # pyvinecopulib returns a 2x2 matrix: [0, 0] = lower-lower, [1, 1] = upper-upper.
    td = np.asarray(cop.taildep)
    return float(td[0, 0]), float(td[1, 1])


def joint_upper(cop: pv.Bicop, q: float) -> float:
    """P(U > q, V > q) under the copula."""
    c = float(cop.cdf(np.array([[q, q]]))[0])
    return 1 - 2 * q + c


def chi_upper_model(cop: pv.Bicop, q: float) -> float:
    return joint_upper(cop, q) / (1 - q)


def chi_lower_model(cop: pv.Bicop, q: float) -> float:
    """P(V < 1-q | U < 1-q)."""
    p = 1 - q
    return float(cop.cdf(np.array([[p, p]]))[0]) / p


def chi_empirical(u: np.ndarray, q: float, tail: str = "upper") -> float:
    if tail == "upper":
        return float(np.mean((u[:, 0] > q) & (u[:, 1] > q)) / (1 - q))
    p = 1 - q
    return float(np.mean((u[:, 0] < p) & (u[:, 1] < p)) / p)


def tail_asymmetry_bootstrap(x: np.ndarray, qs=(0.90, 0.95), n_boot: int = 1000,
                             seed: int = 7) -> pd.DataFrame:
    """Empirical chi_upper(q) - chi_lower(q) with a percentile bootstrap CI over hospitals."""
    rng = np.random.default_rng(seed)
    u = pseudo_obs(x)
    n = len(x)
    rows = []
    for q in qs:
        point = chi_empirical(u, q, "upper") - chi_empirical(u, q, "lower")
        diffs = []
        for _ in range(n_boot):
            ub = pseudo_obs(x[rng.integers(0, n, n)])
            diffs.append(chi_empirical(ub, q, "upper") - chi_empirical(ub, q, "lower"))
        diffs = np.array(diffs)
        rows.append({"q": q,
                     "chi_upper": chi_empirical(u, q, "upper"),
                     "chi_lower": chi_empirical(u, q, "lower"),
                     "difference": point,
                     "ci95_low": float(np.quantile(diffs, 0.025)),
                     "ci95_high": float(np.quantile(diffs, 0.975)),
                     "share_boot_diff_le_0": float(np.mean(diffs <= 0))})
    return pd.DataFrame(rows)


def bootstrap(x: np.ndarray, family, rotation, n_boot: int = 300, seed: int = 2026) -> pd.DataFrame:
    """Resample hospitals, re-rank, refit the given family. Returns tau and tail coefficients."""
    rng = np.random.default_rng(seed)
    n = len(x)
    out = []
    for _ in range(n_boot):
        xb = x[rng.integers(0, n, n)]
        cop = fit_one(pseudo_obs(xb), family, rotation)
        lam_l, lam_u = tail_dependence(cop)
        out.append({"kendall_tau": cop.tau, "lambda_lower": lam_l, "lambda_upper": lam_u})
    return pd.DataFrame(out)
