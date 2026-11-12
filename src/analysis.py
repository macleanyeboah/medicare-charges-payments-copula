"""Main analysis: dependence between hospital list prices (charges) and what Medicare pays."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyvinecopulib as pv
from scipy import stats

import copula_tools as ct
import prepare

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
N_BOOT = 300


def tau(a, b) -> float:
    return float(stats.kendalltau(a, b)[0])


def rho(a, b) -> float:
    return float(stats.spearmanr(a, b)[0])


def dependence_summary(raw: pd.DataFrame, h: pd.DataFrame) -> pd.DataFrame:
    """How the charge-payment association shrinks as confounders are removed."""
    keep = ~raw["state"].isin(prepare.EXCLUDED_STATES)
    r = raw[keep]
    lc, lp = np.log(r["avg_charges"]), np.log(r["avg_total_payments"])
    rows = [
        ("1. Raw rows: log charges vs log payments", len(r), tau(lc, lp), rho(lc, lp)),
        ("2. Rows after removing DRG means", len(r), tau(r["r_charges"], r["r_payments"]),
         rho(r["r_charges"], r["r_payments"])),
        ("3. Hospital indices (DRG-adjusted)", len(h), tau(h["charge_index"], h["payment_index"]),
         rho(h["charge_index"], h["payment_index"])),
        ("4. Hospital indices, within state", len(h),
         tau(h["charge_index_within_state"], h["payment_index_within_state"]),
         rho(h["charge_index_within_state"], h["payment_index_within_state"])),
        ("Sensitivity: Medicare-only payments (level 3)", len(h),
         tau(h["charge_index"], h["medicare_index"]), rho(h["charge_index"], h["medicare_index"])),
    ]
    return pd.DataFrame(rows, columns=["level", "n", "kendall_tau", "spearman_rho"])


def ci(series: pd.Series) -> list:
    return [float(series.quantile(0.025)), float(series.quantile(0.975))]


def joint_extremes(u: np.ndarray, fitted: dict, names: list, qs=(0.90, 0.95)) -> pd.DataFrame:
    rows = []
    for q in qs:
        row = {"threshold": f"top {int(round((1 - q) * 100))}% in both",
               "empirical": float(np.mean((u[:, 0] > q) & (u[:, 1] > q))),
               "independence": (1 - q) ** 2}
        for nm in names:
            row[nm] = ct.joint_upper(fitted[nm], q)
        rows.append(row)
    return pd.DataFrame(rows)


def maryland_check(raw: pd.DataFrame) -> pd.DataFrame:
    ratio = raw["avg_charges"] / raw["avg_total_payments"]
    s = ratio.groupby(raw["state"]).median().rename("median_charge_to_payment").reset_index()
    return s.sort_values("median_charge_to_payment").reset_index(drop=True)


def vine_with_volume(h: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    x = np.column_stack([h["charge_index"], h["payment_index"], np.log(h["discharges"])])
    u = ct.pseudo_obs(x)
    # Same candidate families as the bivariate analysis, so the two parts are comparable
    families = list(dict.fromkeys(f for f, _ in ct.CANDIDATES))
    controls = pv.FitControlsVinecop(family_set=families)
    vc = pv.Vinecop.from_data(u, controls=controls)
    names = ["charges", "payments", "volume"]
    rows = []
    for t, tree in enumerate(vc.pair_copulas):
        for e, pc in enumerate(tree):
            rows.append({"tree": t + 1, "edge": e + 1, "family": ct.label(pc.family, pc.rotation),
                         "kendall_tau": pc.tau})
    # Plain-language edge labels from the structure matrix
    m = np.asarray(vc.matrix)
    d = 3
    for row in rows:
        t, e = row["tree"] - 1, row["edge"] - 1
        a = names[m[d - 1 - e, e] - 1]
        b = names[m[t, e] - 1]
        cond = [names[m[k, e] - 1] for k in range(t)]
        row["pair"] = f"{a}, {b}" + (f" | {', '.join(cond)}" if cond else "")
    return pd.DataFrame(rows)[["tree", "pair", "family", "kendall_tau"]], str(vc)


def main() -> dict:
    TABLES.mkdir(parents=True, exist_ok=True)
    raw = prepare.drg_residuals(prepare.load_raw())
    h = prepare.main()
    h_main = h[~h["state"].isin(prepare.EXCLUDED_STATES)].reset_index(drop=True)

    dep = dependence_summary(raw, h_main)
    dep.to_csv(TABLES / "01_dependence_by_adjustment.csv", index=False)

    # National hospital-level copula
    x = h_main[["charge_index", "payment_index"]].to_numpy()
    u = ct.pseudo_obs(x)
    fit_tab, fitted = ct.fit_candidates(u)
    fit_tab.to_csv(TABLES / "02_copula_selection_national.csv", index=False)
    best = fit_tab.loc[0, "copula"]

    # Best-AIC family that allows tail dependence: the comparator to the AIC winner
    tail_ok = fit_tab[(fit_tab["lambda_lower"] + fit_tab["lambda_upper"]) > 0.01]
    alt = tail_ok.iloc[0]["copula"]

    # Bootstrap (resampling hospitals) for both models
    boot = {}
    for nm in dict.fromkeys([best, alt]):
        cop = fitted[nm]
        b = ct.bootstrap(x, cop.family, cop.rotation, n_boot=N_BOOT)
        est = {"kendall_tau": cop.tau,
               "lambda_lower": ct.tail_dependence(cop)[0],
               "lambda_upper": ct.tail_dependence(cop)[1]}
        boot[nm] = {k: {"estimate": float(est[k]), "ci95": ci(b[k])} for k in b.columns}
        b.to_csv(TABLES / f"03_bootstrap_{nm.replace(' ', '_').replace('(', '').replace(')', '')}.csv",
                 index=False)

    # Is the upper tail really stronger than the lower tail? Model-free bootstrap check.
    asym = ct.tail_asymmetry_bootstrap(x, qs=(0.90, 0.95), n_boot=1000)
    asym.to_csv(TABLES / "03c_tail_asymmetry.csv", index=False)

    # Within-state copula (removes state price levels)
    xs = h_main[["charge_index_within_state", "payment_index_within_state"]].to_numpy()
    us = ct.pseudo_obs(xs)
    fit_ws, fitted_ws = ct.fit_candidates(us)
    fit_ws.to_csv(TABLES / "04_copula_selection_within_state.csv", index=False)

    # Joint extremes: how often is a hospital top-decile on both?
    jx = joint_extremes(u, fitted, list(dict.fromkeys([best, alt, "student"])))
    jx.to_csv(TABLES / "05_joint_extremes.csv", index=False)

    # Tail-dependence curves for the figure
    qs = np.round(np.arange(0.80, 0.991, 0.01), 2)
    curves = pd.DataFrame({"q": qs})
    curves["empirical_upper"] = [ct.chi_empirical(u, q, "upper") for q in qs]
    curves["empirical_lower"] = [ct.chi_empirical(u, q, "lower") for q in qs]
    for nm in dict.fromkeys([best, alt]):
        curves[f"{nm}_upper"] = [ct.chi_upper_model(fitted[nm], q) for q in qs]
        curves[f"{nm}_lower"] = [ct.chi_lower_model(fitted[nm], q) for q in qs]
    curves.to_csv(TABLES / "06_tail_curves.csv", index=False)

    md = maryland_check(raw)
    md.to_csv(TABLES / "07_charge_to_payment_by_state.csv", index=False)

    vine_tab, vine_txt = vine_with_volume(h_main)
    vine_tab.to_csv(TABLES / "08_vine_with_volume.csv", index=False)

    summary = {
        "n_raw_rows": int(len(raw)),
        "n_drgs": int(raw["drg"].nunique()),
        "n_hospitals_all": int(raw["provider_id"].nunique()),
        "n_hospitals_analysed": int(len(h_main)),
        "dependence": dep.to_dict(orient="records"),
        "best_copula_national": best,
        "best_copula_within_state": fit_ws.loc[0, "copula"],
        "tail_dependent_alternative": alt,
        "tail_asymmetry": asym.to_dict(orient="records"),
        "top5_national": fit_tab.head(5)[["copula", "AIC", "delta_AIC", "kendall_tau",
                                          "lambda_lower", "lambda_upper"]].to_dict(orient="records"),
        "top5_within_state": fit_ws.head(5)[["copula", "AIC", "delta_AIC", "kendall_tau",
                                             "lambda_lower", "lambda_upper"]].to_dict(orient="records"),
        "bootstrap": boot,
        "joint_extremes": jx.to_dict(orient="records"),
        "maryland_median_ratio": float(md.loc[md.state == "MD", "median_charge_to_payment"].iloc[0]),
        "national_median_ratio_excl_md": float(
            (raw.loc[raw.state != "MD", "avg_charges"] / raw.loc[raw.state != "MD", "avg_total_payments"]).median()),
        "vine": vine_tab.to_dict(orient="records"),
    }
    with open(ROOT / "results" / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))
    return summary


if __name__ == "__main__":
    main()
