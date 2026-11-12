"""Figures for the README. Run after analysis.py."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import copula_tools as ct
import prepare

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures"

BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "legend.frameon": False,
})


def fig_raw_vs_adjusted(raw: pd.DataFrame, h: pd.DataFrame, summary_tau: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    r = raw[~raw["state"].isin(prepare.EXCLUDED_STATES)]
    ax = axes[0]
    hb = ax.hexbin(np.log10(r["avg_total_payments"]), np.log10(r["avg_charges"]), gridsize=60,
                   bins="log", cmap="Blues", mincnt=1, linewidths=0)
    ax.set_xlabel("log10 average total payment ($)")
    ax.set_ylabel("log10 average covered charge ($)")
    t1 = summary_tau.loc[0, "kendall_tau"]
    ax.set_title(f"Raw hospital x DRG rows (tau = {t1:.2f})", loc="left")
    fig.colorbar(hb, ax=ax, label="rows (log scale)")

    ax = axes[1]
    u = ct.pseudo_obs(h[["payment_index", "charge_index"]].to_numpy())
    ax.scatter(u[:, 0], u[:, 1], s=8, color=BLUE, alpha=0.35, linewidths=0)
    ax.set_xlabel("payment index (rank scale)")
    ax.set_ylabel("charge index (rank scale)")
    t3 = summary_tau.loc[2, "kendall_tau"]
    ax.set_title(f"Hospitals after removing DRG mix (tau = {t3:.2f})", loc="left")
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    fig.text(0.01, 0.01, "Source: CMS IPPS Provider Summary FY2011. Maryland excluded.",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGS / "fig1_raw_vs_adjusted.png", dpi=160)
    plt.close(fig)


def fig_tail_curves(curves: pd.DataFrame, best: str, alt: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, tail, title in [(axes[0], "upper", "Upper tail: both high"),
                            (axes[1], "lower", "Lower tail: both low")]:
        ax.plot(curves["q"], curves[f"empirical_{tail}"], "o", ms=5, color=INK, label="Empirical")
        ax.plot(curves["q"], curves[f"{best}_{tail}"], lw=2, color=BLUE, label=f"{best.capitalize()} (best AIC)")
        ax.plot(curves["q"], curves[f"{alt}_{tail}"], lw=2, color=ORANGE, label=alt.upper().replace("(SURVIVAL)", "(survival)"))
        ax.set_title(title, loc="left")
        ax.set_xlabel("threshold q")
    axes[0].set_ylabel("chi(q): P(other also beyond q | one beyond q)")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_tail_dependence_curves.png", dpi=160)
    plt.close(fig)


def fig_state_ratio(md: pd.DataFrame) -> None:
    md = md.sort_values("median_charge_to_payment")
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [ORANGE if s == "MD" else BLUE for s in md["state"]]
    ax.bar(md["state"], md["median_charge_to_payment"], color=colors, width=0.75)
    ax.axhline(1, color=INK2, lw=1, ls="--")
    ax.set_ylabel("median charges / payments")
    rest = md[md["state"] != "MD"]["median_charge_to_payment"]
    ax.set_title(f"Hospitals bill {rest.min():.1f}x to {rest.max():.1f}x what they are paid, "
                 f"except in rate-regulated Maryland ({md.loc[md.state == 'MD', 'median_charge_to_payment'].iloc[0]:.2f}x)",
                 loc="left")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_charge_to_payment_by_state.png", dpi=160)
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    import json
    s = json.load(open(ROOT / "results" / "summary.json"))
    raw = prepare.drg_residuals(prepare.load_raw())
    h = pd.read_csv(prepare.PROCESSED)
    h = h[~h["state"].isin(prepare.EXCLUDED_STATES)]
    fig_raw_vs_adjusted(raw, h, pd.read_csv(TABLES / "01_dependence_by_adjustment.csv"))
    fig_tail_curves(pd.read_csv(TABLES / "06_tail_curves.csv"), s["best_copula_national"],
                    s["tail_dependent_alternative"])
    fig_state_ratio(pd.read_csv(TABLES / "07_charge_to_payment_by_state.csv"))
    print("figures written to", FIGS.relative_to(ROOT))


if __name__ == "__main__":
    main()
