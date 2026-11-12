"""Build DRG-adjusted hospital charge and payment indices from the CMS IPPS FY2011 file.

Raw rows are hospital x DRG averages. Most of the raw correlation between charges and
payments comes from case mix (expensive DRGs have high charges AND high payments), so we
remove the DRG effect first and work with one index per hospital.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "ipps_fy2011.csv.gz"
PROCESSED = ROOT / "data" / "processed" / "hospital_indices.csv"

# Maryland sets hospital rates for all payers under a federal waiver, so its "charges"
# are regulated prices, not chargemaster list prices. It is analysed separately.
EXCLUDED_STATES = ["MD"]
MIN_DRGS = 10  # hospitals with fewer DRG rows give noisy indices

RENAME = {
    "DRG Definition": "drg",
    "Provider Id": "provider_id",
    "Provider Name": "provider_name",
    "Provider State": "state",
    "Hospital Referral Region Description": "hrr",
    "Total Discharges": "discharges",
    "Average Covered Charges": "avg_charges",
    "Average Total Payments": "avg_total_payments",
    "Average Medicare Payments": "avg_medicare_payments",
}


def load_raw(path: Path = RAW) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=RENAME)[list(RENAME.values())]
    return df


def drg_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """Log amount minus the DRG mean of log amount (computed on the analysis states)."""
    out = df.copy()
    for col, name in [("avg_charges", "r_charges"),
                      ("avg_total_payments", "r_payments"),
                      ("avg_medicare_payments", "r_medicare")]:
        logx = np.log(out[col])
        drg_mean = logx[~out["state"].isin(EXCLUDED_STATES)].groupby(out["drg"]).mean()
        out[name] = logx - out["drg"].map(drg_mean)
    return out


def hospital_indices(df: pd.DataFrame, min_drgs: int = MIN_DRGS) -> pd.DataFrame:
    """Discharge-weighted mean DRG residual per hospital.

    charge_index = 0.30 means the hospital's list prices run about exp(0.30) - 1 = 35%
    above the national average for the same DRGs.
    """
    w = df["discharges"]
    g = df.assign(wc=df["r_charges"] * w, wp=df["r_payments"] * w,
                  wm=df["r_medicare"] * w).groupby("provider_id")
    h = pd.DataFrame({
        "provider_name": g["provider_name"].first(),
        "state": g["state"].first(),
        "n_drgs": g.size(),
        "discharges": g["discharges"].sum(),
        "charge_index": g["wc"].sum() / g["discharges"].sum(),
        "payment_index": g["wp"].sum() / g["discharges"].sum(),
        "medicare_index": g["wm"].sum() / g["discharges"].sum(),
    }).reset_index()
    h = h[h["n_drgs"] >= min_drgs].copy()
    # State-centred versions: what is left after removing state-level price levels
    for col in ["charge_index", "payment_index"]:
        h[col + "_within_state"] = h[col] - h.groupby("state")[col].transform("mean")
    return h


def main() -> pd.DataFrame:
    df = drg_residuals(load_raw())
    h = hospital_indices(df)
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    h.to_csv(PROCESSED, index=False)
    print(f"raw rows: {len(df):,} | hospitals kept: {len(h):,} "
          f"(min {MIN_DRGS} DRGs) | written to {PROCESSED.relative_to(ROOT)}")
    return h


if __name__ == "__main__":
    main()
