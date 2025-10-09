# scripts/compare_baseline_vs_10.py
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa


def get_weights(n: pypsa.Network) -> pd.Series:
    """
    Return snapshot weights as a Series aligned with n.snapshots.
    Tries common PyPSA columns; falls back to ones.
    """
    w = getattr(n, "snapshot_weightings", None)
    if w is None:
        return pd.Series(1.0, index=n.snapshots)
    if isinstance(w, pd.DataFrame):
        for col in ("generators", "objective", "stores"):
            if col in w.columns:
                return w[col]
        # single-column fallback
        if w.shape[1] == 1:
            return w.iloc[:, 0]
        return pd.Series(1.0, index=n.snapshots)
    # already a Series
    return w.reindex(n.snapshots).fillna(1.0)


def summarize_prices(n: pypsa.Network) -> dict:
    w = get_weights(n)
    lmp = n.buses_t.marginal_price.copy()
    # Make sure it's numeric
    lmp = lmp.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # system mean = snapshot-weighted mean across buses
    sys_mean = (lmp.mul(w, axis=0)).sum().sum() / (w.sum() * lmp.shape[1])

    frac_hours_any_lt1 = float((lmp.lt(1.0).any(axis=1)).mean())
    return {
        "system_weighted_mean_eur_per_MWh": float(sys_mean),
        "min_LMP": float(lmp.min().min()),
        "max_LMP": float(lmp.max().max()),
        "pct_hours_any_bus_LMP_lt_1": 100.0 * frac_hours_any_lt1,
    }


def summarize_congestion(n: pypsa.Network) -> dict:
    if n.lines.empty or not hasattr(n.lines_t, "p0"):
        return {"fraction_binding_line_hours": 0.0}

    p = n.lines_t.p0.abs()
    smax_pu = n.lines.get("s_max_pu", pd.Series(1.0, index=n.lines.index)).fillna(1.0)
    s_nom = n.lines.get("s_nom", pd.Series(0.0, index=n.lines.index)).fillna(0.0)
    limit = (s_nom * smax_pu).replace(0, np.nan)  # avoid divide-by-zero artifacts

    # Fraction of (time,line) where near the thermal limit
    binding = p.ge(limit * 0.999, axis=1)
    frac = float(binding.mean().mean())
    return {"fraction_binding_line_hours": frac}


def summarize_curtailment(n: pypsa.Network) -> dict:
    if n.generators.empty or not hasattr(n.generators_t, "p"):
        return {"curtail_share": 0.0}

    w = get_weights(n)
    # Availability if p_max_pu exists; otherwise zero (we only want VRE curtailment)
    if hasattr(n.generators_t, "p_max_pu"):
        avail = n.generators_t.p_max_pu.mul(n.generators.p_nom.fillna(0.0), axis=1)
    else:
        # No availability series → assume no curtailment accounted
        avail = n.generators_t.p * 0.0

    gen = n.generators_t.p.clip(lower=0.0)
    curtail = (avail - gen).clip(lower=0.0)

    gen_MWh = gen.mul(w, axis=0).to_numpy().sum()
    cur_MWh = curtail.mul(w, axis=0).to_numpy().sum()
    denom = gen_MWh + cur_MWh
    share = float(cur_MWh / denom) if denom > 0 else 0.0
    return {"curtail_share": share}


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:,.2f}%"


def main():
    ap = argparse.ArgumentParser(description="Compare baseline vs 10% CO2 reduction scenario (prices, congestion, curtailment).")
    ap.add_argument("--baseline", default="results/networks/solved_baseline_costed_expansion.nc")
    ap.add_argument("--reduction", default="results/networks/solved_reduction_10.nc")
    ap.add_argument("--out", default=None, help="Optional CSV path for a compact summary table.")
    args = ap.parse_args()

    base_path = Path(args.baseline)
    red_path = Path(args.reduction)

    if not base_path.exists():
        sys.exit(f"Baseline network not found: {base_path}")
    if not red_path.exists():
        sys.exit(f"Reduction network not found: {red_path}")

    print("Loading networks…")
    nb = pypsa.Network(str(base_path))
    nr = pypsa.Network(str(red_path))

    print("\n=== Price Summary ===")
    pb = summarize_prices(nb)
    pr = summarize_prices(nr)
    for k in pb.keys():
        vb, vr = pb[k], pr[k]
        if "pct_" in k or "fraction" in k:
            print(f"{k:32s}  baseline={vb:,.2f}   red10={vr:,.2f}   Δ={vr - vb:,.2f}")
        else:
            print(f"{k:32s}  baseline={vb:,.4f}  red10={vr:,.4f}  Δ={vr - vb:,.4f}")

    print("\n=== Congestion Summary ===")
    cb = summarize_congestion(nb)
    cr = summarize_congestion(nr)
    fb, fr = cb["fraction_binding_line_hours"], cr["fraction_binding_line_hours"]
    print(f"fraction_binding_line_hours       baseline={fmt_pct(fb)}  red10={fmt_pct(fr)}  Δ={fmt_pct(fr - fb)}")

    print("\n=== Curtailment Summary ===")
    ub = summarize_curtailment(nb)
    ur = summarize_curtailment(nr)
    print(f"curtail_share                     baseline={fmt_pct(ub['curtail_share'])}  red10={fmt_pct(ur['curtail_share'])}  Δ={fmt_pct(ur['curtail_share'] - ub['curtail_share'])}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        def row_from(tag, d):
            return {"metric": tag, **d}

        rows.append(row_from("prices_baseline", pb))
        rows.append(row_from("prices_red10", pr))
        rows.append(row_from("congestion_baseline", cb))
        rows.append(row_from("congestion_red10", cr))
        rows.append(row_from("curtailment_baseline", ub))
        rows.append(row_from("curtailment_red10", ur))

        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\nWrote summary CSV: {out}")


if __name__ == "__main__":
    main()
