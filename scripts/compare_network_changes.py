# scripts/compare_network_changes.py
"""
Compare infrastructure between the base elec_s_37 network and a solved network.
Reports expansions for:
- AC lines (s_nom / s_nom_opt)
- DC links (p_nom / p_nom_opt)
- Generators (p_nom / p_nom_opt)

Outputs:
  results/tables/expanded_lines.csv
  results/tables/expanded_links.csv
  results/tables/expanded_generators.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pypsa

# --- paths (adjust if needed) ---
BASE_NC = Path("data/raw/elec_s_37.nc")
SOLVED_NC = Path("results/networks/solved_baseline_costed_expansion.nc")
OUT_DIR = Path("results/tables")
OUT_LINES = OUT_DIR / "expanded_lines.csv"
OUT_LINKS = OUT_DIR / "expanded_links.csv"
OUT_GENS  = OUT_DIR / "expanded_generators.csv"

TOL = 1e-6  # numerical tolerance


def _opt_or_current(df, opt_col, cur_col):
    """Return a Series of optimal capacity if available, else current."""
    if opt_col in df:
        return df[opt_col].astype(float)
    return df[cur_col].astype(float)


def compare_lines(n0: pypsa.Network, n1: pypsa.Network) -> pd.DataFrame:
    if len(n0.lines) == 0 or len(n1.lines) == 0:
        return pd.DataFrame(columns=["bus0", "bus1", "added_MVA", "new_MVA", "base_MVA"])

    base = n0.lines[["bus0", "bus1", "s_nom"]].copy()
    base.rename(columns={"s_nom": "base_MVA"}, inplace=True)

    new_mva = _opt_or_current(n1.lines, "s_nom_opt", "s_nom")
    df = n1.lines[["bus0", "bus1"]].copy()
    df["new_MVA"] = new_mva

    # align on common index
    common = base.index.intersection(df.index)
    merged = base.loc[common].join(df.loc[common][["new_MVA"]])

    merged["added_MVA"] = np.clip(merged["new_MVA"] - merged["base_MVA"], 0, None)
    expanded = merged[merged["added_MVA"] > TOL].sort_values("added_MVA", ascending=False)
    return expanded[["bus0", "bus1", "added_MVA", "new_MVA", "base_MVA"]]


def compare_links(n0: pypsa.Network, n1: pypsa.Network) -> pd.DataFrame:
    if len(n0.links) == 0 or len(n1.links) == 0:
        return pd.DataFrame(columns=["bus0", "bus1", "carrier", "added_MW", "new_MW", "base_MW"])

    base = n0.links[["bus0", "bus1", "carrier", "p_nom"]].copy()
    base.rename(columns={"p_nom": "base_MW"}, inplace=True)

    new_mw = _opt_or_current(n1.links, "p_nom_opt", "p_nom")
    df = n1.links[["bus0", "bus1", "carrier"]].copy()
    df["new_MW"] = new_mw

    common = base.index.intersection(df.index)
    merged = base.loc[common].join(df.loc[common][["new_MW"]])

    merged["added_MW"] = np.clip(merged["new_MW"] - merged["base_MW"], 0, None)
    expanded = merged[merged["added_MW"] > TOL].sort_values("added_MW", ascending=False)
    return expanded[["bus0", "bus1", "carrier", "added_MW", "new_MW", "base_MW"]]


def compare_generators(n0: pypsa.Network, n1: pypsa.Network) -> pd.DataFrame:
    if len(n0.generators) == 0 or len(n1.generators) == 0:
        return pd.DataFrame(columns=["bus", "carrier", "added_MW", "new_MW", "base_MW"])

    base = n0.generators[["bus", "carrier", "p_nom"]].copy()
    base.rename(columns={"p_nom": "base_MW"}, inplace=True)

    new_mw = _opt_or_current(n1.generators, "p_nom_opt", "p_nom")
    df = n1.generators[["bus", "carrier"]].copy()
    df["new_MW"] = new_mw

    common = base.index.intersection(df.index)
    merged = base.loc[common].join(df.loc[common][["new_MW"]])

    merged["added_MW"] = np.clip(merged["new_MW"] - merged["base_MW"], 0, None)
    expanded = merged[merged["added_MW"] > TOL].sort_values("added_MW", ascending=False)
    return expanded[["bus", "carrier", "added_MW", "new_MW", "base_MW"]]


def main():
    print("Loading networks…")
    n0 = pypsa.Network(str(BASE_NC))
    n1 = pypsa.Network(str(SOLVED_NC))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lines_exp = compare_lines(n0, n1)
    links_exp = compare_links(n0, n1)
    gens_exp  = compare_generators(n0, n1)

    lines_exp.to_csv(OUT_LINES)
    links_exp.to_csv(OUT_LINKS)
    gens_exp.to_csv(OUT_GENS)

    # Summaries
    total_lines = float(lines_exp["added_MVA"].sum()) if not lines_exp.empty else 0.0
    total_links = float(links_exp["added_MW"].sum()) if not links_exp.empty else 0.0
    total_gens  = float(gens_exp["added_MW"].sum()) if not gens_exp.empty else 0.0

    print("\n=== Infrastructure Expansion Summary ===")
    print(f"AC lines expanded: {len(lines_exp):>4} (total added {total_lines:,.0f} MVA)")
    print(f"DC links expanded: {len(links_exp):>4} (total added {total_links:,.0f} MW)")
    print(f"Generators expanded: {len(gens_exp):>4} (total added {total_gens:,.0f} MW)")

    def head_fmt(df, cols, title, unit):
        if df.empty:
            print(f"\nTop {title}: none")
            return
        print(f"\nTop 10 {title}:")
        print(df[cols].head(10).to_string(index=True))

    head_fmt(lines_exp, ["bus0", "bus1", "added_MVA"], "AC line expansions", "MVA")
    head_fmt(links_exp, ["bus0", "bus1", "carrier", "added_MW"], "DC link expansions", "MW")
    head_fmt(gens_exp,  ["bus", "carrier", "added_MW"], "generator expansions", "MW")

    print("\nWrote:")
    print(f"  {OUT_LINES}")
    print(f"  {OUT_LINKS}")
    print(f"  {OUT_GENS}")
    print("Done.")


if __name__ == "__main__":
    main()
