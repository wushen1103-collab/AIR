#!/usr/bin/env python3
"""Format the fully audited E91/E96 long tables; no new calculations or plots."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
LIT_ORDER = ("Random", "RF/ECFP", "TcsAL-MLP-BALD-50e10",
             "TcsAL-MLP-Exploitation-50e10", "GLARE-50e10",
             "ACActive-50e10", "ChemScreener-50e10", "Positive-Tanimoto",
             "Positive-Tversky-T1", "AIR-Legacy-2e")
LIT_ASSAYS = ("ADRB2", "ALDH1", "FEN1", "GBA", "IDH1", "KAT2A", "MAPK1",
              "MTORC1", "OPRK1", "PKM2", "VDR")
TDC_ORDER = ("RF", "AC-Active", "GLARE", "Neighbor", "Tversky-T1", "AIR-E", "AIR-Legacy")
TDC_AIDS = ("1798", "463087", "485290")


def cell(mean, sd):
    return f"{float(mean):.2f} ± {float(sd):.2f}"


def main():
    lit = pd.read_csv(TABLES / "e96_matched_lit_final_status.csv")
    tdc = pd.read_csv(TABLES / "e91_tdc_authorstart_aligned_status.csv")
    if (len(lit) != 10 * 11 or (lit.status != "complete").any()
            or (lit.n_seeds != 5).any() or set(lit.method) != set(LIT_ORDER)
            or set(lit.assay) != set(LIT_ASSAYS)):
        raise RuntimeError("LIT 11-assay × 10-method × 5-seed audit incomplete")
    if (len(tdc) != 7 * 3 or (tdc.status != "complete").any()
            or (tdc.n_seeds != 5).any() or set(tdc.method) != set(TDC_ORDER)
            or set(tdc.aid.astype(str)) != set(TDC_AIDS)):
        raise RuntimeError("TDC 3-assay × 7-method × 5-seed audit incomplete")
    lit_rows = []
    for method in LIT_ORDER:
        d = lit[lit.method == method].set_index("assay")
        row = {"route": d.route.iloc[0], "method": method,
               "provenance": d.provenance.iloc[0], "configuration": d.model_config.iloc[0]}
        for assay in LIT_ASSAYS:
            row[assay] = cell(d.loc[assay, "mean_hits"], d.loc[assay, "sd_hits"])
        row["macro_mean_hits"] = round(float(d.mean_hits.mean()), 3)
        row["macro_mean_nef"] = round(float(d.mean_nef.mean()), 3)
        lit_rows.append(row)
    lit_out = TABLES / "e92_fcs_lit_q192_paper_table.csv"
    pd.DataFrame(lit_rows).to_csv(lit_out, index=False)

    tdc_rows = []
    for method in TDC_ORDER:
        d = tdc[tdc.method == method].copy()
        d.aid = d.aid.astype(str)
        d = d.set_index("aid")
        row = {"route": d.route.iloc[0], "method": method,
               "provenance": d.provenance.iloc[0], "configuration": d.model_config.iloc[0]}
        for aid in TDC_AIDS:
            row[f"AID{aid}"] = cell(d.loc[aid, "mean_hits"], d.loc[aid, "sd_hits"])
        row["macro_mean_hits"] = round(float(d.mean_hits.mean()), 3)
        tdc_rows.append(row)
    tdc_out = TABLES / "e92_fcs_tdc_q384_paper_table.csv"
    pd.DataFrame(tdc_rows).to_csv(tdc_out, index=False)
    print("WROTE", lit_out, tdc_out)
    print("LIT_MACRO", {r["method"]: r["macro_mean_hits"] for r in lit_rows})
    print("TDC_MACRO", {r["method"]: r["macro_mean_hits"] for r in tdc_rows})


if __name__ == "__main__":
    main()
