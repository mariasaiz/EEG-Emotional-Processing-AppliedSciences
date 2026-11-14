"""RQ3: hemispheric EEG asymmetry from final channel-level band power (N=35).

Input: OUTPUT_FINAL_UNIFIED/BandPower_channel_level.csv
       Power entries are V² provided the original Bitbrain µV -> V conversion
       was correctly applied in the upstream preprocessing pipeline.
Output: OUTPUT_RQ3_ASYMMETRY/*.csv and RQ3_DASM_RASM_FINAL_N35.xlsx.

DASM = P_R - P_L, reported in µV² (multiply V² by 1e12).
RASM = P_R/P_L; ln(RASM) and RASM are dimensionless.
All reported absolute means are computed by taking absolute values BEFORE averaging.

Privacy: never publish the generated participant-level outputs to GitHub.
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd


def project_root() -> Path:
    env_root = os.environ.get("EEG_PROJECT_DIR")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parent


ROOT = project_root()
INPUT_FILE = ROOT / "OUTPUT_FINAL_UNIFIED" / "BandPower_channel_level.csv"
OUTPUT_DIR = ROOT / "OUTPUT_RQ3_ASYMMETRY"
POWER_TO_UV2 = 1e12

PAIRS = {
    "AF7-AF8": ("AF7", "AF8"),
    "Fp1-Fp2": ("Fp1", "Fp2"),
    "F3-F4": ("F3", "F4"),
    "P3-P4": ("P3", "P4"),
    "P7-P8": ("P7", "P8"),  # Verified against Bitbrain info.json EEG signal_loc.
    "O1-O2": ("O1", "O2"),
}
BANDS = ["Delta", "Theta", "Alpha", "Beta", "LowGamma"]
STIMULI = ["Surprise", "Anger", "Happiness", "Sadness", "Fear"]
EXPECTED_N = 35


def main() -> None:
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(f"Missing final band-power input: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    required = {"Code", "Stimulus", "Event_code", "Channel", "Band", "Power", "Power_unit"}
    if missing := required.difference(df.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if len(df) != EXPECTED_N * len(STIMULI) * len(BANDS) * 12:
        raise ValueError(f"Expected 10,500 input rows, found {len(df)}")
    if df["Code"].nunique() != EXPECTED_N or 32 in set(df["Code"]):
        raise ValueError("Expected N=35 and participant 32 excluded")
    if set(df["Stimulus"]) != set(STIMULI) or set(df["Band"]) != set(BANDS):
        raise ValueError("Unexpected stimulus or frequency-band labels")
    channels = {ch for pair in PAIRS.values() for ch in pair}
    if set(df["Channel"]) != channels:
        raise ValueError(f"Channel mismatch: expected {sorted(channels)}, found {sorted(set(df['Channel']))}")
    if df.duplicated(["Code", "Stimulus", "Event_code", "Band", "Channel"]).any():
        raise ValueError("Duplicate participant/stimulus/event/band/channel entries")
    powers = pd.to_numeric(df["Power"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(powers).all() or (powers <= 0).any():
        raise ValueError("RASM requires strictly positive, finite band power")
    # 'working_units2' is the preprocessing script's label. Physical units
    # are inferred from its original amplitude conversion µV * 1e-6 -> V.
    if set(df["Power_unit"].astype(str)) != {"working_units2"}:
        raise ValueError("Unexpected Power_unit; verify input scaling before proceeding")

    wide = df.pivot(index=["Code", "Stimulus", "Event_code", "Band"], columns="Channel", values="Power")
    if len(wide) != EXPECTED_N * len(STIMULI) * len(BANDS) or wide.isna().any().any():
        raise ValueError("Expected 875 complete participant-stimulus-band combinations")

    long_parts = []
    for pair_name, (left, right) in PAIRS.items():
        # Convert both powers to µV² before subtraction. Ratios are invariant.
        p_l = wide[left].to_numpy(dtype=float) * POWER_TO_UV2
        p_r = wide[right].to_numpy(dtype=float) * POWER_TO_UV2
        dasm = p_r - p_l
        rasm = p_r / p_l
        log_rasm = np.log(rasm)
        index_frame = wide.index.to_frame(index=False)
        index_frame["Pair"] = pair_name
        index_frame["Left_channel"] = left
        index_frame["Right_channel"] = right
        index_frame["P_L"] = p_l
        index_frame["P_R"] = p_r
        index_frame["DASM"] = dasm
        index_frame["Abs_DASM"] = np.abs(dasm)
        index_frame["RASM"] = rasm
        index_frame["Log_RASM"] = log_rasm
        index_frame["Abs_Log_RASM"] = np.abs(log_rasm)
        long_parts.append(index_frame)

    asymmetry_df = pd.concat(long_parts, ignore_index=True)
    if len(asymmetry_df) != 5250:
        raise ValueError(f"Expected 5250 asymmetry observations; got {len(asymmetry_df)}")
    if not np.isfinite(asymmetry_df[["DASM", "RASM", "Log_RASM"]].to_numpy()).all():
        raise ValueError("Invalid asymmetry indices")

    summary = (asymmetry_df.groupby(["Code", "Band"], as_index=False)
               .agg(Mean_DASM=("DASM", "mean"),
                    Mean_Abs_DASM=("Abs_DASM", "mean"),
                    Mean_Log_RASM=("Log_RASM", "mean"),
                    Mean_Abs_Log_RASM=("Abs_Log_RASM", "mean")))
    summary["Geometric_Mean_RASM"] = np.exp(summary["Mean_Log_RASM"])
    summary["DASM_unit"] = "uV2"

    dasm_ranking = summary.sort_values(["Band", "Mean_Abs_DASM"], ascending=[True, False]).copy()
    dasm_ranking["DASM_Rank"] = dasm_ranking.groupby("Band").cumcount() + 1
    rasm_ranking = summary.sort_values(["Band", "Mean_Abs_Log_RASM"], ascending=[True, False]).copy()
    rasm_ranking["RASM_Rank"] = rasm_ranking.groupby("Band").cumcount() + 1
    top3_dasm = dasm_ranking[dasm_ranking["DASM_Rank"] <= 3].copy()
    top3_rasm = rasm_ranking[rasm_ranking["RASM_Rank"] <= 3].copy()

    band_summary = (asymmetry_df.groupby("Band", as_index=False)
                    .agg(Mean_DASM=("DASM", "mean"),
                         SD_DASM=("DASM", "std"),
                         Median_DASM=("DASM", "median"),
                         Mean_Abs_DASM=("Abs_DASM", "mean"),
                         Mean_Log_RASM=("Log_RASM", "mean"),
                         SD_Log_RASM=("Log_RASM", "std"),
                         Median_Log_RASM=("Log_RASM", "median"),
                         Mean_Abs_Log_RASM=("Abs_Log_RASM", "mean")))
    band_summary["Geometric_Mean_RASM"] = np.exp(band_summary["Mean_Log_RASM"])

    band_type = pd.CategoricalDtype(categories=BANDS, ordered=True)
    for table in [asymmetry_df, summary, top3_dasm, top3_rasm, band_summary]:
        table["Band"] = table["Band"].astype(band_type)
    asymmetry_df = asymmetry_df.sort_values(["Code", "Stimulus", "Band", "Pair"]).reset_index(drop=True)
    summary = summary.sort_values(["Code", "Band"]).reset_index(drop=True)
    top3_dasm = top3_dasm.sort_values(["Band", "DASM_Rank"]).reset_index(drop=True)
    top3_rasm = top3_rasm.sort_values(["Band", "RASM_Rank"]).reset_index(drop=True)
    band_summary = band_summary.sort_values("Band").reset_index(drop=True)

    # Unit verification against the reference values used in manuscript Table 10.
    published_first_dasm = {"Delta": (27, 46542.59), "Theta": (24, 11047.93),
                            "Alpha": (24, 13571.38), "Beta": (24, 21596.52),
                            "LowGamma": (24, 13545.57)}
    for band, (expected_code, expected_value) in published_first_dasm.items():
        first = top3_dasm[top3_dasm["Band"] == band].iloc[0]
        if int(first["Code"]) != expected_code or not np.isclose(
            float(first["Mean_Abs_DASM"]), expected_value, rtol=0, atol=0.015
        ):
            raise AssertionError(f"Table 10 mismatch for {band}; inspect input and power units")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "RQ3_DASM_RASM_long.csv": asymmetry_df,
        "RQ3_participant_band_summary.csv": summary,
        "RQ3_TOP3_DASM.csv": top3_dasm,
        "RQ3_TOP3_RASM.csv": top3_rasm,
        "RQ3_band_summary.csv": band_summary,
    }
    for filename, table in outputs.items():
        table.to_csv(OUTPUT_DIR / filename, index=False)
    excel_file = OUTPUT_DIR / "RQ3_DASM_RASM_FINAL_N35.xlsx"
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        for sheet, table in [("DASM_RASM_long", asymmetry_df), ("Participant_band", summary),
                             ("Top3_DASM", top3_dasm), ("Top3_RASM", top3_rasm),
                             ("Band_summary", band_summary)]:
            table.to_excel(writer, sheet_name=sheet, index=False)
    print("RQ3 complete; 35 participants, 5250 paired observations")
    print("DASM/P_L/P_R units: µV² (V² × 10¹²); RASM/log(RASM) dimensionless")
    print("Table 10 top-1 DASM values verified in all five bands")
    print(top3_dasm[["Band", "DASM_Rank", "Code", "Mean_Abs_DASM"]].to_string(index=False))
    print("Files saved to:", OUTPUT_DIR)
    print("Run 06_RQ3_sensitivity.py next to reproduce Appendix A.")


if __name__ == "__main__":
    main()
