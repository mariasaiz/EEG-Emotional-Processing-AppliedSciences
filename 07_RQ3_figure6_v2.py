# Supplementary analysis code – Applied Sciences EEG study
# Reproducible Figure 6: all 35 participants; signed DASM (µV²) and ln(RASM).
# Uses the outputs of 05_RQ3_asymmetry_V3.py (no private paths).
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, TwoSlopeNorm

PROJECT_DATA = Path(os.environ.get("EEG_PROJECT_DIR", str(Path(__file__).resolve().parent))).expanduser().resolve()
ASYMMETRY_DIR = PROJECT_DATA / "OUTPUT_RQ3_ASYMMETRY"
INPUT_FILE = ASYMMETRY_DIR / "RQ3_participant_band_summary.csv"
OUTPUT_DIR = ASYMMETRY_DIR
BANDS = ["Delta", "Theta", "Alpha", "Beta", "LowGamma"]
BAND_LABELS = ["Delta", "Theta", "Alpha", "Beta", "Low gamma"]


def main() -> None:
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"Missing participant-band summary: {INPUT_FILE}. "
            "Run 05_RQ3_asymmetry_V3.py first."
        )
    summary = pd.read_csv(INPUT_FILE)
    required = {"Code", "Band", "Mean_DASM", "Mean_Log_RASM"}
    if not required.issubset(summary.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(summary.columns))}")
    codes = sorted(summary["Code"].astype(int).unique())
    if len(codes) != 35 or 32 in codes:
        raise ValueError("Figure 6 requires 35 participants, excluding participant 32")
    if len(summary) != 175 or summary.duplicated(["Code", "Band"]).any():
        raise ValueError("Expected 175 unique participant-band observations")
    if set(summary["Band"]) != set(BANDS):
        raise ValueError("Unexpected frequency bands")
    dasm = summary.pivot(index="Code", columns="Band", values="Mean_DASM").reindex(index=codes, columns=BANDS)
    lrasm = summary.pivot(index="Code", columns="Band", values="Mean_Log_RASM").reindex(index=codes, columns=BANDS)
    if dasm.isna().any().any() or lrasm.isna().any().any():
        raise ValueError("Missing participant-band values")
    D, R = dasm.to_numpy(dtype=float), lrasm.to_numpy(dtype=float)
    if not np.isfinite(D).all() or not np.isfinite(R).all():
        raise ValueError("Non-finite asymmetry values")
    nonzero = np.abs(D[D != 0])
    if nonzero.size == 0:
        raise ValueError("Cannot construct SymLogNorm from all-zero DASM")
    linthresh = float(np.percentile(nonzero, 20))
    dmax, rmax = float(np.max(np.abs(D))), float(np.max(np.abs(R)))
    if rmax <= 0:
        raise ValueError("Cannot construct ln(RASM) colour scale from all zeros")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 9.6), sharey=True, constrained_layout=True)
    labels = [f"P0_{code}" for code in codes]
    im1 = ax1.imshow(D, aspect="auto", cmap="RdBu_r", interpolation="nearest",
                     norm=SymLogNorm(linthresh=linthresh, linscale=1,
                                     vmin=-dmax, vmax=dmax, base=10))
    im2 = ax2.imshow(R, aspect="auto", cmap="RdBu_r", interpolation="nearest",
                     norm=TwoSlopeNorm(vcenter=0, vmin=-rmax, vmax=rmax))
    for ax in (ax1, ax2):
        ax.set_xticks(range(len(BANDS)))
        ax.set_xticklabels(BAND_LABELS, rotation=35, ha="right")
        ax.set_yticks(range(len(codes)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.tick_params(axis="both", length=0)
        ax.set_xlabel("Frequency band")
    ax1.set_ylabel("Participant")
    ax1.set_title("Differential asymmetry (DASM)")
    ax2.set_title("Log-transformed power ratio (ln RASM)")
    fig.colorbar(im1, ax=ax1, fraction=0.045, pad=0.025).set_label(
        "Mean signed DASM (µV²; symmetric log scale)")
    fig.colorbar(im2, ax=ax2, fraction=0.045, pad=0.025).set_label(
        "Mean signed ln(RASM) (dimensionless)")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "tiff", "pdf"):
        target = OUTPUT_DIR / f"Figure6_DASM_lnRASM_N35_corrected.{extension}"
        fig.savefig(target, dpi=350 if extension != "pdf" else None,
                    bbox_inches="tight")
        print("Saved:", target)
    plt.close(fig)
    print("Figure 6 validated: N=35; DASM in µV²; ln(RASM) dimensionless")


if __name__ == "__main__":
    main()
