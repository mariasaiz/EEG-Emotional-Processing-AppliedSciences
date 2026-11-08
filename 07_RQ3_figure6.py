# Supplementary analysis code – Applied Sciences EEG study
# Curated from the final analysis cells of PROLEPSIS(4).ipynb.
# Only code is distributed; no participant-level recordings or feature matrices are included.
# The variable EEG_PROJECT_DIR may point to a local research-data directory.
# Refer to README.md for data requirements, limitations and execution order.
import os
from pathlib import Path
PROJECT_DATA = Path(os.environ.get("EEG_PROJECT_DIR", str(Path(__file__).resolve().parent))).expanduser()
PREPROCESS_DIR = PROJECT_DATA / "OUTPUT_FINAL_UNIFIED"
SOCIO_DIR = PROJECT_DATA / "OUTPUT_PHASE_C1"
ASYMMETRY_DIR = PROJECT_DATA / "OUTPUT_RQ3_ASYMMETRY"

# 07: Figure 6 signed-mean heat maps for DASM and ln(RASM).
# ============================================================
# FIGURE 6 - HEAT MAPS DASM + ln(RASM)
# FINAL RQ3 - N = 35
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================

# ============================================================

INPUT_FILE = ASYMMETRY_DIR / "RQ3_DASM_RASM_FINAL_N35.xlsx"

OUTPUT_DIR = ASYMMETRY_DIR

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================

# ============================================================

BANDS = [
    "Delta",
    "Theta",
    "Alpha",
    "Beta",
    "LowGamma"
]


# ============================================================

# ============================================================

participant_band = pd.read_excel(
    INPUT_FILE,
    sheet_name="Participant_band"
)

top3_dasm = pd.read_excel(
    INPUT_FILE,
    sheet_name="Top3_DASM"
)

top3_rasm = pd.read_excel(
    INPUT_FILE,
    sheet_name="Top3_RASM"
)


# ============================================================
# 4. VALIDATION CHECKS
# ============================================================

required_summary_cols = {
    "Code",
    "Band",
    "Mean_DASM",
    "Mean_Log_RASM"
}

missing = required_summary_cols - set(
    participant_band.columns
)

if missing:
    raise ValueError(
        f"Required Participant_band columns missing: {sorted(missing)}"
    )

if "Code" not in top3_dasm.columns:
    raise ValueError(
        "The Code column is missing from Top3_DASM."
    )

if "Code" not in top3_rasm.columns:
    raise ValueError(
        "The Code column is missing from Top3_RASM."
    )


# ============================================================

# ============================================================

selected_codes = sorted(
    set(
        top3_dasm["Code"]
        .dropna()
        .astype(int)
        .tolist()
    )
    |
    set(
        top3_rasm["Code"]
        .dropna()
        .astype(int)
        .tolist()
    )
)

print("=" * 100)
print("PARTICIPANTS SELECTED FOR FIGURE 6")
print("=" * 100)

print(selected_codes)

print(
    "\nNumber of unique participants:",
    len(selected_codes)
)


# ============================================================

# ============================================================

participant_band["Participant"] = (
    "P0_"
    +
    participant_band["Code"]
    .astype(int)
    .astype(str)
)


# ============================================================

# ============================================================

plot_df = (
    participant_band
    .loc[
        participant_band["Code"]
        .astype(int)
        .isin(selected_codes)
    ]
    .copy()
)


# ============================================================

# ============================================================

plot_df["Band"] = pd.Categorical(
    plot_df["Band"],
    categories=BANDS,
    ordered=True
)


# ============================================================
# 9. DASM MATRIX
# ============================================================

dasm_matrix = (
    plot_df
    .pivot(
        index="Participant",
        columns="Band",
        values="Mean_DASM"
    )
)


# ============================================================
# 10. ln(RASM) MATRIX
# ============================================================

ln_rasm_matrix = (
    plot_df
    .pivot(
        index="Participant",
        columns="Band",
        values="Mean_Log_RASM"
    )
)


# ============================================================

# ============================================================

ordering_df = (
    plot_df
    .groupby(
        "Participant",
        as_index=False
    )
    .agg(
        MaxAbsDASM=(
            "Mean_DASM",
            lambda x: np.max(
                np.abs(x)
            )
        ),
        MaxAbsLnRASM=(
            "Mean_Log_RASM",
            lambda x: np.max(
                np.abs(x)
            )
        )
    )
)


ordering_df = (
    ordering_df
    .sort_values(
        [
            "MaxAbsDASM",
            "MaxAbsLnRASM"
        ],
        ascending=[
            False,
            False
        ]
    )
)


participant_order = (
    ordering_df[
        "Participant"
    ]
    .tolist()
)


dasm_matrix = (
    dasm_matrix
    .reindex(
        index=participant_order,
        columns=BANDS
    )
)


ln_rasm_matrix = (
    ln_rasm_matrix
    .reindex(
        index=participant_order,
        columns=BANDS
    )
)


# ============================================================

# ============================================================

if dasm_matrix.isna().any().any():
    raise ValueError(
        "The DASM matrix contains missing values."
    )


if ln_rasm_matrix.isna().any().any():
    raise ValueError(
        "The ln(RASM) matrix contains missing values."
    )


print("\nShape DASM:", dasm_matrix.shape)
print("Shape ln(RASM):", ln_rasm_matrix.shape)


# ============================================================

# ============================================================

dasm_lim = np.max(
    np.abs(
        dasm_matrix.to_numpy()
    )
)


ln_rasm_lim = np.max(
    np.abs(
        ln_rasm_matrix.to_numpy()
    )
)


print(
    "\nDASM symmetric limit:",
    dasm_lim
)

print(
    "ln(RASM) symmetric limit:",
    ln_rasm_lim
)


# ============================================================

# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(
        13,
        6.8
    ),
    constrained_layout=True
)


# ============================================================
# 15. PANEL A - DASM
# ============================================================

im1 = axes[0].imshow(
    dasm_matrix.to_numpy(),
    aspect="auto",
    vmin=-dasm_lim,
    vmax=dasm_lim,
    cmap="coolwarm"
)


axes[0].set_title(
    "(a) DASM"
)


axes[0].set_xlabel(
    "Frequency band"
)

axes[0].set_ylabel(
    "Participant"
)


axes[0].set_xticks(
    np.arange(
        len(BANDS)
    )
)

axes[0].set_xticklabels(
    [
        "Delta",
        "Theta",
        "Alpha",
        "Beta",
        "Low-Gamma"
    ],
    rotation=45,
    ha="right"
)


axes[0].set_yticks(
    np.arange(
        len(participant_order)
    )
)

axes[0].set_yticklabels(
    participant_order
)


for i in range(
    dasm_matrix.shape[0]
):

    for j in range(
        dasm_matrix.shape[1]
    ):

        value = (
            dasm_matrix
            .iloc[
                i,
                j
            ]
        )

        axes[0].text(
            j,
            i,
            f"{value:.2e}",
            ha="center",
            va="center",
            fontsize=7
        )


cbar1 = fig.colorbar(
    im1,
    ax=axes[0],
    shrink=0.90
)

cbar1.set_label(
    "DASM"
)


# ============================================================
# 16. PANEL B - ln(RASM)
# ============================================================

im2 = axes[1].imshow(
    ln_rasm_matrix.to_numpy(),
    aspect="auto",
    vmin=-ln_rasm_lim,
    vmax=ln_rasm_lim,
    cmap="coolwarm"
)


axes[1].set_title(
    "(b) ln(RASM)"
)


axes[1].set_xlabel(
    "Frequency band"
)

axes[1].set_ylabel(
    ""
)


axes[1].set_xticks(
    np.arange(
        len(BANDS)
    )
)

axes[1].set_xticklabels(
    [
        "Delta",
        "Theta",
        "Alpha",
        "Beta",
        "Low-Gamma"
    ],
    rotation=45,
    ha="right"
)


axes[1].set_yticks(
    np.arange(
        len(participant_order)
    )
)

axes[1].set_yticklabels(
    participant_order
)


for i in range(
    ln_rasm_matrix.shape[0]
):

    for j in range(
        ln_rasm_matrix.shape[1]
    ):

        value = (
            ln_rasm_matrix
            .iloc[
                i,
                j
            ]
        )

        axes[1].text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=7
        )


cbar2 = fig.colorbar(
    im2,
    ax=axes[1],
    shrink=0.90
)

cbar2.set_label(
    "ln(RASM)"
)


# ============================================================

# ============================================================

png_file = (
    OUTPUT_DIR
    /
    "Figure_6_DASM_lnRASM_FINAL_N35.png"
)

pdf_file = (
    OUTPUT_DIR
    /
    "Figure_6_DASM_lnRASM_FINAL_N35.pdf"
)


fig.savefig(
    png_file,
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    pdf_file,
    bbox_inches="tight"
)


plt.show()


# ============================================================

# ============================================================

print("\n")
print("=" * 100)
print("FIGURE 6 GENERATED")
print("=" * 100)

print(
    "\nPNG:"
)

print(
    png_file
)

print(
    "\nPDF:"
)

print(
    pdf_file
)

print(
    "\nParticipants included:"
)

print(
    participant_order
)