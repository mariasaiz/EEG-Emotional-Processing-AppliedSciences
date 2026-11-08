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

# 05: Participant-band hemispheric asymmetry using the manuscript definitions: DASM=R-L; RASM=R/L.
# ============================================================
# RQ3 - HEMISPHERIC EEG ASYMMETRY
# DASM = P_R - P_L
# RASM = P_R / P_L
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================

# ============================================================

INPUT_FILE = PREPROCESS_DIR / "BandPower_channel_level.csv"

OUTPUT_DIR = ASYMMETRY_DIR

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================

#

# LEFT -> RIGHT
#
# DASM = RIGHT - LEFT
# RASM = RIGHT / LEFT
# ============================================================

HOMOLOGOUS_PAIRS = {
    "AF7-AF8": ("AF7", "AF8"),
    "Fp1-Fp2": ("Fp1", "Fp2"),
    "F3-F4": ("F3", "F4"),
    "P3-P4": ("P3", "P4"),
    "P7-P8": ("P7", "P8"),
    "O1-O2": ("O1", "O2"),
}


EXPECTED_STIMULI = [
    "Surprise",
    "Anger",
    "Happiness",
    "Sadness",
    "Fear"
]


EXPECTED_BANDS = [
    "Delta",
    "Theta",
    "Alpha",
    "Beta",
    "LowGamma"
]


EXPECTED_N_PARTICIPANTS = 35


# ============================================================

# ============================================================

df = pd.read_csv(
    INPUT_FILE
)


print("=" * 100)
print("RQ3 - INPUT FILE CHECK")
print("=" * 100)

print("File:", INPUT_FILE)
print("Shape:", df.shape)
print("Participants:", df["Code"].nunique())
print("Stimuli:", df["Stimulus"].unique())
print("Bands:", df["Band"].unique())
print("Channels:", df["Channel"].unique())


# ============================================================
# 4. INITIAL VALIDATION CHECKS
# ============================================================

required_columns = {
    "Code",
    "Stimulus",
    "Event_code",
    "Channel",
    "Band",
    "Power",
    "Power_unit"
}


missing_columns = (
    required_columns
    -
    set(df.columns)
)


if missing_columns:

    raise ValueError(
        f"Required columns missing: "
        f"{sorted(missing_columns)}"
    )


if df["Code"].nunique() != EXPECTED_N_PARTICIPANTS:

    raise ValueError(
        f"Expected {EXPECTED_N_PARTICIPANTS} participants "
        f"but found {df['Code'].nunique()}."
    )


if sorted(df["Stimulus"].unique()) != sorted(EXPECTED_STIMULI):

    raise ValueError(
        "Stimulus labels do not match the expected values."
    )


if sorted(df["Band"].unique()) != sorted(EXPECTED_BANDS):

    raise ValueError(
        "Frequency bands do not match the expected values."
    )


# ============================================================

# ============================================================

if df["Power"].isna().any():

    raise ValueError(
        "Missing values found in Power."
    )


if not np.isfinite(
    df["Power"].to_numpy()
).all():

    raise ValueError(
        "Non-finite values found in Power."
    )


if (
    df["Power"] <= 0
).any():

    problematic = df.loc[
        df["Power"] <= 0
    ]

    raise ValueError(
        "Non-positive spectral power values found. "
        "RASM = P_R/P_L requires positive power values.\n"
        f"{problematic.head()}"
    )


print("\nAll spectral power values are positive and finite.")


# ============================================================
# 6. CONVERT TO WIDE FORMAT
#
# One row:

#
# Columns:
# AF7, Fp1, Fp2, ...
# ============================================================

wide = (
    df
    .pivot_table(
        index=[
            "Code",
            "Stimulus",
            "Event_code",
            "Band"
        ],
        columns="Channel",
        values="Power",
        aggfunc="first"
    )
    .reset_index()
)


# ============================================================

# ============================================================

required_channels = set()

for left_ch, right_ch in HOMOLOGOUS_PAIRS.values():

    required_channels.add(
        left_ch
    )

    required_channels.add(
        right_ch
    )


missing_channels = (
    required_channels
    -
    set(wide.columns)
)


if missing_channels:

    raise ValueError(
        f"Required EEG channels missing for RQ3: "
        f"{sorted(missing_channels)}"
    )


expected_rows_wide = (
    EXPECTED_N_PARTICIPANTS
    *
    len(EXPECTED_STIMULI)
    *
    len(EXPECTED_BANDS)
)


if len(wide) != expected_rows_wide:

    raise ValueError(
        f"Expected {expected_rows_wide} rows in wide format "
        f"but the number obtained was {len(wide)}."
    )


# ============================================================
# 8. CALCULATE DASM AND RASM
#



# ============================================================

asymmetry_records = []


for _, row in wide.iterrows():

    for pair_name, (
        left_channel,
        right_channel
    ) in HOMOLOGOUS_PAIRS.items():

        p_left = float(
            row[
                left_channel
            ]
        )

        p_right = float(
            row[
                right_channel
            ]
        )


        # ----------------------------------------------------
        # DASM
        # ----------------------------------------------------

        dasm = (
            p_right
            -
            p_left
        )


        # ----------------------------------------------------
        # RASM
        # ----------------------------------------------------

        rasm = (
            p_right
            /
            p_left
        )


        # ----------------------------------------------------
        # LOG-RATIO
        #

        # log(RASM) > 0   -> greater right-hemisphere power
        # log(RASM) < 0   -> greater left-hemisphere power
        #


        # ----------------------------------------------------

        log_rasm = np.log(
            rasm
        )

        abs_log_rasm = abs(
            log_rasm
        )


        asymmetry_records.append(
            {
                "Code": int(
                    row["Code"]
                ),
                "Stimulus": row[
                    "Stimulus"
                ],
                "Event_code": int(
                    row["Event_code"]
                ),
                "Band": row[
                    "Band"
                ],
                "Pair": pair_name,
                "Left_channel": left_channel,
                "Right_channel": right_channel,
                "P_L": p_left,
                "P_R": p_right,
                "DASM": dasm,
                "Abs_DASM": abs(
                    dasm
                ),
                "RASM": rasm,
                "Log_RASM": log_rasm,
                "Abs_Log_RASM": abs_log_rasm
            }
        )


asymmetry_df = pd.DataFrame(
    asymmetry_records
)


# ============================================================

# ============================================================

expected_asymmetry_rows = (
    EXPECTED_N_PARTICIPANTS
    *
    len(EXPECTED_STIMULI)
    *
    len(EXPECTED_BANDS)
    *
    len(HOMOLOGOUS_PAIRS)
)


print("\n")
print("=" * 100)
print("CALCULATION VALIDATION")
print("=" * 100)

print(
    "Observed rows:",
    len(asymmetry_df)
)

print(
    "Expected rows:",
    expected_asymmetry_rows
)


if len(
    asymmetry_df
) != expected_asymmetry_rows:

    raise ValueError(
        f"Expected "
        f"{expected_asymmetry_rows} rows "
        f"but the number obtained was "
        f"{len(asymmetry_df)}."
    )


if asymmetry_df[
    [
        "DASM",
        "RASM",
        "Log_RASM",
        "Abs_DASM",
        "Abs_Log_RASM"
    ]
].isna().any().any():

    raise ValueError(
        "Missing values were found "
        "in the asymmetry indices."
    )


if not np.isfinite(
    asymmetry_df[
        [
            "DASM",
            "RASM",
            "Log_RASM",
            "Abs_DASM",
            "Abs_Log_RASM"
        ]
    ].to_numpy()
).all():

    raise ValueError(
        "Non-finite values were found "
        "in the asymmetry indices."
    )


print("DASM/RASM calculations: OK")


# ============================================================

#
# Each value summarises:

#
# DASM:


#
# RASM:



# ============================================================

participant_band_summary = (
    asymmetry_df
    .groupby(
        [
            "Code",
            "Band"
        ],
        as_index=False
    )
    .agg(
        Mean_DASM=(
            "DASM",
            "mean"
        ),
        Mean_Abs_DASM=(
            "Abs_DASM",
            "mean"
        ),
        Mean_Log_RASM=(
            "Log_RASM",
            "mean"
        ),
        Mean_Abs_Log_RASM=(
            "Abs_Log_RASM",
            "mean"
        )
    )
)



# exp(mean log RASM)

participant_band_summary[
    "Geometric_Mean_RASM"
] = np.exp(
    participant_band_summary[
        "Mean_Log_RASM"
    ]
)


# ============================================================
# 11. RANKING DESCRIPTIVO DASM
#
# Larger Mean_Abs_DASM =

# ============================================================

dasm_ranking = (
    participant_band_summary
    .sort_values(
        [
            "Band",
            "Mean_Abs_DASM"
        ],
        ascending=[
            True,
            False
        ]
    )
    .copy()
)


dasm_ranking[
    "DASM_Rank"
] = (
    dasm_ranking
    .groupby(
        "Band"
    )
    .cumcount()
    +
    1
)


# ============================================================
# 12. RANKING DESCRIPTIVO RASM
#

#

#
# RASM = 2
# RASM = 0.5
#

# ============================================================

rasm_ranking = (
    participant_band_summary
    .sort_values(
        [
            "Band",
            "Mean_Abs_Log_RASM"
        ],
        ascending=[
            True,
            False
        ]
    )
    .copy()
)


rasm_ranking[
    "RASM_Rank"
] = (
    rasm_ranking
    .groupby(
        "Band"
    )
    .cumcount()
    +
    1
)


# ============================================================
# 13. TOP THREE BY FREQUENCY BAND
# ============================================================

top3_dasm = (
    dasm_ranking
    .loc[
        dasm_ranking[
            "DASM_Rank"
        ] <= 3
    ]
    .copy()
)


top3_rasm = (
    rasm_ranking
    .loc[
        rasm_ranking[
            "RASM_Rank"
        ] <= 3
    ]
    .copy()
)


# ============================================================
# 14. COMBINED TOP-THREE TABLE
# ============================================================

top3_dasm_table = (
    top3_dasm[
        [
            "Band",
            "DASM_Rank",
            "Code",
            "Mean_DASM",
            "Mean_Abs_DASM"
        ]
    ]
    .rename(
        columns={
            "Code": "DASM_Code"
        }
    )
)


top3_rasm_table = (
    top3_rasm[
        [
            "Band",
            "RASM_Rank",
            "Code",
            "Geometric_Mean_RASM",
            "Mean_Log_RASM",
            "Mean_Abs_Log_RASM"
        ]
    ]
    .rename(
        columns={
            "Code": "RASM_Code"
        }
    )
)


top3_combined = pd.merge(
    top3_dasm_table,
    top3_rasm_table,
    left_on=[
        "Band",
        "DASM_Rank"
    ],
    right_on=[
        "Band",
        "RASM_Rank"
    ],
    how="outer"
)


# ============================================================

#
# Solo descriptivo
# ============================================================

band_summary = (
    asymmetry_df
    .groupby(
        "Band",
        as_index=False
    )
    .agg(
        Mean_DASM=(
            "DASM",
            "mean"
        ),
        SD_DASM=(
            "DASM",
            "std"
        ),
        Median_DASM=(
            "DASM",
            "median"
        ),
        Mean_Abs_DASM=(
            "Abs_DASM",
            "mean"
        ),
        Mean_Log_RASM=(
            "Log_RASM",
            "mean"
        ),
        SD_Log_RASM=(
            "Log_RASM",
            "std"
        ),
        Median_Log_RASM=(
            "Log_RASM",
            "median"
        ),
        Mean_Abs_Log_RASM=(
            "Abs_Log_RASM",
            "mean"
        )
    )
)


band_summary[
    "Geometric_Mean_RASM"
] = np.exp(
    band_summary[
        "Mean_Log_RASM"
    ]
)


# ============================================================

# ============================================================

band_order = pd.CategoricalDtype(
    categories=EXPECTED_BANDS,
    ordered=True
)


for table in [
    asymmetry_df,
    participant_band_summary,
    dasm_ranking,
    rasm_ranking,
    top3_dasm,
    top3_rasm,
    top3_combined,
    band_summary
]:

    if "Band" in table.columns:

        table["Band"] = table[
            "Band"
        ].astype(
            band_order
        )


asymmetry_df = (
    asymmetry_df
    .sort_values(
        [
            "Code",
            "Stimulus",
            "Band",
            "Pair"
        ]
    )
    .reset_index(
        drop=True
    )
)


participant_band_summary = (
    participant_band_summary
    .sort_values(
        [
            "Code",
            "Band"
        ]
    )
    .reset_index(
        drop=True
    )
)


top3_dasm = (
    top3_dasm
    .sort_values(
        [
            "Band",
            "DASM_Rank"
        ]
    )
    .reset_index(
        drop=True
    )
)


top3_rasm = (
    top3_rasm
    .sort_values(
        [
            "Band",
            "RASM_Rank"
        ]
    )
    .reset_index(
        drop=True
    )
)


band_summary = (
    band_summary
    .sort_values(
        "Band"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================

# ============================================================

asymmetry_file = (
    OUTPUT_DIR
    /
    "RQ3_DASM_RASM_long.csv"
)

participant_summary_file = (
    OUTPUT_DIR
    /
    "RQ3_participant_band_summary.csv"
)

top3_dasm_file = (
    OUTPUT_DIR
    /
    "RQ3_TOP3_DASM.csv"
)

top3_rasm_file = (
    OUTPUT_DIR
    /
    "RQ3_TOP3_RASM.csv"
)

band_summary_file = (
    OUTPUT_DIR
    /
    "RQ3_band_summary.csv"
)


asymmetry_df.to_csv(
    asymmetry_file,
    index=False
)

participant_band_summary.to_csv(
    participant_summary_file,
    index=False
)

top3_dasm.to_csv(
    top3_dasm_file,
    index=False
)

top3_rasm.to_csv(
    top3_rasm_file,
    index=False
)

band_summary.to_csv(
    band_summary_file,
    index=False
)


# ============================================================

# ============================================================

excel_file = (
    OUTPUT_DIR
    /
    "RQ3_DASM_RASM_FINAL_N35.xlsx"
)


with pd.ExcelWriter(
    excel_file,
    engine="openpyxl"
) as writer:

    asymmetry_df.to_excel(
        writer,
        sheet_name="DASM_RASM_long",
        index=False
    )

    participant_band_summary.to_excel(
        writer,
        sheet_name="Participant_band",
        index=False
    )

    top3_dasm.to_excel(
        writer,
        sheet_name="Top3_DASM",
        index=False
    )

    top3_rasm.to_excel(
        writer,
        sheet_name="Top3_RASM",
        index=False
    )

    band_summary.to_excel(
        writer,
        sheet_name="Band_summary",
        index=False
    )


# ============================================================
# 19. MOSTRAR RESULTADOS PRINCIPALES
# ============================================================

print("\n")
print("=" * 100)
print("TOP THREE DASM VALUES BY FREQUENCY BAND")
print("=" * 100)

print(
    top3_dasm[
        [
            "Band",
            "DASM_Rank",
            "Code",
            "Mean_DASM",
            "Mean_Abs_DASM"
        ]
    ].to_string(
        index=False
    )
)


print("\n")
print("=" * 100)
print("TOP THREE RASM VALUES BY FREQUENCY BAND")
print("=" * 100)

print(
    top3_rasm[
        [
            "Band",
            "RASM_Rank",
            "Code",
            "Geometric_Mean_RASM",
            "Mean_Log_RASM",
            "Mean_Abs_Log_RASM"
        ]
    ].to_string(
        index=False
    )
)


print("\n")
print("=" * 100)
print("SUMMARY BY FREQUENCY BAND")
print("=" * 100)

print(
    band_summary.to_string(
        index=False
    )
)


# ============================================================

# ============================================================

print("\n")
print("=" * 100)
print("RQ3 COMPLETED")
print("=" * 100)

print(
    "\nArchivo principal:"
)

print(
    excel_file
)

print(
    "\nDatos largos DASM/RASM:"
)

print(
    asymmetry_file
)

print(
    "\nParticipant-by-band summary:"
)

print(
    participant_summary_file
)

print(
    "\nTop 3 DASM:"
)

print(
    top3_dasm_file
)

print(
    "\nTop 3 RASM:"
)

print(
    top3_rasm_file
)