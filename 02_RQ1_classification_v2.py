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

# 02: LOOCV, within-fold scaling/feature selection, four classifiers and bootstrap CIs.
# ============================================================
# RQ1 - FINAL MACHINE-LEARNING ANALYSIS
# Final sample N = 35
# Students = 10 | Lecturers = 25
#
# - 25 EEG spectral-power features
# - LOOCV
# - RobustScaler within each training fold
# - SelectKBest(f_classif, k=8) within each training fold
# - RF, Logistic Regression, KNN, BPNN/MLP
# - Balanced Accuracy, Macro-F1, AUC
# - 95% bootstrap CI (5000 valid resamples)
# - Feature-selection frequency across LOOCV folds
# - AUC from continuous out-of-fold probabilities
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.base import clone
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# 1. PATHS
# ============================================================

EEG_FILE = PREPROCESS_DIR / "BandPower_stimulus_band_mean.csv"

SOCIO_FILE = SOCIO_DIR / "BBDD_SOCIO_FINAL_N35.xlsx"

OUTPUT = SOCIO_DIR

OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

eeg = pd.read_csv(EEG_FILE)
socio = pd.read_excel(SOCIO_FILE)

print("=" * 75)
print("INPUT DATA")
print("=" * 75)

print("EEG shape:", eeg.shape)
print("Socio shape:", socio.shape)

print("\nEEG columns:")
print(eeg.columns.tolist())

print("\nSocio columns:")
print(socio.columns.tolist())


# ============================================================
# 3. VALIDATE FINAL COHORT
# ============================================================

EXPECTED_CODES = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    33, 34, 35, 37, 38
]

assert sorted(eeg["Code"].unique()) == EXPECTED_CODES
assert sorted(socio["Code"].unique()) == EXPECTED_CODES

assert eeg["Code"].nunique() == 35
assert socio["Code"].nunique() == 35

assert len(eeg) == 35 * 5 * 5

print("\nFinal cohort validated: N = 35")


# ============================================================
# 4. CREATE THE 25 EEG FEATURES
# ============================================================

stimulus_order = [
    "Surprise",
    "Anger",
    "Happiness",
    "Sadness",
    "Fear"
]

band_order = [
    "Delta",
    "Theta",
    "Alpha",
    "Beta",
    "LowGamma"
]

# Validate expected stimuli and bands
assert set(eeg["Stimulus"].unique()) == set(stimulus_order)
assert set(eeg["Band"].unique()) == set(band_order)

# Create feature names
eeg["Feature"] = (
    eeg["Stimulus"].astype(str)
    + "_"
    + eeg["Band"].astype(str)
)

# Participant × Feature matrix
X_df = eeg.pivot(
    index="Code",
    columns="Feature",
    values="Mean_power_12ch"
)

# Force scientifically meaningful order
feature_names = [
    f"{stimulus}_{band}"
    for stimulus in stimulus_order
    for band in band_order
]

X_df = X_df[feature_names]

assert X_df.shape == (35, 25)
assert X_df.isna().sum().sum() == 0

print("\nFeature matrix:", X_df.shape)
print("25 features:")
print(feature_names)


# ============================================================
# 5. CREATE TARGET VARIABLE
#
# 0 = Students
# 1 = Lecturers
# ============================================================

demo = socio[
    ["Code", "Group_label"]
].copy()

demo = demo.set_index("Code").loc[X_df.index]

print("\nGroup labels:")
print(demo["Group_label"].value_counts())

mapping = {
    "Students": 0,
    "Lecturers": 1
}

y = (
    demo["Group_label"]
    .map(mapping)
    .astype(int)
    .to_numpy()
)

X = X_df.to_numpy(dtype=float)

assert len(X) == 35
assert len(y) == 35

assert np.sum(y == 0) == 10
assert np.sum(y == 1) == 25

print("\nTarget validated:")
print("Students =", np.sum(y == 0))
print("Lecturers =", np.sum(y == 1))


# ============================================================
# 6. CLASSIFIERS
# ============================================================

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=5000,
        solver="liblinear",
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "BPNN (MLP)": MLPClassifier(
        hidden_layer_sizes=(100,),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        max_iter=5000,
        random_state=42
    )
}


# ============================================================
# 7. LOOCV
# ============================================================

loo = LeaveOneOut()

n = len(y)

# Out-of-fold predicted class
oof_pred = {
    name: np.zeros(n, dtype=int)
    for name in models
}

# Out-of-fold continuous probability for Lecturer class
oof_prob = {
    name: np.zeros(n, dtype=float)
    for name in models
}

# Count how often each feature is selected
selection_counts = np.zeros(
    len(feature_names),
    dtype=int
)

print("\n" + "=" * 75)
print("STARTING LOOCV")
print("=" * 75)


for fold, (train_idx, test_idx) in enumerate(
    loo.split(X),
    start=1
):

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]

    # --------------------------------------------------------
    # RobustScaler:
    # fit ONLY on training participants
    # --------------------------------------------------------

    scaler = RobustScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # --------------------------------------------------------
    # SelectKBest:
    # fit ONLY on training participants
    # --------------------------------------------------------

    selector = SelectKBest(
        score_func=f_classif,
        k=8
    )

    X_train_selected = selector.fit_transform(
        X_train_scaled,
        y_train
    )

    X_test_selected = selector.transform(
        X_test_scaled
    )

    # Record selected features
    selected_mask = selector.get_support()

    selection_counts += selected_mask.astype(int)


    # --------------------------------------------------------
    # Train/test all classifiers
    # --------------------------------------------------------

    for name, base_model in models.items():

        model = clone(base_model)

        model.fit(
            X_train_selected,
            y_train
        )

        # Predicted class
        pred = model.predict(
            X_test_selected
        )[0]

        # Continuous probability:
        # class 1 = Lecturer
        prob = model.predict_proba(
            X_test_selected
        )[0, 1]

        oof_pred[name][test_idx[0]] = pred
        oof_prob[name][test_idx[0]] = prob


    print(f"Fold {fold:02d}/35 completed")


print("\nLOOCV completed successfully.")


# ============================================================
# 8. POINT ESTIMATES
# ============================================================

point_results = []

for name in models:

    BA = balanced_accuracy_score(
        y,
        oof_pred[name]
    )

    macro_f1 = f1_score(
        y,
        oof_pred[name],
        average="macro"
    )

    AUC = roc_auc_score(
        y,
        oof_prob[name]
    )

    point_results.append({
        "Model": name,
        "Balanced_Accuracy": BA,
        "Macro_F1": macro_f1,
        "AUC": AUC
    })


point_df = pd.DataFrame(point_results)

print("\n" + "=" * 75)
print("POINT ESTIMATES")
print("=" * 75)

print(
    point_df.to_string(
        index=False,
        formatters={
            "Balanced_Accuracy": "{:.6f}".format,
            "Macro_F1": "{:.6f}".format,
            "AUC": "{:.6f}".format
        }
    )
)


# ============================================================
# 9. BOOTSTRAP 95% CONFIDENCE INTERVALS
#
# Resample participant-level OOF predictions.
#
# Important:
# For BA and AUC, a bootstrap resample must contain
# both classes. We therefore retain only valid resamples
# containing at least one Student and one Lecturer.
#
# Generate exactly 5000 VALID bootstrap resamples.
# ============================================================

N_BOOT = 5000
BOOT_RANDOM_STATE = 42

rng = np.random.default_rng(
    BOOT_RANDOM_STATE
)


def bootstrap_ci(
    y_true,
    y_pred,
    y_prob,
    n_boot=5000
):

    ba_values = []
    f1_values = []
    auc_values = []

    n_obs = len(y_true)

    attempts = 0

    # Safety limit
    max_attempts = n_boot * 20

    while (
        len(auc_values) < n_boot
        and attempts < max_attempts
    ):

        attempts += 1

        idx = rng.integers(
            0,
            n_obs,
            size=n_obs
        )

        y_b = y_true[idx]

        # AUC/BA require both classes for meaningful
        # evaluation in this analysis
        if np.unique(y_b).size < 2:
            continue

        pred_b = y_pred[idx]
        prob_b = y_prob[idx]

        ba_values.append(
            balanced_accuracy_score(
                y_b,
                pred_b
            )
        )

        f1_values.append(
            f1_score(
                y_b,
                pred_b,
                average="macro"
            )
        )

        auc_values.append(
            roc_auc_score(
                y_b,
                prob_b
            )
        )


    if len(auc_values) < n_boot:
        raise RuntimeError(
            "Could not obtain the requested number "
            "of valid bootstrap samples."
        )

    ba_values = np.asarray(ba_values)
    f1_values = np.asarray(f1_values)
    auc_values = np.asarray(auc_values)

    # Percentile 95% CI
    ba_ci = np.percentile(
        ba_values,
        [2.5, 97.5]
    )

    f1_ci = np.percentile(
        f1_values,
        [2.5, 97.5]
    )

    auc_ci = np.percentile(
        auc_values,
        [2.5, 97.5]
    )

    return (
        ba_ci,
        f1_ci,
        auc_ci,
        attempts
    )


# ============================================================
# 10. BOOTSTRAP EACH MODEL
# ============================================================

final_results = []

print("\n" + "=" * 75)
print("BOOTSTRAP 95% CONFIDENCE INTERVALS")
print("=" * 75)


for name in models:

    BA = balanced_accuracy_score(
        y,
        oof_pred[name]
    )

    macro_f1 = f1_score(
        y,
        oof_pred[name],
        average="macro"
    )

    AUC = roc_auc_score(
        y,
        oof_prob[name]
    )

    ba_ci, f1_ci, auc_ci, attempts = bootstrap_ci(
        y,
        oof_pred[name],
        oof_prob[name],
        n_boot=N_BOOT
    )

    final_results.append({

        "Model": name,

        "Balanced Accuracy": BA,
        "BA CI lower": ba_ci[0],
        "BA CI upper": ba_ci[1],

        "Macro-F1": macro_f1,
        "Macro-F1 CI lower": f1_ci[0],
        "Macro-F1 CI upper": f1_ci[1],

        "AUC": AUC,
        "AUC CI lower": auc_ci[0],
        "AUC CI upper": auc_ci[1],

        "Valid bootstrap samples": N_BOOT,
        "Bootstrap attempts": attempts
    })


final_results_df = pd.DataFrame(
    final_results
)


print(
    final_results_df.to_string(
        index=False
    )
)


# ============================================================
# 11. FEATURE-SELECTION FREQUENCY
# ============================================================

selection_frequency = (
    selection_counts / n
) * 100


feature_selection_df = pd.DataFrame({

    "Feature": feature_names,

    "Selected_folds": selection_counts,

    "Total_folds": n,

    "Selection_frequency_percent":
        selection_frequency
})


feature_selection_df = (
    feature_selection_df
    .sort_values(
        "Selection_frequency_percent",
        ascending=False
    )
    .reset_index(drop=True)
)


print("\n" + "=" * 75)
print("FEATURE-SELECTION CONSISTENCY ACROSS LOOCV FOLDS")
print("=" * 75)

print(
    feature_selection_df.to_string(
        index=False,
        formatters={
            "Selection_frequency_percent":
                "{:.1f}".format
        }
    )
)


# ============================================================
# 12. SAVE OUT-OF-FOLD PREDICTIONS
# ============================================================

oof_df = pd.DataFrame({
    "Code": X_df.index,
    "True_group": [
        "Lecturers" if value == 1
        else "Students"
        for value in y
    ]
})

for name in models:

    safe_name = (
        name
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )

    oof_df[
        f"{safe_name}_pred"
    ] = oof_pred[name]

    oof_df[
        f"{safe_name}_prob_Lecturer"
    ] = oof_prob[name]


# ============================================================
# 13. SAVE EVERYTHING TO EXCEL
# ============================================================

output_excel = (
    OUTPUT /
    "RQ1_FINAL_LOOCV_N35_FULL_ANALYSIS.xlsx"
)

with pd.ExcelWriter(
    output_excel,
    engine="openpyxl"
) as writer:

    final_results_df.to_excel(
        writer,
        sheet_name="Performance",
        index=False
    )

    feature_selection_df.to_excel(
        writer,
        sheet_name="Feature_selection",
        index=False
    )

    oof_df.to_excel(
        writer,
        sheet_name="OOF_predictions",
        index=False
    )

    X_df.reset_index().to_excel(
        writer,
        sheet_name="Feature_matrix",
        index=False
    )


print("\n" + "=" * 75)
print("ANALYSIS SAVED")
print("=" * 75)

print(output_excel)