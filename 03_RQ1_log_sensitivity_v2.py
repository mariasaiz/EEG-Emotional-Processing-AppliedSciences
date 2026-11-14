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

# 03: Repeat RQ1 with natural-log-transformed power, as reported in Results.
# ============================================================
# RQ1 - SENSITIVITY ANALYSIS USING LOG-TRANSFORMED POWER
# Final sample N = 35
#
# Sensitivity analysis requested by reviewer:
# natural-log transformation of the 25 spectral-power features
#
# Pipeline:
# ln(Power)
# -> LOOCV
# -> RobustScaler fitted within training fold
# -> SelectKBest(f_classif, k=8) fitted within training fold
# -> RF / Logistic Regression / KNN / BPNN-MLP
# -> Balanced Accuracy / Macro-F1 / AUC
# -> 95% percentile bootstrap CI (5000 valid resamples)
# -> feature-selection consistency
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

print("=" * 80)
print("INPUT DATA")
print("=" * 80)

print("EEG shape:", eeg.shape)
print("Socio shape:", socio.shape)


# ============================================================
# 3. FINAL COHORT VALIDATION
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
assert len(eeg) == 875

print("Final cohort validated: N = 35")


# ============================================================
# 4. DEFINE FEATURE ORDER
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

feature_names = [
    f"{stimulus}_{band}"
    for stimulus in stimulus_order
    for band in band_order
]


# ============================================================
# 5. CHECK POSITIVE POWER VALUES
# ============================================================

min_power = eeg["Mean_power_12ch"].min()

print("\nMinimum spectral-power value:", min_power)

if min_power <= 0:
    raise ValueError(
        "Log transformation cannot be applied because "
        "zero or negative power values are present."
    )


# ============================================================
# 6. NATURAL-LOG TRANSFORMATION
# ============================================================

eeg["Log_power"] = np.log(
    eeg["Mean_power_12ch"]
)

print("\nLog-power range:")
print(
    eeg["Log_power"].min(),
    "to",
    eeg["Log_power"].max()
)


# ============================================================
# 7. BUILD 25-FEATURE MATRIX
# ============================================================

eeg["Feature"] = (
    eeg["Stimulus"].astype(str)
    + "_"
    + eeg["Band"].astype(str)
)

X_df = eeg.pivot(
    index="Code",
    columns="Feature",
    values="Log_power"
)

X_df = X_df[feature_names]

assert X_df.shape == (35, 25)
assert X_df.isna().sum().sum() == 0
assert np.isfinite(X_df.to_numpy()).all()

X = X_df.to_numpy(dtype=float)

print("\nLog-transformed feature matrix:", X_df.shape)


# ============================================================
# 8. TARGET
#
# 0 = Students
# 1 = Lecturers
# ============================================================

demo = (
    socio[
        ["Code", "Group_label"]
    ]
    .set_index("Code")
    .loc[X_df.index]
)

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

assert np.sum(y == 0) == 10
assert np.sum(y == 1) == 25

print("\nStudents:", np.sum(y == 0))
print("Lecturers:", np.sum(y == 1))


# ============================================================
# 9. CLASSIFIERS
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
        solver="liblinear",
        max_iter=5000,
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
# 10. LOOCV
# ============================================================

loo = LeaveOneOut()
n = len(y)

oof_pred = {
    name: np.zeros(n, dtype=int)
    for name in models
}

oof_prob = {
    name: np.zeros(n, dtype=float)
    for name in models
}

selection_counts = np.zeros(
    len(feature_names),
    dtype=int
)

print("\n" + "=" * 80)
print("LOOCV - LOG-TRANSFORMED POWER")
print("=" * 80)


for fold, (train_idx, test_idx) in enumerate(
    loo.split(X),
    start=1
):

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]

    # --------------------------------------------------------
    # RobustScaler fitted only on training data
    # --------------------------------------------------------

    scaler = RobustScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # --------------------------------------------------------
    # Feature selection fitted only on training data
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

    selection_counts += (
        selector
        .get_support()
        .astype(int)
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    for name, base_model in models.items():

        model = clone(base_model)

        model.fit(
            X_train_selected,
            y_train
        )

        pred = model.predict(
            X_test_selected
        )[0]

        prob = model.predict_proba(
            X_test_selected
        )[0, 1]

        oof_pred[name][test_idx[0]] = pred
        oof_prob[name][test_idx[0]] = prob


print("LOOCV completed.")


# ============================================================
# 11. POINT ESTIMATES
# ============================================================

point_results = []

for name in models:

    ba = balanced_accuracy_score(
        y,
        oof_pred[name]
    )

    macro_f1 = f1_score(
        y,
        oof_pred[name],
        average="macro"
    )

    auc = roc_auc_score(
        y,
        oof_prob[name]
    )

    point_results.append({
        "Model": name,
        "Balanced Accuracy": ba,
        "Macro-F1": macro_f1,
        "AUC": auc
    })


point_df = pd.DataFrame(
    point_results
)

print("\n" + "=" * 80)
print("POINT ESTIMATES - LOG POWER")
print("=" * 80)

print(
    point_df.to_string(
        index=False,
        formatters={
            "Balanced Accuracy": "{:.6f}".format,
            "Macro-F1": "{:.6f}".format,
            "AUC": "{:.6f}".format
        }
    )
)


# ============================================================
# 12. BOOTSTRAP FUNCTION
#
# 5000 VALID participant-level resamples
# ============================================================

N_BOOT = 5000


def bootstrap_ci(
    y_true,
    y_pred,
    y_prob,
    n_boot=5000,
    random_state=42
):

    rng = np.random.default_rng(
        random_state
    )

    ba_values = []
    f1_values = []
    auc_values = []

    n_obs = len(y_true)

    attempts = 0
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

        # AUC undefined for a single-class sample
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
            "Unable to obtain 5000 valid bootstrap resamples."
        )

    return {
        "BA_lower": np.percentile(
            ba_values,
            2.5
        ),
        "BA_upper": np.percentile(
            ba_values,
            97.5
        ),
        "F1_lower": np.percentile(
            f1_values,
            2.5
        ),
        "F1_upper": np.percentile(
            f1_values,
            97.5
        ),
        "AUC_lower": np.percentile(
            auc_values,
            2.5
        ),
        "AUC_upper": np.percentile(
            auc_values,
            97.5
        ),
        "Attempts": attempts
    }


# ============================================================
# 13. BOOTSTRAP RESULTS
# ============================================================

final_results = []

print("\n" + "=" * 80)
print("BOOTSTRAP 95% CI - LOG POWER")
print("=" * 80)


for model_index, name in enumerate(models):

    ba = balanced_accuracy_score(
        y,
        oof_pred[name]
    )

    macro_f1 = f1_score(
        y,
        oof_pred[name],
        average="macro"
    )

    auc = roc_auc_score(
        y,
        oof_prob[name]
    )

    # Use separate but reproducible seed per model
    ci = bootstrap_ci(
        y,
        oof_pred[name],
        oof_prob[name],
        n_boot=N_BOOT,
        random_state=42 + model_index
    )

    final_results.append({

        "Model": name,

        "Balanced Accuracy": ba,
        "BA CI lower": ci["BA_lower"],
        "BA CI upper": ci["BA_upper"],

        "Macro-F1": macro_f1,
        "Macro-F1 CI lower": ci["F1_lower"],
        "Macro-F1 CI upper": ci["F1_upper"],

        "AUC": auc,
        "AUC CI lower": ci["AUC_lower"],
        "AUC CI upper": ci["AUC_upper"],

        "Valid bootstrap samples": N_BOOT,
        "Bootstrap attempts": ci["Attempts"]
    })


results_df = pd.DataFrame(
    final_results
)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 14. FEATURE-SELECTION CONSISTENCY
# ============================================================

feature_selection_df = pd.DataFrame({

    "Feature": feature_names,

    "Selected_folds": selection_counts,

    "Total_folds": n,

    "Selection_frequency_percent":
        selection_counts / n * 100
})


feature_selection_df = (
    feature_selection_df
    .sort_values(
        "Selection_frequency_percent",
        ascending=False
    )
    .reset_index(drop=True)
)

print("\n" + "=" * 80)
print("FEATURE-SELECTION CONSISTENCY - LOG POWER")
print("=" * 80)

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
# 15. OUT-OF-FOLD PREDICTIONS
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
# 16. SAVE
# ============================================================

output_file = (
    OUTPUT /
    "RQ1_SENSITIVITY_LOG_POWER_N35.xlsx"
)

with pd.ExcelWriter(
    output_file,
    engine="openpyxl"
) as writer:

    results_df.to_excel(
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
        sheet_name="Log_feature_matrix",
        index=False
    )


print("\n" + "=" * 80)
print("SENSITIVITY ANALYSIS SAVED")
print("=" * 80)

print(output_file)