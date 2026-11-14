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

# 01: MNE, ICA/ICLabel, six-second epochs, Welch and Simpson band power.
# ============================================================

# ============================================================
#
# PIPELINE
# Raw EEG

# -> flag bad channels
# -> 1–100 Hz filter


# -> ICLabel
# -> exclude artefactual independent components

# -> interpolate bad channels

# -> low-pass 45 Hz

# -> six-second epochs
# -> Welch PSD


#


# N final = 35
# ============================================================


from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import mne

from mne.preprocessing import ICA
from mne_icalabel import label_components

from scipy.signal import welch
from scipy.integrate import simpson


# ============================================================

# ============================================================

ROOT = PROJECT_DATA / "INPUT"

OUTPUT = PREPROCESS_DIR

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)

EPOCHS_DIR = OUTPUT / "epochs"

EPOCHS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================

# ============================================================

SFREQ = 256.0

# Channel labels and acquisition order follow the original Bitbrain SennsLab
# info.json metadata (EEG signal_loc). In particular, the 9th and 12th EEG
# channels are labelled P7 and P8 (not PO7 and PO8) in the source export.
# Preserve these labels in the spatial montage and subsequent analyses.
EEG_CHANNELS = [
    "AF7",
    "Fp1",
    "Fp2",
    "AF8",
    "F3",
    "F4",
    "P3",
    "P4",
    "P7",
    "O1",
    "O2",
    "P8"
]

N_EEG = len(EEG_CHANNELS)


# ============================================================
# 3. WORKING SCALE
#

#
# IMPORTANT:


# ============================================================

RAW_SCALE = 1e-6


# ============================================================
# 4. BAD CHANNELS AND EXCLUSIONS
# ============================================================

BAD_CHANNELS = {
    24: ["F3", "P3"],
    27: ["AF7"],
    28: ["O2"],
    33: ["F4"]
}

NOT_IN_STUDY = [
    22,
    36
]

EXCLUDED_QC = [
    32
]


# ============================================================
# 5. ICA / ICLABEL
# ============================================================

ICA_RANDOM_STATE = 97

ICLABEL_THRESHOLD = 0.80

ARTIFACT_LABELS = {
    "eye blink",
    "muscle artifact",
    "heart beat",
    "line noise",
    "channel noise"
}


# ============================================================

# ============================================================

EVENT_MAP = {
    1: "Surprise",
    3: "Anger",
    4: "Happiness",
    5: "Sadness",
    6: "Fear"
}

EXPECTED_CODES = [
    1,
    3,
    4,
    5,
    6
]


# ============================================================
# MNE requiere:

# ============================================================

MNE_EVENT_ID = {
    name: code
    for code, name in EVENT_MAP.items()
}


EPOCH_DURATION = 6.0


# ============================================================

# ============================================================

BANDS = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "LowGamma": (30.0, 45.0)
}


# ============================================================
# 8. WELCH
# ============================================================

NPERSEG = 512
NOVERLAP = 256
NFFT = 512


# ============================================================
# 9. MONTAGE
# ============================================================

montage = mne.channels.make_standard_montage(
    "standard_1020"
)


# ============================================================

# ============================================================

MIN_SUBJECT = 1
MAX_SUBJECT = 38


available_subjects = sorted(
    int(folder.name)
    for folder in ROOT.iterdir()
    if folder.is_dir()
    and folder.name.isdigit()
    and MIN_SUBJECT <= int(folder.name) <= MAX_SUBJECT
)


print("=" * 100)
print("AVAILABLE PARTICIPANT FOLDERS BETWEEN 1 AND 38")
print("=" * 100)

print(
    available_subjects
)

print(
    "\nNumber of available folders:",
    len(available_subjects)
)


# ============================================================

# ============================================================

study_subjects = [
    sid
    for sid in available_subjects
    if sid not in NOT_IN_STUDY
]


print("\n" + "=" * 100)
print("ORIGINAL STUDY COHORT")
print("=" * 100)

print(
    study_subjects
)

print(
    "\nOriginal cohort size:",
    len(study_subjects)
)


# ============================================================

# ============================================================

subjects_to_process = [
    sid
    for sid in study_subjects
    if sid not in EXCLUDED_QC
]


print(
    "\nQC exclusion:"
)

print(
    EXCLUDED_QC
)

print(
    "\nParticipants to process:"
)

print(
    subjects_to_process
)

print(
    "\nN final:",
    len(subjects_to_process)
)


# ============================================================

# ============================================================

EXPECTED_ORIGINAL = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 37, 38
]


EXPECTED_FINAL = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    23, 24, 25, 26, 27, 28, 29, 30, 31,
    33, 34, 35, 37, 38
]


if study_subjects != EXPECTED_ORIGINAL:

    raise ValueError(
        "The original cohort does not match the expected "
        "36 participants.\n"
        f"Observed: {study_subjects}"
    )


if subjects_to_process != EXPECTED_FINAL:

    raise ValueError(
        "The final cohort does not match the expected "
        "35 participants.\n"
        f"Observed: {subjects_to_process}"
    )


if len(study_subjects) != 36:

    raise ValueError(
        f"Incorrect original sample size: "
        f"{len(study_subjects)}"
    )


if len(subjects_to_process) != 35:

    raise ValueError(
        f"Incorrect final sample size: "
        f"{len(subjects_to_process)}"
    )


print("\n" + "=" * 100)
print("COHORT SUCCESSFULLY VERIFIED")
print("N ORIGINAL = 36")
print("N FINAL = 35")
print("=" * 100)


# ============================================================
# 11. RESULT CONTAINERS
# ============================================================

ica_records = []
sync_records = []
epoch_records = []
band_records = []
feature_records = []
errors = []


# ============================================================

# ============================================================

def compute_bandpower(signal, sfreq):

    freqs, psd = welch(
        signal,
        fs=sfreq,
        window="hann",
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
        nfft=NFFT,
        detrend="constant",
        scaling="density"
    )

    powers = {}

    for band_name, (fmin, fmax) in BANDS.items():

        if band_name == "LowGamma":

            mask = (
                (freqs >= fmin)
                &
                (freqs <= fmax)
            )

        else:

            mask = (
                (freqs >= fmin)
                &
                (freqs < fmax)
            )

        band_freqs = freqs[
            mask
        ]

        band_psd = psd[
            mask
        ]

        power = simpson(
            band_psd,
            x=band_freqs
        )

        powers[
            band_name
        ] = float(
            power
        )

    return powers


# ============================================================

# ============================================================

for sid in subjects_to_process:

    print("\n")
    print("=" * 100)
    print(f"PARTICIPANT {sid}")
    print("=" * 100)

    try:

        subject_dir = ROOT / str(
            sid
        )

        eeg_file = (
            subject_dir
            / "eeg_eeg.csv"
        )

        stim_file = (
            subject_dir
            / "stimulus.csv"
        )


        # ====================================================

        # ====================================================

        if not eeg_file.exists():

            raise FileNotFoundError(
                eeg_file
            )

        if not stim_file.exists():

            raise FileNotFoundError(
                stim_file
            )


        # ====================================================

        # ====================================================

        eeg_df = pd.read_csv(
            eeg_file,
            header=None
        )


        if eeg_df.shape[1] < 15:

            raise ValueError(
                f"EEG participant {sid}: "
                f"at least 15 columns were expected "
                f"but {eeg_df.shape[1]} were found."
            )


        eeg_data_raw = (
            eeg_df
            .iloc[:, :12]
            .to_numpy(
                dtype=float
            )
        )


        sample_timestamps = (
            eeg_df
            .iloc[:, 14]
            .to_numpy(
                dtype=float
            )
        )


        # ====================================================
        # 13.3. ESCALA
        # ====================================================

        eeg_data = (
            eeg_data_raw
            *
            RAW_SCALE
        )


        # MNE:


        eeg_data = (
            eeg_data.T
        )


        # ====================================================
        # 13.4. CREATE RAW OBJECT
        # ====================================================

        info = mne.create_info(
            ch_names=EEG_CHANNELS,
            sfreq=SFREQ,
            ch_types="eeg"
        )


        raw = mne.io.RawArray(
            eeg_data,
            info,
            verbose=False
        )


        raw.set_montage(
            montage,
            on_missing="raise",
            verbose=False
        )


        # ====================================================
        # 13.5. BAD CHANNELS
        # ====================================================

        bads = BAD_CHANNELS.get(
            sid,
            []
        )


        raw.info[
            "bads"
        ] = bads.copy()


        print(
            "Bad channels:",
            bads
            if bads
            else "None"
        )


        good_channels = [
            ch
            for ch in EEG_CHANNELS
            if ch not in bads
        ]


        print(
            "Good channels:",
            len(
                good_channels
            )
        )


        # ====================================================
        # 13.6. 1–100 Hz FILTER
        # ====================================================

        raw_filt = (
            raw.copy()
        )


        raw_filt.filter(
            l_freq=1.0,
            h_freq=100.0,
            method="fir",
            phase="zero",
            fir_design="firwin",
            verbose=False
        )


        # ====================================================


        # ====================================================

        raw_filt.set_eeg_reference(
            ref_channels=good_channels,
            projection=False,
            verbose=False
        )


        # ====================================================

        # ====================================================

        raw_ica = (
            raw_filt
            .copy()
            .pick(
                good_channels
            )
        )


        # ====================================================

        # ====================================================

        rank_dict = mne.compute_rank(
            raw_ica,
            rank=None,
            verbose=False
        )


        rank_eeg = int(
            rank_dict[
                "eeg"
            ]
        )


        print(
            "EEG rank for ICA:",
            rank_eeg
        )


        if rank_eeg < 2:

            raise ValueError(
                f"Participant {sid}: "
                f"insufficient EEG rank "
                f"({rank_eeg})."
            )


        # ====================================================
        # 13.10. ICA EXTENDED INFOMAX
        # ====================================================

        ica = ICA(
            n_components=rank_eeg,
            method="infomax",
            fit_params={
                "extended": True
            },
            random_state=ICA_RANDOM_STATE,
            max_iter="auto"
        )


        with warnings.catch_warnings():

            warnings.simplefilter(
                "ignore"
            )

            ica.fit(
                raw_ica,
                verbose=False
            )


        print(
            "ICA components:",
            ica.n_components_
        )


        # ====================================================
        # 13.11. ICLABEL
        # ====================================================

        with warnings.catch_warnings():

            warnings.simplefilter(
                "ignore"
            )

            iclabel = label_components(
                raw_ica,
                ica,
                method="iclabel"
            )


        labels = (
            iclabel[
                "labels"
            ]
        )


        probabilities = np.asarray(
            iclabel[
                "y_pred_proba"
            ]
        )


        if len(
            labels
        ) != ica.n_components_:

            raise ValueError(
                f"ICLabel returned "
                f"{len(labels)} labels "
                f"for "
                f"{ica.n_components_} components."
            )


        # ====================================================
        # 13.12. ARTEFACTUAL COMPONENTS
        # ====================================================

        exclude = []


        for ic_idx, (
            label,
            probability
        ) in enumerate(
            zip(
                labels,
                probabilities
            )
        ):

            probability = float(
                probability
            )


            is_artifact = (
                label
                in ARTIFACT_LABELS
                and
                probability
                >= ICLABEL_THRESHOLD
            )


            if is_artifact:

                exclude.append(
                    ic_idx
                )


            ica_records.append(
                {
                    "Code": sid,
                    "IC": ic_idx,
                    "Label": label,
                    "Probability": probability,
                    "Excluded": bool(
                        is_artifact
                    ),
                    "Rank": rank_eeg,
                    "N_good_channels": len(
                        good_channels
                    ),
                    "Bad_channels": ",".join(
                        bads
                    )
                }
            )


        ica.exclude = exclude


        print(
            "ICs excluidos:",
            exclude
        )


        # ====================================================

        # ====================================================

        raw_ica_clean = (
            raw_ica.copy()
        )


        ica.apply(
            raw_ica_clean,
            exclude=exclude,
            verbose=False
        )


        # Full recording

        cleaned = (
            raw_filt.copy()
        )


        # Restore the full recording


        for ch_name in good_channels:

            idx_full = (
                cleaned
                .ch_names
                .index(
                    ch_name
                )
            )

            idx_good = (
                raw_ica_clean
                .ch_names
                .index(
                    ch_name
                )
            )


            cleaned._data[
                idx_full,
                :
            ] = (
                raw_ica_clean
                ._data[
                    idx_good,
                    :
                ]
            )


        # ====================================================

        # ====================================================

        if len(
            bads
        ) > 0:

            print(
                "Interpolating after ICA:",
                bads
            )


            cleaned.interpolate_bads(
                reset_bads=True,
                mode="accurate",
                verbose=False
            )


        # ====================================================

        # ====================================================

        cleaned.set_eeg_reference(
            ref_channels="average",
            projection=False,
            verbose=False
        )


        # ====================================================
        # 13.16. LOW-PASS FINAL 45 Hz
        # ====================================================

        cleaned.filter(
            l_freq=None,
            h_freq=45.0,
            method="fir",
            phase="zero",
            fir_design="firwin",
            verbose=False
        )


        # ====================================================

        # ====================================================

        stim = pd.read_csv(
            stim_file,
            header=None
        )


        if stim.shape[1] < 3:

            raise ValueError(
                f"stimulus.csv participant "
                f"{sid}: "
                f"expected 3 columnas."
            )


        stim.columns = [
            "timestamp",
            "event_type",
            "event_code"
        ]


        targets = (
            stim.loc[
                (
                    stim[
                        "event_type"
                    ] == 1
                )
                &
                (
                    stim[
                        "event_code"
                    ].isin(
                        EXPECTED_CODES
                    )
                )
            ]
            .copy()
        )


        targets = (
            targets
            .sort_values(
                "timestamp"
            )
            .reset_index(
                drop=True
            )
        )


        found_codes = (
            targets[
                "event_code"
            ]
            .astype(
                int
            )
            .tolist()
        )


        if found_codes != EXPECTED_CODES:

            raise ValueError(
                f"Participant {sid}: "
                f"events found "
                f"{found_codes}; "
                f"expected "
                f"{EXPECTED_CODES}."
            )


        # ====================================================

        # ====================================================

        events = []


        for _, row in (
            targets
            .iterrows()
        ):

            event_timestamp = float(
                row[
                    "timestamp"
                ]
            )


            event_code = int(
                row[
                    "event_code"
                ]
            )


            nearest_idx = int(
                np.argmin(
                    np.abs(
                        sample_timestamps
                        -
                        event_timestamp
                    )
                )
            )


            nearest_timestamp = float(
                sample_timestamps[
                    nearest_idx
                ]
            )


            # Timestamps in microseconds
            # diferencia /1000 = ms

            sync_error_ms = (
                nearest_timestamp
                -
                event_timestamp
            ) / 1000.0


            sync_records.append(
                {
                    "Code": sid,
                    "Event_code": event_code,
                    "Stimulus": EVENT_MAP[
                        event_code
                    ],
                    "Event_timestamp": event_timestamp,
                    "Nearest_sample": nearest_idx,
                    "Sample_timestamp": nearest_timestamp,
                    "Sync_error_ms": sync_error_ms
                }
            )


            events.append(
                [
                    nearest_idx,
                    0,
                    event_code
                ]
            )


        events = np.asarray(
            events,
            dtype=int
        )


        # ====================================================

        # ====================================================

        participant_sync = [
            r[
                "Sync_error_ms"
            ]
            for r in sync_records
            if r[
                "Code"
            ] == sid
        ]


        max_sync = float(
            np.max(
                np.abs(
                    participant_sync
                )
            )
        )


        print(
            f"Max |sync error| = "
            f"{max_sync:.6f} ms"
        )


        half_sample_ms = (
            1000.0
            /
            SFREQ
            /
            2.0
        )


        if max_sync > (
            half_sample_ms
            +
            1e-6
        ):

            raise ValueError(
                f"Participant {sid}: "
                f"timestamp alignment error "
                f"{max_sync:.6f} ms > "
                f"half a sample "
                f"({half_sample_ms:.6f} ms)."
            )


        # ====================================================
        # 13.20. SIX-SECOND EPOCHS
        # ====================================================

        tmax = (
            EPOCH_DURATION
            -
            1.0 / SFREQ
        )


        epochs = mne.Epochs(
            cleaned,
            events,
            event_id=MNE_EVENT_ID,
            tmin=0.0,
            tmax=tmax,
            baseline=None,
            preload=True,
            reject_by_annotation=False,
            verbose=False
        )


        if len(
            epochs
        ) != 5:

            raise ValueError(
                f"Participant {sid}: "
                f"five epochs were expected "
                f"but the number obtained was "
                f"{len(epochs)}."
            )


        n_samples_epoch = (
            epochs
            .get_data(
                copy=False
            )
            .shape[
                -1
            ]
        )


        expected_samples = int(
            SFREQ
            *
            EPOCH_DURATION
        )


        if (
            n_samples_epoch
            !=
            expected_samples
        ):

            raise ValueError(
                f"Participant {sid}: "
                f"expected "
                f"{expected_samples} samples "
                f"per epoch, but the number obtained was "
                f"{n_samples_epoch}."
            )


        # ====================================================

        # ====================================================

        epochs_file = (
            EPOCHS_DIR
            /
            f"sub-{sid:02d}_epo.fif"
        )


        epochs.save(
            epochs_file,
            overwrite=True,
            verbose=False
        )


        # ====================================================

        # ====================================================

        epoch_data = (
            epochs
            .get_data(
                copy=True
            )
        )


        # ====================================================
        # 13.23. BAND POWER
        # ====================================================

        participant_features = {
            "Code": sid
        }


        for epoch_idx, event_code in enumerate(
            EXPECTED_CODES
        ):

            stimulus_name = (
                EVENT_MAP[
                    event_code
                ]
            )


            channel_band_values = {
                band: []
                for band in BANDS
            }


            for ch_idx, ch_name in enumerate(
                EEG_CHANNELS
            ):

                signal = (
                    epoch_data[
                        epoch_idx,
                        ch_idx,
                        :
                    ]
                )


                powers = compute_bandpower(
                    signal,
                    SFREQ
                )


                for (
                    band_name,
                    power
                ) in powers.items():

                    channel_band_values[
                        band_name
                    ].append(
                        power
                    )


                    band_records.append(
                        {
                            "Code": sid,
                            "Stimulus": stimulus_name,
                            "Event_code": event_code,
                            "Channel": ch_name,
                            "Band": band_name,
                            "Power": power,
                            "Power_unit": "working_units2"
                        }
                    )


            # =================================================

            # =================================================

            for band_name in BANDS:

                mean_power = float(
                    np.mean(
                        channel_band_values[
                            band_name
                        ]
                    )
                )


                feature_name = (
                    f"{stimulus_name}_"
                    f"{band_name}"
                )


                participant_features[
                    feature_name
                ] = mean_power


                epoch_records.append(
                    {
                        "Code": sid,
                        "Stimulus": stimulus_name,
                        "Event_code": event_code,
                        "Band": band_name,
                        "Mean_power_12ch": mean_power,
                        "Power_unit": "working_units2"
                    }
                )


        feature_records.append(
            participant_features
        )


        print(
            f"Participant {sid}: OK"
        )


    except Exception as exc:

        errors.append(
            {
                "Code": sid,
                "Error": repr(
                    exc
                )
            }
        )


        print(
            f"ERROR for participant {sid}:",
            repr(
                exc
            )
        )


# ============================================================
# 14. CREATE DATAFRAMES
# ============================================================

ica_df = pd.DataFrame(
    ica_records
)

sync_df = pd.DataFrame(
    sync_records
)

epoch_df = pd.DataFrame(
    epoch_records
)

band_df = pd.DataFrame(
    band_records
)

features_df = pd.DataFrame(
    feature_records
)

errors_df = pd.DataFrame(
    errors
)


# ============================================================
# 15. ORDENAR FEATURES
# ============================================================

expected_feature_cols = []


for event_code in EXPECTED_CODES:

    stimulus = EVENT_MAP[
        event_code
    ]

    for band in BANDS:

        expected_feature_cols.append(
            f"{stimulus}_{band}"
        )


if not features_df.empty:

    features_df = features_df[
        ["Code"]
        +
        expected_feature_cols
    ]


    features_df = (
        features_df
        .sort_values(
            "Code"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================

# ============================================================

print("\n")
print("=" * 100)
print("FINAL VALIDATION")
print("=" * 100)


print(
    "\nParticipantes procesados:",
    len(
        features_df
    )
)


print(
    "Errors:",
    len(
        errors_df
    )
)


if len(
    errors_df
) > 0:

    print(
        "\nERRORES:"
    )

    print(
        errors_df
        .to_string(
            index=False
        )
    )


# ============================================================
# 16.1. CONTROLES ESTRICTOS
# ============================================================

if len(
    errors_df
) == 0:

    assert len(
        features_df
    ) == 35


    assert (
        features_df[
            "Code"
        ]
        .nunique()
        ==
        35
    )


    assert (
        features_df[
            "Code"
        ]
        .tolist()
        ==
        EXPECTED_FINAL
    )


    assert 32 not in (
        features_df[
            "Code"
        ]
        .values
    )


    assert 22 not in (
        features_df[
            "Code"
        ]
        .values
    )


    assert 36 not in (
        features_df[
            "Code"
        ]
        .values
    )


    assert len(
        expected_feature_cols
    ) == 25


    assert not (
        features_df[
            expected_feature_cols
        ]
        .isna()
        .any()
        .any()
    )


    assert np.isfinite(
        features_df[
            expected_feature_cols
        ]
        .to_numpy()
    ).all()


    # 35 × 5 eventos
    assert len(
        sync_df
    ) == 175



    assert len(
        band_df
    ) == 10500



    assert len(
        epoch_df
    ) == 875


# ============================================================

# ============================================================

if len(
    sync_df
) > 0:

    print("\n")
    print("=" * 100)
    print("SYNCHRONISATION")
    print("=" * 100)


    print(
        sync_df[
            "Sync_error_ms"
        ].describe()
    )


    print(
        "\nGlobal max |sync error|:"
    )


    print(
        sync_df[
            "Sync_error_ms"
        ]
        .abs()
        .max()
    )


# ============================================================
# 18. ICA / ICLABEL
# ============================================================

if len(
    ica_df
) > 0:

    print("\n")
    print("=" * 100)
    print("ICA / ICLABEL")
    print("=" * 100)


    ica_summary = (
        ica_df
        .groupby(
            "Code"
        )
        .agg(
            N_IC=(
                "IC",
                "count"
            ),
            N_excluded=(
                "Excluded",
                "sum"
            )
        )
    )


    print(
        ica_summary
        .to_string()
    )


# ============================================================
# 19. FEATURES
# ============================================================

if len(
    features_df
) > 0:

    print("\n")
    print("=" * 100)
    print("FEATURES")
    print("=" * 100)


    print(
        "Shape:",
        features_df.shape
    )


    print(
        "\nCodes:"
    )


    print(
        features_df[
            "Code"
        ].tolist()
    )


    print(
        "\nN features EEG:",
        len(
            expected_feature_cols
        )
    )


# ============================================================

# ============================================================

ica_file = (
    OUTPUT
    /
    "ICA_ICLabel_QC.csv"
)

sync_file = (
    OUTPUT
    /
    "Synchronization_QC.csv"
)

band_file = (
    OUTPUT
    /
    "BandPower_channel_level.csv"
)

epoch_file = (
    OUTPUT
    /
    "BandPower_stimulus_band_mean.csv"
)

features_file = (
    OUTPUT
    /
    "BandPower_25features_FINAL.csv"
)

errors_file = (
    OUTPUT
    /
    "Processing_errors.csv"
)


ica_df.to_csv(
    ica_file,
    index=False
)

sync_df.to_csv(
    sync_file,
    index=False
)

band_df.to_csv(
    band_file,
    index=False
)

epoch_df.to_csv(
    epoch_file,
    index=False
)

features_df.to_csv(
    features_file,
    index=False
)

errors_df.to_csv(
    errors_file,
    index=False
)


# ============================================================

# ============================================================

excel_file = (
    OUTPUT
    /
    "EEG_FINAL_UNIFIED_N35.xlsx"
)


with pd.ExcelWriter(
    excel_file,
    engine="openpyxl"
) as writer:

    features_df.to_excel(
        writer,
        sheet_name="25_features",
        index=False
    )


    epoch_df.to_excel(
        writer,
        sheet_name="stimulus_band",
        index=False
    )


    band_df.to_excel(
        writer,
        sheet_name="channel_band",
        index=False
    )


    sync_df.to_excel(
        writer,
        sheet_name="synchronization",
        index=False
    )


    ica_df.to_excel(
        writer,
        sheet_name="ICA_ICLabel",
        index=False
    )


    errors_df.to_excel(
        writer,
        sheet_name="errors",
        index=False
    )


# ============================================================

# ============================================================

print("\n")
print("=" * 100)
print("GENERATED FILES")
print("=" * 100)


print(
    "\n25 features:"
)

print(
    features_file
)


print(
    "\nComplete Excel workbook:"
)

print(
    excel_file
)


print(
    "\nQC ICA:"
)

print(
    ica_file
)


print(
    "\nSynchronisation QC:"
)

print(
    sync_file
)


print(
    "\nEpochs:"
)

print(
    EPOCHS_DIR
)


if len(
    errors_df
) == 0:

    print("\n")
    print("=" * 100)
    print(
        "UNIFIED PREPROCESSING COMPLETED"
    )
    print(
        "N FINAL = 35"
    )
    print("=" * 100)

else:

    print("\n")
    print("=" * 100)
    print(
        "PREPROCESSING FINISHED WITH ERRORS."
    )
    print(
        "DO NOT PROCEED TO THE ANALYSES."
    )
    print("=" * 100)