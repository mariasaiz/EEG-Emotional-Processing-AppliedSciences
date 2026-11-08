# Supplementary Code — EEG Spectral Patterns during Emotional Processing

This code-only package has been curated from `PROLEPSIS(4).ipynb` against the revised *Applied Sciences* manuscript. It contains no EEG recordings, identifiable information, participant-level derived features or model-prediction tables.

## Included scripts

1. `01_EEG_preprocessing.py`: Bitbrain EEG import, technical channel QC, 1–100 Hz FIR filtering, average referencing, Extended Infomax ICA and ICLabel (threshold .80), post-ICA channel interpolation, 45 Hz low-pass filtering, six-second emotional-stimulus epochs, Welch PSD (512-point FFT, Hann window, 50% overlap), Simpson integration, five bands, 12 channels.
2. `02_RQ1_classification.py`: main analysis of 25 participant-level spectral features, LOOCV, within-fold RobustScaler and SelectKBest(f_classif, k=8), Random Forest, Logistic Regression, KNN and MLP, bootstrap uncertainty intervals.
3. `03_RQ1_log_sensitivity.py`: the corresponding log-transformed sensitivity analysis (all four models), including feature-selection and out-of-fold prediction computations. Individual-level outputs are *not distributed*.
4. `04_RQ2_statistics_LMM.py`: Shapiro–Wilk, signed Mann–Whitney effect sizes, Friedman tests, ML full/reduced random-intercept mixed models, Holm adjustments, model-based fixed-effect 95% CIs and residual plots. It compares the calculated five LR statistics with Table 9 values.
5. `05_RQ3_asymmetry.py`: six homologous pairs and manuscript-defined DASM = right minus left power and RASM = right/left power; directional signed means and mean absolute indices for descriptive ranking.
6. `06_RQ3_sensitivity.py`: manuscript Appendix A scenarios excluding participants 24 and 27 or specific homologous pairs with signal loss; additional high-amplitude scenario; aggregate descriptive output.
7. `07_RQ3_figure6.py`: generation of signed DASM and ln(RASM) heat maps.
8. `08_record_software_versions.py`: generate `software_versions.json` in the *original* analysis environment. The manuscript reports MNE-Python 1.11.0; the versions of other packages should be obtained from the research environment, not inferred.

## Data requirements and execution

- Set `EEG_PROJECT_DIR` to a **local, access-controlled** research folder containing `INPUT/` with recordings and event files, and `OUTPUT_PHASE_C1/BBDD_SOCIO_FINAL_N35.xlsx` with the verified participant-group mapping.
- Execute scripts in numerical order. EEG preprocessing creates `OUTPUT_FINAL_UNIFIED/`; RQ1 and RQ2 use `BandPower_stimulus_band_mean.csv`; RQ3 uses `BandPower_channel_level.csv` and creates its output workbook.
- These scripts may write local CSV/Excel outputs containing participant-level derived measures. Such outputs **must remain in the controlled research environment and must not be added to the Supplementary Materials**. This archive includes Python source and instructions only.
- Dependencies include Python, NumPy, pandas, SciPy, MNE-Python, mne-icalabel, statsmodels, scikit-learn, matplotlib, openpyxl and the ICLabel execution backend. Version compatibility should be tested in the original analysis environment.

## Scope and verification

- The selected historical source cells used are 312, 346, 348, 351 and 353. The RQ2 script consolidates the methods and model comparisons from cells 329–344; the RQ3 sensitivity script follows the current manuscript's Appendix A specifications instead of retaining the earlier alternative RASM definition and the obsolete 32-participant workflow.
- All scripts pass Python syntax checks. **The archive has not been end-to-end executed against the protected raw recordings here**, so exact reproduction of all tables is not claimed. The RQ2 script checks original LR values to a tolerance of 0.01.
- The original raw-power preprocessing contains an explicit working-scale conversion; physical-unit provenance must be verified before re-labelling values in physical units.
- The supplied manuscript contains additional editorial inconsistencies (e.g. the former Figure 7 still appears in some paragraphs); these are outside the code archive and are not silently changed.

## Privacy

No raw or processed human EEG data, participant identifiers, individual-level feature matrices, or participant-level predictions are included in this code-only ZIP. Source-code scripts and output schema information do not override the ethical or informed-consent restrictions on access to recordings and derived individual measures.
