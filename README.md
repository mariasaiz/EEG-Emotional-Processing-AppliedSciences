# Supplementary Code — EEG Spectral Patterns during Emotional Processing

Updated code-only package accompanying the revised *Applied Sciences* manuscript. This repository must contain **source code only**, not raw EEG, personal data, participant-level derived features, or participant-level prediction tables. The source code may include study-specific coded participant IDs used in quality-control rules; these are not participant-level datasets.

## Included scripts and execution order

1. `01_EEG_preprocessing_v2.py` — import Bitbrain EEG and events; check channel quality; 1–100 Hz FIR filtering; average reference; Extended Infomax ICA; ICLabel; bad-channel interpolation; final 45 Hz low-pass filter; extract 6-s epochs; Welch PSD with 512-point FFT and 50% overlap; Simpson band-power integration (five bands, twelve channels). Produces `OUTPUT_FINAL_UNIFIED/`.
2. `02_RQ1_classification_v2.py` — RQ1 supervised classification of 25 spectral features, LOOCV, fold-specific RobustScaler and SelectKBest, RF/LR/KNN/MLP and participant-level bootstrap CIs.
3. `03_RQ1_log_sensitivity_v2.py` — RQ1 repeated with natural-log-transformed spectral power.
4. `04_RQ2_statistics_LMM_v2.py` — RQ2 distribution checks, Mann–Whitney U, Friedman, Holm adjustment, ML-fitted random-intercept mixed models, fixed-effect CIs and residual diagnostics. Checks Table 9 likelihood-ratio statistics within tolerance.
5. `05_RQ3_asymmetry_V3.py` — RQ3 signed and mean-absolute hemispheric asymmetry, six homologous channel pairs, top-three rankings and participant-band summaries for the final N=35 sample. **Uses published DASM units in µV².**
6. `06_RQ3_sensitivity_v2.py` — Appendix A participant- and electrode-pair-exclusion sensitivity scenarios. **Requires the output of script 05 V3**, whose `P_L`, `P_R` and `DASM` columns are in µV².
7. `07_RQ3_figure6_v2.py` — regenerate Figure 6 from script 05 V3's participant-band summary, including **all 35 participants**, a symmetric-logarithmic colour scale for signed DASM (µV²), and a common-centred linear scale for signed ln(RASM) (dimensionless).
8. `08_record_software_versions_v2.py` — write `software_versions.json` containing package versions in the environment in which it is executed.

## Local folder configuration (no personal paths stored)

Set the optional `EEG_PROJECT_DIR` environment variable to the **local, access-controlled research-data directory**. If unset, scripts 01–07 use the folder containing their `.py` file as their project directory. Do **not** upload the contents of your local research-data directory.

The expected directory layout is:

```text
<EEG_PROJECT_DIR>/
  INPUT/                         # participant folders with raw eeg_eeg.csv and stimulus.csv
  OUTPUT_PHASE_C1/
    BBDD_SOCIO_FINAL_N35.xlsx    # local group mapping; NOT for publication
  OUTPUT_FINAL_UNIFIED/          # produced by script 01
    BandPower_channel_level.csv
    BandPower_stimulus_band_mean.csv
  OUTPUT_RQ3_ASYMMETRY/          # produced by scripts 05–07
```

`05_RQ3_asymmetry_V3.py` writes `OUTPUT_RQ3_ASYMMETRY/RQ3_DASM_RASM_long.csv` and `RQ3_participant_band_summary.csv`. Script 06 reads the first file; script 07 reads the second. These results contain participant-level observations and **must remain private**.

### DASM conversion and RASM definition

Script 01 applies `RAW_SCALE = 1e-6` to the exported EEG amplitudes before MNE processing. **If the original export amplitudes are in µV**, MNE receives signals in V, Welch PSD has units V²/Hz and Simpson integration produces spectral power in V². The source file labels this field `working_units2`; the physical-unit provenance depends on verifying the original device-export units. That verification should be retained in laboratory documentation.

Script 05 multiplies left and right channel band powers by **10¹²** before calculating:

- `DASM = P_R − P_L` (reported in **µV²**);
- `RASM = P_R/P_L` (dimensionless);
- `ln(RASM) = ln(P_R/P_L)` (dimensionless).

For participant rankings in Table 10, absolute values are taken **before** averaging the five emotional stimuli and six homologous pairs. Appendix Table A1 reports the sample median of these participant-level mean absolute indices. Figure 6 and Appendix Figures A1/A2 represent **mean signed** indices and must not be confused with mean absolute magnitudes. Changing units by a positive constant does not affect RASM or participant rank order.

### EEG channel names verified from original Bitbrain metadata

The acquisition-channel names and ordering were cross-checked against the original Bitbrain SennsLab `info.json` metadata, specifically the EEG entry (`device_type: bth_diadem_ec`, `num_channels: 12`, `sampling_rate: 256`, and `signal_loc`). The EEG channel sequence is:

`AF7, Fp1, Fp2, AF8, F3, F4, P3, P4, P7, O1, O2, P8`.

The reference and ground entries are `REF(A1)` and `GND(Fpz)`. In particular, the posterior lateral EEG channels are identified in the source metadata as **P7** and **P8**, not PO7 and PO8. All supplied scripts therefore retain **P7/P8** as exported. Any manuscript occurrences of **PO7/PO8** should be corrected to **P7/P8** for consistency with this dataset. No electrode names, spatial montage, interpolation parameters, or numerical results were changed as a result of this metadata review.

According to the *SennsLab User Manual*, version 7.20 (March 2025), Section 4.5 (*Researchers Guide*), the metadata field `signal_loc` lists the EEG electrode positions; it does **not** establish the physical amplitude unit of the numeric EEG samples. Thus channel-name verification is independent from the outstanding raw-amplitude-unit provenance check.

### Numerical validation and limitations

Scripts include study-specific checks (N=35; exclusion of participant 32; expected bands, stimuli and channel pairs). Script 05 also checks its five first-place DASM values against Table 10 at the published precision. These are **dataset-specific reproducibility checks**, not independent external validation. Syntax checking alone does not prove that an entire pipeline reproduces the manuscript. The complete preprocessing pipeline must be run with the protected data and correct library versions to verify end-to-end reproduction.

The repository does **not** include standalone scripts for Appendix Figures A1/A2; those figures were generated locally from the same final band-power table. If the article is intended to claim complete figure-level reproduction from the GitHub package, add the data-driven A1/A2 figure scripts (with no local paths and without including any participant data).

## Privacy and data availability

Raw recordings, event timestamps, socio-demographic workbooks, channel-level and participant-level power tables, `RQ3_DASM_RASM_long.csv`, Excel workbooks, epoch files, prediction tables and other derived individual-level datasets **must not be committed**. Only the `.py` sources, the `README.md`, and suitable metadata/instructions should be published, subject to the approved data-sharing restrictions.

## Dependencies and versioning

Python, NumPy, pandas, SciPy, MNE-Python, mne-icalabel, scikit-learn, statsmodels, matplotlib and openpyxl are required for the corresponding analyses. Capture package versions on the original analysis workstation using script 08. Version tags/releases in GitHub are immutable snapshots: updating `main` does not change a previous release; create a new release and update the manuscript's link if the manuscript cites a specific tag.
