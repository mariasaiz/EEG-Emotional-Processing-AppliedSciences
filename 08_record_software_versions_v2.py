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

# 08: Capture versions in the research analysis environment (metadata only).

import sys, importlib.metadata as md, json
from pathlib import Path
packages=['numpy','pandas','scipy','mne','mne-icalabel','statsmodels','scikit-learn','matplotlib','seaborn','openpyxl','onnxruntime']
record={'Python':sys.version,'packages':{}}
for name in packages:
 try: record['packages'][name]=md.version(name)
 except md.PackageNotFoundError: record['packages'][name]='not installed'
Path('software_versions.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
