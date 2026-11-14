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

# 06: Appendix A reference and exclusion scenarios; produces aggregate-only summary.
# 06: RQ3 sensitivity analyses reported in Appendix A (no individual-level data included in this archive).
import numpy as np
import pandas as pd

INPUT=ASYMMETRY_DIR/'RQ3_DASM_RASM_long.csv'
OUTPUT=ASYMMETRY_DIR/'sensitivity'
OUTPUT.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(INPUT)
assert df.Code.nunique()==35
assert {'Code','Band','Stimulus','Left_channel','Right_channel','P_L','P_R'}.issubset(df.columns)
assert df.P_L.gt(0).all() and df.P_R.gt(0).all()
assert not df.duplicated(['Code','Band','Stimulus','Left_channel','Right_channel']).any()

def derive(df):
 d=df.copy()
 d['DASM']=d.P_R-d.P_L
 d['ln_RASM']=np.log(d.P_R/d.P_L)
 d['abs_DASM']=d.DASM.abs()
 d['abs_ln_RASM']=d.ln_RASM.abs()
 return d

# Exclusion of signal-loss pairs: exactly as described in Appendix A.
BAD_PAIRS={24:{frozenset(('F3','F4')),frozenset(('P3','P4'))},
           27:{frozenset(('AF7','AF8'))},28:{frozenset(('O1','O2'))},
           33:{frozenset(('F3','F4'))}}
ADDITIONAL_HIGH_AMPLITUDE={26:{frozenset(('O1','O2'))},34:{frozenset(('P3','P4'))}}

def remove_pairs(data,bad):
 keep=[frozenset((r.Left_channel,r.Right_channel)) not in bad.get(int(r.Code),set()) for r in data.itertuples(index=False)]
 return data.loc[keep].copy()

def summary(data,scenario):
 d=derive(data)
 individual=d.groupby(['Code','Band'],as_index=False).agg(
   Mean_Abs_DASM=('abs_DASM','mean'),Mean_Abs_ln_RASM=('abs_ln_RASM','mean'),
   Mean_Signed_DASM=('DASM','mean'),Mean_Signed_ln_RASM=('ln_RASM','mean'))
 rows=[]
 for band,group in individual.groupby('Band'):
  for variable in ['Mean_Abs_DASM','Mean_Abs_ln_RASM']:
   vals=group[variable]
   rows.append(dict(Scenario=scenario,Band=band,Index=variable,n=len(vals),median=vals.median(),Q1=vals.quantile(.25),Q3=vals.quantile(.75),minimum=vals.min(),maximum=vals.max()))
 return pd.DataFrame(rows)

scenarios=[(df,'Reference N35'),(df.loc[~df.Code.isin([24,27])],'Participant-exclusion N33'),
          (remove_pairs(df,BAD_PAIRS),'Signal-loss-pair exclusion N35'),
          (remove_pairs(df,{k:BAD_PAIRS.get(k,set())|ADDITIONAL_HIGH_AMPLITUDE.get(k,set()) for k in set(BAD_PAIRS)|set(ADDITIONAL_HIGH_AMPLITUDE)}),'Exploratory high-amplitude-pair exclusion N35')]
assert [d.Code.nunique() for d,_ in scenarios]==[35,33,35,35]
out=pd.concat([summary(d,name) for d,name in scenarios],ignore_index=True)
out.to_csv(OUTPUT/'RQ3_asymmetry_sensitivity_aggregate.csv',index=False)
print('Aggregated RQ3 sensitivity summary written to', OUTPUT)
