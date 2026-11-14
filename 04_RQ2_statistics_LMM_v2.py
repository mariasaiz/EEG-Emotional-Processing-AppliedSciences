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

# 04: RQ2 analyses and per-band full-model residual diagnostics.
# 04: RQ2 distributional checks, Mann–Whitney U, Friedman and maximum-likelihood mixed models.
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro, mannwhitneyu, norm, friedmanchisquare, chi2
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
import scipy.stats as st

OUT = SOCIO_DIR / "RQ2_results"
OUT.mkdir(parents=True, exist_ok=True)
eeg = pd.read_csv(PREPROCESS_DIR / "BandPower_stimulus_band_mean.csv")
socio = pd.read_excel(SOCIO_DIR / "BBDD_SOCIO_FINAL_N35.xlsx")
expected = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,33,34,35,37,38}
assert set(eeg.Code)==set(socio.Code)==expected
assert len(eeg)==875 and eeg.Code.nunique()==35
assert set(socio.Group_label)=={'Students','Lecturers'}
assert socio.Group_label.value_counts().to_dict()=={'Lecturers':25,'Students':10}
assert eeg.Mean_power_12ch.gt(0).all()
assert not eeg.duplicated(['Code','Stimulus','Band']).any()
data = eeg.merge(socio[['Code','Group_label']],on='Code',validate='many_to_one')
stimuli=['Surprise','Anger','Happiness','Sadness','Fear']
bands=['Delta','Theta','Alpha','Beta','LowGamma']

# Shapiro–Wilk by group, stimulus and band.
normality=[]
for group in ['Students','Lecturers']:
 for stimulus in stimuli:
  for band in bands:
   y=data.loc[(data.Group_label==group)&(data.Stimulus==stimulus)&(data.Band==band),'Mean_power_12ch']
   W,p=shapiro(y)
   normality.append(dict(Group=group,Stimulus=stimulus,Band=band,n=len(y),Shapiro_W=W,p=p))
pd.DataFrame(normality).to_csv(OUT/'RQ2_normality.csv',index=False)

# Signed Mann–Whitney effect size: direction is based on the Student U statistic.
inter=[]
for stimulus in stimuli:
 for band in bands:
  d=data[(data.Stimulus==stimulus)&(data.Band==band)]
  a=d.loc[d.Group_label=='Students','Mean_power_12ch'].to_numpy()
  b=d.loc[d.Group_label=='Lecturers','Mean_power_12ch'].to_numpy()
  U,p=mannwhitneyu(a,b,alternative='two-sided',method='auto')
  zmag=norm.isf(p/2) if p<1 else 0.
  signed_r=float(np.sign(U-len(a)*len(b)/2)*zmag/np.sqrt(len(a)+len(b)))
  inter.append(dict(Stimulus=stimulus,Band=band,U=U,p=p,signed_r=signed_r))
inter=pd.DataFrame(inter)
inter['p_Holm']=multipletests(inter.p,method='holm')[1]
inter.to_csv(OUT/'RQ2_intergroup_signed_effects.csv',index=False)

# Friedman within each group and frequency band.
intra=[]
for group in ['Students','Lecturers']:
 for band in bands:
  d=data[(data.Group_label==group)&(data.Band==band)]
  wide=d.pivot(index='Code',columns='Stimulus',values='Mean_power_12ch')[stimuli]
  assert not wide.isna().any().any()
  statistic,p=friedmanchisquare(*(wide[col] for col in stimuli))
  intra.append(dict(Group=group,Band=band,n=len(wide),chi2=statistic,p=p,Kendall_W=statistic/(len(wide)*(len(stimuli)-1))))
intra=pd.DataFrame(intra)
intra['p_Holm']=np.nan
for group in intra.Group.unique():
 mask=intra.Group==group
 intra.loc[mask,'p_Holm']=multipletests(intra.loc[mask,'p'],method='holm')[1]
intra.to_csv(OUT/'RQ2_Friedman.csv',index=False)

# Full and reduced random-intercept LMMs fitted with ML; compare LR values to published Table 9.
LRT_REFERENCE={'Delta':5.400774,'Theta':8.323309,'Alpha':11.703620,'Beta':7.229186,'LowGamma':1.368736}
data['log_Power']=np.log(data.Mean_power_12ch)
data['Group_label']=pd.Categorical(data.Group_label,categories=['Students','Lecturers'])
data['Stimulus']=pd.Categorical(data.Stimulus,categories=stimuli)
LRT=[]; fixed=[]; residstats=[]
for band in bands:
 d=data.loc[data.Band==band].copy()
 reduced=smf.mixedlm('log_Power ~ C(Group_label) + C(Stimulus)',data=d,groups=d.Code).fit(reml=False,method='lbfgs')
 full=smf.mixedlm('log_Power ~ C(Group_label) * C(Stimulus)',data=d,groups=d.Code).fit(reml=False,method='lbfgs')
 LR=2*(full.llf-reduced.llf);df_diff=len(full.fe_params)-len(reduced.fe_params)
 LRT.append(dict(Band=band,LR_chi2=LR,df=df_diff,p=chi2.sf(LR,df_diff),Full_converged=bool(full.converged),Reduced_converged=bool(reduced.converged),Matches_Table9=abs(LR-LRT_REFERENCE[band])<0.01))
 ci=full.conf_int().loc[full.fe_params.index]
 for term, beta in full.fe_params.items():
  fixed.append(dict(Band=band,Term=term,beta=beta,SE=full.bse_fe[term],CI95_lower=ci.loc[term,0],CI95_upper=ci.loc[term,1]))
 resid=np.asarray(full.resid); fitted=np.asarray(full.fittedvalues)
 sw=shapiro(resid)
 residstats.append(dict(Band=band,Shapiro_W=sw.statistic,Shapiro_p=sw.pvalue,n=len(resid)))
 fig,ax=plt.subplots(1,2,figsize=(11,4.5))
 ax[0].scatter(fitted,resid,s=18,alpha=.65);ax[0].axhline(0,color='black',lw=.8)
 ax[0].set(xlabel='Fitted log power',ylabel='Conditional residual',title=f'{band}: residuals vs fitted')
 st.probplot(resid,dist='norm',plot=ax[1]);ax[1].set_title(f'{band}: residual normal Q–Q')
 fig.tight_layout();fig.savefig(OUT/f'RQ2_residuals_{band}.png',dpi=300);plt.close(fig)
lrt=pd.DataFrame(LRT);lrt['p_Holm']=multipletests(lrt.p,method='holm')[1]
lrt.to_csv(OUT/'RQ2_LRT_convergence.csv',index=False)
pd.DataFrame(fixed).to_csv(OUT/'RQ2_fixed_effects_CI95.csv',index=False)
pd.DataFrame(residstats).to_csv(OUT/'RQ2_residual_diagnostics.csv',index=False)
assert lrt.Matches_Table9.all() and lrt.Full_converged.all() and lrt.Reduced_converged.all(), 'The original Table 9 likelihood-ratio values or convergence checks were not reproduced.'
print('RQ2 Table 9 LR statistics reproduced. Results written to',OUT)
