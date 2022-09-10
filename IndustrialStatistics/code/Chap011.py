## Chapter 11
#
# Industrial Statistics: A Computer Based Approach with Python<br>
# by Ron Kenett, Shelemyahu Zacks, Peter Gedeck
# 
# Publisher: Springer International Publishing; 1st edition (2023) <br>
# <!-- ISBN-13: 978-3031075650 -->
# 
# (c) 2022 Ron Kenett, Shelemyahu Zacks, Peter Gedeck
# 
# The code needs to be executed in sequence.
import os
os.environ['OUTDATED_IGNORE'] = '1'
import warnings
from outdated import OutdatedPackageWarning
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=OutdatedPackageWarning)

# Sampling Plans for Batch and Sequential Inspection
import random
import pandas as pd
import numpy as np
from scipy import stats, optimize
from scipy.special import gamma
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from mistat import acceptanceSampling
import mistat
import lifelines

from dataclasses import dataclass
from typing import List, Optional


## General Discussion
## Single-Stage Sampling Plans for Attributes
# note that scipy calls the pdf of discrete distributions pmf
j = [0, 1, 2, 3, 4]
print(pd.DataFrame({
    'j': j,
    'pdf': stats.hypergeom.pmf(j, M=100, n=10, N=5),
    'cdf': stats.hypergeom.cdf(j, M=100, n=10, N=5),
}).round(4))

j = [0, 1, 2, 3, 4, 5]
print(pd.DataFrame({
    'j': j,
    'pdf': stats.hypergeom.pmf(j, M=100, n=10, N=10),
    'cdf': stats.hypergeom.cdf(j, M=100, n=10, N=10),
}).round(4))

from mistat.acceptanceSampling import findPlan
findPlan(PRP=[0.01, 0.95], CRP=[0.08, 0.05], oc_type='hypergeom', N=100)

p0 = np.array([*[0.01] * 10, *[0.03] * 10])
pt = np.linspace(0.05, 0.32, 10)
pt = np.array([*pt, *pt])

result = []
for p0i, pti in zip(p0, pt):
    result.append([p0i, pti, *findPlan(PRP=(p0i, 0.95), CRP=(pti, 0.05),
                   oc_type="hypergeom", N=100)])
result = pd.DataFrame(result, columns=['p0', 'pt', 'n', 'c', 'r'])

def latexTable(result):
    style = result.style.hide(axis='index')
    style = style.format(subset=['p0', 'pt'], precision=2)
    s = style.to_latex(hrules=True)
    s = s.replace('p0', '$p_0$').replace('pt', '$p_t$')
    s = s.replace(' n ', ' $n$ ').replace(' r ', ' $r$ ')
    return s.replace(' c ', ' $c$ ')

p0 = np.array([*[0.01] * 10, *[0.03] * 10])
pt = np.linspace(0.05, 0.32, 10)
pt = np.array([*pt, *pt])

result = []
for p0i, pti in zip(p0, pt):
    result.append([p0i, pti, *findPlan(PRP=(p0i, 0.90), CRP=(pti, 0.20),
                   oc_type="hypergeom", N=100)])
result = pd.DataFrame(result, columns=['p0', 'pt', 'n', 'c', 'r'])

from mistat.acceptanceSampling import OperatingCharacteristics2c

X = OperatingCharacteristics2c(50, 1, oc_type='hypergeom', N=100,
                               pd=np.linspace(0, 0.15, 300))
df = pd.DataFrame({'p': X.pd, 'OC(p)': X.paccept})
ax = df.plot(x='p', y='OC(p)', legend=False, linestyle=':', color='grey')
ax.set_ylabel('OC(p)')

X = OperatingCharacteristics2c(50, 1, oc_type='hypergeom', N=100,
                               pd=[i / 100 for i in range(16)])
df = pd.DataFrame({'p': X.pd, 'OC(p)': X.paccept})
ax = df.plot.scatter(x='p', y='OC(p)', legend=False, ax=ax, color='black')
plt.show()

def latexTable(result):
    style = result.style.hide(axis='index')
    style = style.format(subset='p', precision=2)
    s = style.to_latex(column_format='cc', hrules=True)
    s = s.replace(' p ', ' $p$ ').replace('OC(p)', '$\mbox{OC}(p)$')
    return s

import math
# Note that the naming of the arguments follows the convention in scipy
# and therefore differs from the book
def normalApproximationOfH(j, M, n, N):
    P = N / M
    Q = 1 - P
    return stats.norm.cdf((j + 0.5 - n*P) / math.sqrt(n * P * Q * (1 - n/M)))

df0 = pd.DataFrame({
    'a': np.array(range(20)),
})
idx = [('', 'a')]
j = np.array(range(14))
df1 = pd.DataFrame({
    'H(j;100,30,20)': stats.hypergeom.cdf(j, M=100, n=20, N=30),
    'Normal': normalApproximationOfH(j, M=100, n=20, N=30),
})
idx.extend([('H(j;100,30,20)', 'Hypergeometric'), ('H(j;100,30,20)', 'Normal')])
j = np.array(range(18))
df2 = pd.DataFrame({
    'H(j;100,50,20)': stats.hypergeom.cdf(j, M=100, n=20, N=50),
    'Normal': normalApproximationOfH(j, M=100, n=20, N=50),
})
idx.extend([('H(j;100,50,20)', 'Hypergeometric'), ('H(j;100,50,20)', 'Normal')])
j = np.array(range(20))
df3 = pd.DataFrame({
    'H(j;100,80,20)': stats.hypergeom.cdf(j, M=100, n=20, N=80),
    'Normal': normalApproximationOfH(j, M=100, n=20, N=80),
})
idx.extend([('H(j;100,80,20)', 'Hypergeometric'), ('H(j;100,80,20)', 'Normal')])
df = pd.concat([df0, df1, df2, df3], axis=1)

df.columns = pd.MultiIndex.from_tuples(idx, names=["first", "second"])
style = df.style.hide(axis='index')
style = style.format(precision=4, na_rep='')
print(style.to_latex(hrules=True, column_format='ccccccc', multicol_align='c'))

## Approximate Determination of the Sampling Plan
from mistat.acceptanceSampling import findPlanApprox

def attainedRiskLevels(plan, p0, pt):
  hat_alpha = 1 - stats.hypergeom(N, int(p0 * N), plan.n).cdf(plan.c)
  hat_beta = stats.hypergeom(N, int(pt * N), plan.n).cdf(plan.c)
  return np.array([hat_alpha, hat_beta])

print('Exact results (p0=0.01, pt=0.03)')
for N in (500, 1000, 2000):
    plan = findPlan(PRP=[0.01, 0.95], CRP=[0.03, 0.05], oc_type='hypergeom', N=N)
    print(N, plan, attainedRiskLevels(plan, 0.01, 0.03).round(3))


print('Approximate results (p0=0.01, pt=0.03)')
for N in (500, 1000, 2000):
    plan = findPlanApprox(PRP=[0.01, 0.95], CRP=[0.03, 0.05], N=N)
    print(N, plan, attainedRiskLevels(plan, 0.01, 0.03).round(3))

print('Exact results (p0=0.01, pt=0.05)')
for N in (500, 1000, 2000):
    plan = findPlan(PRP=[0.01, 0.95], CRP=[0.05, 0.05], oc_type='hypergeom', N=N)
    print(N, plan, attainedRiskLevels(plan, 0.01, 0.05).round(3))

print('Approximate results (p0=0.01, pt=0.05)')
for N in (500, 1000, 2000):
    plan = findPlanApprox(PRP=[0.01, 0.95], CRP=[0.05, 0.05], N=N)
    print(N, plan, attainedRiskLevels(plan, 0.01, 0.05).round(3))

## Double-Sampling Plans for Attributes
def latexTable(df):
  style = pd.DataFrame(df).style.hide(axis='index')
  style = style.format(subset='p', precision=3)
  style = style.format(subset='OC', precision=4)
  style = style.format(subset='ASN', precision=1)
  s = style.to_latex(column_format='ccc', hrules=True)
  s = s.replace(' p &', ' $p$ &')
  s = s.replace('OC', 'OC($p$)')
  s = s.replace('ASN', 'ASN($p$)')
  return s
dsPlan = acceptanceSampling.DSPlanHypergeom(150, 20, 40, 2, 6, 6, p=np.arange(0, 0.525, 0.025))
df = pd.DataFrame({
  'p': dsPlan.p,
  'OC': dsPlan.OC,
  'ASN': dsPlan.ASN,
})

dsPlan = acceptanceSampling.DSPlanHypergeom(150, 20, 40, 1, 3, 3, p=np.arange(0, 0.525, 0.025))
df_stringent = pd.DataFrame({
  'p': dsPlan.p,
  'OC': dsPlan.OC,
  'ASN': dsPlan.ASN,
})

combined = pd.DataFrame({
    'p': df['p'],
    'OC (20,40,2,6,6)': df['OC'],
    'OC (20,40,1,3,3)': df_stringent['OC'],
})
ax = combined.plot(x='p', y='OC (20,40,2,6,6)', color='black', linestyle='dotted', zorder=10)
combined.plot(x='p', y='OC (20,40,1,3,3)', ax=ax, color='black', zorder=10)
ax.axvline(0.1, color='lightgray', zorder=5)
ax.axvline(0.15, color='lightgray', zorder=5)
ax.set_xlabel('Percentage defectives')
ax.set_ylabel('Probability of accepting defectives')
plt.tight_layout()

dsPlan = acceptanceSampling.DSPlanHypergeom(1000, 100, 200, 3, 6, 6, p=np.arange(0.01, 0.1, 0.01))
dsPlanApprox = acceptanceSampling.DSPlanNormal(1000, 100, 200, 3, 6, 6, p=np.arange(0.01, 0.1, 0.01))
df = pd.DataFrame({
  'p': dsPlan.p,
  'OC Exact': dsPlan.OC,
  'OC Approx.': dsPlanApprox.OC,
  'ASN Exact': dsPlan.ASN,
  'ASN Approx.': dsPlanApprox.ASN,
})
style = pd.DataFrame(df).style.hide(axis='index')
style = style.format(subset='p', precision=2)
style = style.format(subset=['OC Exact', 'OC Approx.'], precision=3)
style = style.format(subset=['ASN Exact', 'ASN Approx.'], precision=1)
s = style.to_latex(column_format='c|cc|cc', hrules=True)
s = s.replace(' p &', ' $p$ ak&')
s = s.replace('OC Exact', 'Exact')
s = s.replace('OC Approx.', 'Approx.')
s = s.replace('ASN Exact', 'Exact')
s = s.replace('ASN Approx.', 'Approx.')
s = s.replace(r'\toprule', r'\toprule & \multicolumn{2}{c|}{OC($p$)} & \multicolumn{2}{c}{ASN($p$)} \\')
print(s)

## Sequential Sampling and A/B testing
### The One-Armed Bernoulli Bandits
#### I.  The Bayesian Strategy
print(f'0.1-quantile B(10,0.6):  {stats.binom(10, 0.6).ppf(0.1)}')

print(f'B(5,7)-cdf :  {stats.beta(5, 7).cdf(0.5)}')

print(f'B(5,8)-cdf :  {stats.beta(5, 8).cdf(0.5)}')

np.random.seed(5)
print(f'{stats.binom(10, 0.3).rvs(1)} wins')

print(f'B(3,9)-cdf :  {stats.beta(3, 9).cdf(0.5)}')

np.random.seed(1)

N=50; lambda_=0.5; k0=10; gamma=0.95; Ns=1000

results = []
for p in (0.4, 0.45, 0.5, 0.55, 0.6, 0.7):
  r = acceptanceSampling.simulateOAB(N, p, lambda_, k0, gamma, Ns)
  results.append({
      'p': p,
      'Mgamma_mean': r.mgamma.mean,
      'Mgamma_std': r.mgamma.std,
      'Reward_mean': r.reward.mean,
      'Reward_std': r.reward.std,
  })

style = pd.DataFrame(results).style.hide(axis='index')
style = style.format(precision=3)
style = style.format(subset='p', precision=2)
s = style.to_latex(column_format='ccccc', hrules=True)
s = s.replace(' p &', ' $p$ &')
s = s.replace('Mgamma_mean', '$E\\{M_{\gamma}\\}$')
s = s.replace('Mgamma_std', '$\\text{std}\\{M_{\gamma}\\}$')
s = s.replace('Reward_mean', '$E\\{\\text{Reward}\\}$')
s = s.replace('Reward_std', '$\\text{std}\\{\\text{Reward}\\}$')
print(s)

from mistat.acceptanceSampling import optimalOAB
result = optimalOAB(10, 0.5)
print(f'Case (10, 0.5): {result.max_reward:.3f}')
print(f'Case (50, 0.5): {optimalOAB(50, 0.5).max_reward:.3f}')

table = pd.DataFrame(result.rewards)
table.index.name = 'n'
width = table.shape[1]
for i in range(1, len(table)):
  table.iloc[i, width-i:] = np.nan
style = table.style
style = style.format(precision=3, na_rep='')
s = style.to_latex(hrules=True)
print(s)

### Two-Armed Bernoulli Bandits
optimalOAB(45, 0.143).max_reward

## Acceptance Sampling Plans for Variables
## Rectifying Inspection of Lots
from mistat.acceptanceSampling import SSPlanBinomial
plan = SSPlanBinomial(N=1000, n=250, Ac=5,
                      p=np.linspace(0.005, 0.035, 200))
plan.plot()
plt.tight_layout()

plan = SSPlanBinomial(N=1000, n=250, Ac=5,
                      p=np.linspace(0.0, 0.15, 200))
plan.plot()
plt.tight_layout()

## National and International Standards
plan = acceptanceSampling.SSPlanBinomial(1000, 80, 2,
                                         p=(0.01, 0.02, 0.03, 0.04, 0.05))

## Skip-Lot Sampling Plans for Attributes
### The ISO 2859 Skip-Lot Sampling Procedures
#### Skip Lot Switching Rules
## The Deming Inspection Criterion
## Published Tables for Acceptance Sampling
#### I.  
#### II.  
## Sequential Reliability Testing
# use Goel-Okumoto model as the example
a = 1
b = 0.75

t = np.linspace(0, 5, 100)

curves = pd.DataFrame({
    't': t,
    'm(t)':  a * (1 - np.exp(-b*t)),
    'lambda(t)': a * b * np.exp(-b*t),
})
ax = curves.plot(x='t', y='m(t)', color='black', label='$m(t)$')
curves.plot(x='t', y='lambda(t)', ax=ax, color='black', linestyle=':', label='$\lambda(t)$')
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.set_xlabel('Time $t$')
plt.show()

def GoelOkumoto(t, a, b):
    return a * (1 - np.exp(-b * t))

def Yamada(t, a, b):
    return a * (1 - (1+b*t)*np.exp(-b*t))

warnings.filterwarnings('ignore', category=RuntimeWarning)

def optimizeModelFit(model, data):
    fit = optimize.curve_fit(model, data['T'], data['CFC'])
    popt = fit[0]
    # add the fit to the data set
    data[model.__name__] = [model(t, *popt) for t in data['T']]
    return popt

data = mistat.load_data('FAILURE_J3')
goFit = optimizeModelFit(GoelOkumoto, data)
ohbaFit = optimizeModelFit(Yamada, data)

fig, axes = plt.subplots(ncols=2)
data.plot(x='T', y='CFC', color='grey', ax=axes[0])
data.plot(x='T', y='GoelOkumoto', color='black', ax=axes[0])
axes[0].set_title('Goel-Okumoto')
axes[0].set_xlabel('Cumulative failure count')
axes[0].set_xlabel('Time')
axes[0].set_ylim(0, 400)
data.plot(x='T', y='CFC', color='grey', ax=axes[1])
data.plot(x='T', y='Yamada', color='black', ax=axes[1])
axes[1].set_title('Yamada')
axes[1].set_xlabel('Time')
axes[1].set_ylim(0, 400)
plt.tight_layout()
plt.show()

## Chapter Highlights
## Exercises
