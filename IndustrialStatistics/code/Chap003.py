## Chapter 3
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

# Advanced Methods of Statistical Process Control
import numpy as np
import pandas as pd
from scipy import stats
import mistat
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

## Tests of Randomness
### Testing the Number of Runs
### Runs Above and Below a Specified Level
rnorm10 = mistat.load_data('RNORM10')
ax = rnorm10.plot(marker='o', linestyle=':', color='black')
ax.axhline(10, color='grey')
ax.set_ylabel('Sample')
ax.set_xlabel('Index')
plt.show()

rnorm10 = mistat.load_data('RNORM10')

x = [0 if xi <= 10 else 1 for xi in rnorm10]
_ = mistat.runsTest(x, alternative='less', verbose=True)

### Runs Up and Down
### Testing the Length of Runs Up and Down
## Modified Shewhart Control Charts for $\bar X$
Ps = mistat.PistonSimulator(n_simulation=5 * 20, seed=1).simulate()
Ps = mistat.simulationGroup(Ps, 5)

cycleTime = mistat.qcc_groups(Ps['seconds'], Ps['group'])
qcc = mistat.QualityControlChart(cycleTime)

fig, ax = plt.subplots(figsize=(8, 6))
qcc.plot(title='for cycleTime', ax=ax)
st = qcc.std_dev / np.sqrt(qcc.sizes[0])
ax.axhline(qcc.center + 1.5 * st, linestyle=':', color='black')
ax.axhline(qcc.center - 1.5 * st, linestyle=':', color='black')
plt.show()

## The Size and Frequency of Sampling for Shewhart Control Charts
### The Economic Design for $\bar X$-charts
### Increasing The Sensitivity of $p$-charts
jandefect = mistat.load_data('JANDEFECT')
qcc = mistat.QualityControlChart(jandefect, qcc_type='p', sizes=20,
                                 center=0.048, std_dev=np.sqrt(0.048 * (1 - 0.048)))
qcc.oc_curves()

## Cumulative Sum Control Charts
### Upper Page's Scheme
np.random.seed(1)
x = [*stats.norm(10).rvs(20), *stats.norm(13).rvs(20)]

analysis = mistat.Cusum(x, center=10)
ax = pd.Series(analysis.pos).plot(marker='o', color='black')
ax.set_xlabel('Group')
ax.set_ylabel('Cumulative Sum')
plt.show()

ipl = mistat.load_data('IPL')

analysis = mistat.Cusum(ipl, center=1.07, std_dev=1, se_shift=0, sizes=1, decision_interval=4.16)
ax = pd.Series(analysis.pos).plot(marker='o', color='black')
ax.set_xlabel('Group')
ax.set_ylabel('Cumulative Sum')
ax.axhline(analysis.decision_interval, color='gray', linestyle=':')
plt.show()

### Some Theoretical Background
#### A. Normal Distribution
#### B. Binomial Distributions
#### C. Poisson Distributions
### Lower and Two-Sided Page's Scheme
coal = mistat.load_data('COAL')
coal.index = range(1850, 1961)

_, ax = plt.subplots(figsize=(8, 6))
# use zorder to plot the dots over the bars
ax.scatter(coal.index, coal, color='black', zorder=2)
ax.bar(coal.index, coal, color='lightgrey', zorder=1)
plt.show()

analysis = mistat.Cusum(coal[:50], center=1.82, std_dev=1, se_shift=0, decision_interval=4.19)
ax = pd.Series(analysis.neg).plot(color='lightgrey', marker='o',
      markerfacecolor='black', markeredgecolor='black')
ax.set_xlabel('Group')
ax.set_ylabel('Cumulative Sum')
ax.axhline(-analysis.decision_interval, color='grey', linestyle=':')
plt.show()

thickdiff = mistat.load_data('THICKDIFF')

analysis = mistat.Cusum(thickdiff, center=0, std_dev=1, se_shift=6,
                        decision_interval=9)
analysis.plot()
plt.show()

### Average Run Length, Probability of False Alarm And Conditional Expected Delay
results = []
for loc in (0, 0.5, 1.0, 1.5):
    arl = mistat.cusumArl(randFunc=stats.norm(loc=loc), N=100,
                          limit=10_000, seed=100, verbose=False)
    results.append({
        'theta': loc,
        'ARL': arl['statistic']['ARL'],
        '2 S.E.': 2 * arl['statistic']['Std. Error'],
    })
print(pd.DataFrame(results))

arl = mistat.cusumArl(randFunc=stats.norm(loc=10, scale=5), N=300, limit=7000, seed=1,
                     kp=12, km=8, hp=29, hm=-29, verbose=False)
rls = [r.rl for r in arl['run']]
ax = pd.Series(rls).hist(bins=15, color='lightgrey', zorder=1)
ax.set_xlabel('Run Length')
arlstats = arl['statistic']
ax.axvline(arlstats['ARL'], linestyle=':', zorder=3, color='black')
ax.fill_between([arlstats['ARL'] - arlstats['Std. Error'], arlstats['ARL'] + arlstats['Std. Error']],
        120, color='grey', zorder=10, alpha=0.5)
ax.text(510, 110, f"ARL(0) = {arlstats['ARL']:.0f} $\pm$ {arlstats['Std. Error']:.1f}")
plt.show()

for h in (18.7, 28, 28.5, 28.6, 28.7, 29, 30):
  arl = mistat.cusumArl(randFunc=stats.norm(loc=10, scale=5),
          N=300, limit=7000, seed=1, kp=12, km=8, hp=h, hm=-h,
          verbose=False)
  print(f"h {h:5.1f}: ARL(0) {arl['statistic']['ARL']:5.1f} ",
        f"+/- {arl['statistic']['Std. Error']:4.1f}")

results = []
for p in (0.05, 0.06, 0.07):
    arl = mistat.cusumArl(randFunc=stats.binom(n=100, p=p), N=100, limit=2000,
                          seed=1, kp=5.95, km=3.92, hp=12.87, hm=-8.66)
    results.append({
        'p': p,
        'delta': p/0.05,
        'ARL': arl['statistic']['ARL'],
        '2 S.E.': 2 * arl['statistic']['Std. Error'],
    })
print(pd.DataFrame(results))

arl = mistat.cusumArl(randFunc=stats.poisson(mu=10), N=100, limit=2000, seed=1,
                     kp=12.33, km=8.41, hp=11.36, hm=-12.91)
arl['statistic']

results = []
for loc in (0.5, 1.0, 1.5):
    pfaced = mistat.cusumPfaCed(randFunc1=stats.norm(),
                                randFunc2=stats.norm(loc=loc),
                                tau=100, N=100, limit=1_000, seed=1,
                                verbose=False)
    results.append({
        'theta': loc,
        'PFA': pfaced['statistic']['PFA'],
        'CED': pfaced['statistic']['CED'],
        'S.E.': pfaced['statistic']['Std. Error'],
    })

print(pd.DataFrame(results))

## Bayesian Detection
common = {'mean0': 10, 'sd': 3, 'n': 5, 'tau': 10, 'w': 99, 'seed': 1,
          'verbose': False}
pd.DataFrame([
    mistat.shroArlPfaCedNorm(delta=0.5, **common)['statistic'],
    mistat.shroArlPfaCedNorm(delta=1.0, **common)['statistic'],
    mistat.shroArlPfaCedNorm(delta=1.5, **common)['statistic'],
    mistat.shroArlPfaCedNorm(delta=2.0, **common)['statistic'],
], index=[0.5, 1.0, 1.5, 2.0])

common = {'mean0': 10, 'sd': 3, 'n': 5, 'delta': 2.0, 'seed': 1,
          'verbose': False}
pd.DataFrame([
    mistat.shroArlPfaCedNorm(w=19, **common)['statistic'],
    mistat.shroArlPfaCedNorm(w=50, **common)['statistic'],
    mistat.shroArlPfaCedNorm(w=99, **common)['statistic'],
], index=[19, 50, 99])

result = mistat.shroArlPfaCedNorm(mean0=10, sd=3, n=5, delta=2.0, w=99, seed=1, verbose=False)

ax = pd.Series(result['rls']).plot.box(color='black')
ax.set_ylabel('Run Length')
plt.show()

## Process Tracking
### The EWMA Procedure
np.random.seed(1)

x = [*stats.norm(loc=10, scale=3).rvs(11*5), *stats.norm(loc=14).rvs(9*5)]

groups = [y for xi in range(1, 21) for y in [xi]*5]
grouped = mistat.qcc_groups(x, groups=groups)

ewma = mistat.EWMA(grouped, center=10, std_dev=3, smooth=0.2, nsigmas=3)
ewma.plot()
plt.show()

### The BECM Procedure
### The Kalman Filter
dojo1935 = mistat.load_data('DOJO1935')

# solve the regression equation
m = 20
sqrt_t = np.sqrt(range(1, m + 1))
df = pd.DataFrame({
    'Ut': dojo1935[:m]/sqrt_t,
    'x1t': 1 / sqrt_t,
    'x2t': sqrt_t,
})
model = smf.ols(formula='Ut ~ x1t + x2t - 1', data=df).fit()
mu0, delta = model.params
var_eta = np.var(model.resid, ddof=2)
pd.Series({'mu0': mu0, 'delta': delta, 'Var(eta)': var_eta})

# choose sig2e and w20
sig2e = 0.0597
w20 = 0.0015

# apply the filter
results = []
mu_tm1 = mu0
w2_tm1 = w20
y_tm1 = mu0
for i in range(0, len(dojo1935)):
    y_t = dojo1935[i]
    B_t = sig2e / (var_eta + w2_tm1)
    mu_t = B_t * (mu_tm1 + delta) + (1 - B_t) * y_t
    results.append({
        't': i + 1, # adjust for Python indexing starting at 0
        'y_t': y_t,
        'mu_t': mu_t,
    })
    w2_tm1 = B_t * (var_eta - sig2e + w2_tm1)
    mu_tm1 = mu_t
    y_tm1 = y_t
results = pd.DataFrame(results)

fig, ax = plt.subplots()
# ax = dojo1935.plot(color='black')
ax.plot(results['t'], results['y_t'], color='grey')
ax.plot(results['t'], results['mu_t'], color='black')
ax.set_ylabel('Dow Jones')
plt.show()

def renderResults(results):
  style = results.iloc[:25,].style.hide(axis='index')
  style = style.format(precision=2)
  s = style.to_latex(hrules=True)
  s = s.replace('y_t', '$y_t$').replace('mu_t', '$\\mu_t$')
  print(s)

### Hoadley's QMP
soldef = mistat.load_data('SOLDEF')

print('Batches above quality standard: ', sum(soldef > 100))
print('Batches above UCL: ', sum(soldef > 130))

xbar = np.cumsum(soldef) / np.arange(1, len(soldef)+1)
results = []
for i in range(2, len(soldef)):
    xbar_tm1 = np.mean(xbar[i-1])
    S2_tm1 = np.var(soldef[:i])
    gamma_tm1 = S2_tm1/xbar_tm1 - 1
    nu_tm1 = xbar_tm1 / gamma_tm1
    result = {
        't': i + 1,
        'Xt': soldef[i],
        'xbar_tm1': xbar_tm1,
        'S2_tm1': S2_tm1,
        'Gamma_tm1': gamma_tm1,
        'nu_tm1': nu_tm1,
    }
    f = gamma_tm1 / (gamma_tm1 + 1)
    shape = nu_tm1 + soldef[i]
    result['lambda_t'] = f * shape
    result.update(((f'lambda({p})', f * stats.gamma.ppf(p, a=shape, scale=1))
                   for p in (0.01, 0.05, 0.95, 0.99)))
    results.append(result)
results = pd.DataFrame(results)

style = results.iloc[7:18,:6].style.hide(axis='index')
style = style.format(precision=2)
s = style.to_latex(hrules=True)
s = s.replace(' Xt ', ' $X_t$ ').replace('xbar_tm1', '$\\bar X_{t-1}$')
s = s.replace('S2_tm1', '$S^2_{t-1}$').replace(' t ', ' $t$ ')
s = s.replace('Gamma_tm1', '$\\hat\\Lambda_{t-1}$')
s = s.replace('nu_tm1', '$\\hat\\nu_{t-1}$')
print(s)

columns = ['t', 'lambda_t', 'lambda(0.01)', 'lambda(0.05)', 'lambda(0.95)', 'lambda(0.99)']
style = results[columns].iloc[7:18,:].style.hide(axis='index')
style = style.format(precision=2)
s = style.to_latex(hrules=True)
s = s.replace('lambda_t', '$\\lambda_t$')
for p in (0.01, 0.05, 0.95, 0.99):
  s = s.replace(f'lambda({p})', f'$\\lambda_{{t,{p}}}$')
print(s)

## Automatic Process Control
c_A = 100
c_d = 1000
b = 1
q_tp1 = c_d
data = []
for t in range(14, 0, -1):
    q_t = c_A * q_tp1 / (c_A + q_tp1 * b**2)
    p_t = b * q_tp1 / (c_A + q_tp1 * b**2)
    data.append({'t': t, 'q_t': q_t, 'p_t': p_t})
    q_tp1 = q_t
result = pd.DataFrame(data)

result = pd.concat([pd.DataFrame({'t': [15]}), result]).reset_index(drop=True)
style = result.style.hide(axis='index')
style = style.format(na_rep='---', precision=3)
s = style.to_latex(hrules=True)
s = s.replace('q_t', '$q_t$').replace('p_t', '$p_t$')
print(s)

speed = mistat.load_data('FILMSP')

groups = [y for x in range(1, 44) for y in [x]*5]
grouped = mistat.qcc_groups(speed[:215], groups=groups)

ewma = mistat.EWMA(grouped, center=105, std_dev=6.53, nsigmas=2)
ax = ewma.plot()
plt.show()

## Chapter Highlights
## Exercises
