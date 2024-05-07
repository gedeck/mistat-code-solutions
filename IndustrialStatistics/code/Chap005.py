## Chapter 5
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
# 
# Python packages and Python itself change over time. This can cause warnings or errors. We
# "Warnings" are for information only and can usually be ignored. 
# "Errors" will stop execution and need to be fixed in order to get results. 
# 
# If you come across an issue with the code, please follow these steps
# 
# - Check the repository (https://gedeck.github.io/mistat-code-solutions/) to see if the code has been upgraded. This might solve the problem.
# - Report the problem using the issue tracker at https://github.com/gedeck/mistat-code-solutions/issues
# - Paste the error message into Google and see if someone else already found a solution
import os
os.environ['OUTDATED_IGNORE'] = '1'
import warnings
from outdated import OutdatedPackageWarning
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=OutdatedPackageWarning)

# Classical Design and Analysis of Experiments
import random
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats import anova
import seaborn as sns
import mistat
import matplotlib.pyplot as plt

## Basic Steps and Guiding Principles
## Blocking and Randomization
## Additive and Non-additive Linear Models
## The Analysis of Randomized Complete Block Designs
### Several Blocks, Two Treatments per Block:  Paired Comparison
#### The $t$-Test
#### Randomization Tests
random.seed(1)
X = [1.1, 0.3, -0.7, -0.1]
m = 20000

Di = pd.DataFrame([random.choices((-1, 1), k=len(X)) for _ in range(m)])
DiX = (Di * X)

np.mean(DiX.mean(axis=1) > np.mean(X))

X = [0.8, 0.6, 0.3, -0.1, 1.1, -0.2, 0.3, 0.5, 0.5, 0.3]
statistic, pvalue = stats.ttest_1samp(X, 0.0)
print(f't {statistic:.2f}')
print(f'pvalue {pvalue:.4f}')

random.seed(1)
X = [0.8, 0.6, 0.3, -0.1, 1.1, -0.2, 0.3, 0.5, 0.5, 0.3]
m = 200

Di = pd.DataFrame([random.choices((-1, 1), k=len(X)) for _ in range(m)])
DiX = (Di * X)

means = DiX.mean(axis=1)
mistat.stemLeafDiagram(means, 2, leafUnit=0.01)

random.seed(1)
X = [0.8, 0.6, 0.3, -0.1, 1.1, -0.2, 0.3, 0.5, 0.5, 0.3]
m = 200

Di = pd.DataFrame([random.choices((-1, 1), k=len(X)) for _ in range(m)])
DiX = (Di * X)

means = DiX.mean(axis=1)
Pestimate = np.mean(DiX.mean(axis=1) > np.mean(X))
print(f'P_estimate: {Pestimate}')

### Several Blocks, $t$ Treatments per Block
hadpas = mistat.load_data('HADPAS')

model = smf.ols('res3 ~ C(diska) + C(hyb)', data=hadpas).fit()
print(anova.anova_lm(model))

model.conf_int().tail(5)

ci = model.conf_int().tail(5)
hyb_mean = hadpas.groupby(by='hyb').mean()['res3'] - hadpas['res3'].mean()
print(hyb_mean.round(2))
(ci.iloc[:,1] - ci.iloc[:,0]) / 2

## Balanced Incomplete Block Designs
## Latin Square Design
fig, axes = plt.subplots(nrows=3, figsize=(5, 10))
keyboards = mistat.load_data('KEYBOARDS.csv')
effects = [('keyboard', 'Keyboard'), ('job', 'Job type'), ('typist', 'Typist')]

for ax, (effect, label) in zip(axes, effects):
  keyboards.boxplot(column='errors', by=effect, color='black', ax=ax)
  ax.set_title('')
  ax.get_figure().suptitle('')
  ax.set_xlabel(label)
  ax.set_ylabel('Errors')
plt.subplots_adjust(hspace=0.3)
plt.show()

keyboards = mistat.load_data('KEYBOARDS.csv')
model = smf.ols('errors ~ C(keyboard) + C(job) + C(typist)', data=keyboards).fit()
print(anova.anova_lm(model))

## Full Factorial Experiments
### The Structure of Factorial Experiments
### The ANOVA for Full Factorial Designs
from mistat.design import doe
np.random.seed(2)

# Build design from factors
FacDesign = doe.full_fact({
    'k': [1500, 3000, 4500],
    's': [0.005, 0.0125, 0.02],
})

# Randomize design
FacDesign = FacDesign.sample(frac=1).reset_index(drop=True)

# Setup and run simulator with five replicates 
# for each combination of factors
simulator = mistat.PistonSimulator(n_replicate=5, **FacDesign,
                                   m=30, v0=0.005, p0=95_000, t=293, t0=350)
result = simulator.simulate()

model = smf.ols('seconds ~ C(k) * C(s)', data=result).fit()
print(anova.anova_lm(model).round(4))

from matplotlib.ticker import FormatStrFormatter

fig, axes = plt.subplots(ncols=2)
result.boxplot('seconds', by='k', color='black', ax=axes[0])
result.boxplot('seconds', by='s', color='black', ax=axes[1])
for ax in axes:
    ax.set_title('')
    ax.get_figure().suptitle('')
axes[0].set_ylabel('seconds')

# without using the formatter, the tick labels for s are 
# printed without rounding - try it
svalues = [0.005, 0.0125, 0.020]
axes[1].xaxis.set_major_formatter(lambda x, _: svalues[x-1])
plt.tight_layout()
plt.show()

from matplotlib.ticker import FormatStrFormatter

fig, axes = plt.subplots(figsize=(10,4), ncols=2)

linestyle = ['-', '--', ':']
marker = ['s', 'o', '^']
grouped = result.groupby(by=['s', 'k'], as_index=False).mean()
for i, (k, g) in enumerate(grouped.groupby('k')):
    g.plot(x='s', y='seconds', ax=axes[0], label=f'k={k:.0f}',
           color='black', marker=marker[i], markersize=8, linestyle=linestyle[i])
axes[0].set_ylabel('mean of Ps seconds')
axes[0].xaxis.set_major_formatter(FormatStrFormatter('%.4g'))

grouped = result.groupby(by=['k', 's'], as_index=False).mean()
for i, (s, g) in enumerate(grouped.groupby('s')):
    g.plot(x='k', y='seconds', ax=axes[1], label=f's={s:.4g}',
           color='black', marker=marker[i], markersize=8, linestyle=linestyle[i])
ax.set_ylabel('mean of Ps seconds')
plt.show()

anova_result = anova.anova_lm(model)
not_signif = ['C(k)', 'C(k):C(s)', 'Residual']
SS = anova_result['sum_sq'].loc[not_signif].sum()
DF = anova_result['df'].loc[not_signif].sum()
sigma2 = SS / DF
print(SS, DF, sigma2)

Ymean = result.groupby('s').mean()['seconds']
print('Ymean', Ymean)
print('Grand', Ymean.sum() / 3)
print('Main effects', Ymean - Ymean.sum() / 3)

Grand = Ymean.sum() / 3
MainEffects = Ymean - Grand
SEtau = np.sqrt(sigma2 / 30)

Salpha = np.sqrt(2 * stats.f.ppf(0.95, 2, 42))

tau = Ymean - Ymean.sum() / 3
df = pd.DataFrame({
  'tau': tau,
  'lower limit': tau - SEtau*Salpha,
  'upper limit': tau + SEtau*Salpha,
})
df.round(4)

### Estimating Main Effects and Interactions
### $2^m$ Factorial Designs
d1 = {
    'A': [-1, 1],
    'B': [-1, 1],
    'C': [-1, 1],
    'D': [-1, 1],
    'E': [-1, 1],
}
mistat.addTreatments(doe.frac_fact_res(d1, 4), mainEffects=['A', 'B', 'C', 'D', 'E'])

d1 = {
    'A': [1, 2],
    'B': [1, 2],
    'C': [1, 2],
    'D': [1, 2],
    'E': [1, 2],
}
Design = doe.full_fact(d1)
Design = mistat.addTreatments(Design, mainEffects=['A', 'B', 'C', 'D', 'E'])
print(Design.head(3).round(0))
print(Design.tail(3).round(0))

np.random.seed(3)
factors = {
  'm': [30, 60],
  's': [0.005, 0.02],
  'v0': [0.002, 0.01],
  'k': [1000, 5000],
  't': [290, 296],
}
Design = doe.full_fact(factors)

# Randomize design
Design = Design.sample(frac=1).reset_index(drop=True)

# Run the simulation with 5 replications for each setting
simulator = mistat.PistonSimulator(**{k:list(Design[k]) for k in Design},
                                   p0=90_000, t0=340, n_replicate=5)
result = simulator.simulate()

table = result.groupby(list(factors.keys()), as_index=False)
table = table.agg({'seconds':['mean','std']})
table.columns = [' '.join(c).strip() for c in table.columns.to_flat_index()]

table

for column in ['m', 'k', 't']:
  table[column] = table[column].astype(np.int64)
style = table.style.hide(axis='index')
style = style.format(subset=['s', 'v0', 'seconds mean', 'seconds std'], precision=3)
style = style.format(subset=['m', 'k', 't'], precision=0)
s = style.to_latex(hrules=True, column_format='ccccccc')
s = s.replace('seconds mean', '$\\bar{Y}_\\nu$')
s = s.replace('seconds std', '$S_\\nu$')
print(s)
# Pooled standard deviation 13.7.41
byFactors = result.groupby(list(factors.keys()))
groupedStd = byFactors.std()['seconds']
pooledVar = np.mean(groupedStd**2)
Vparam = pooledVar  / (5 * len(byFactors))
SE = np.sqrt(Vparam)

# Perform analysis of variance
Design['response'] = result['seconds']
model = smf.ols('seconds ~ (m + s + v0 + k + t) ** 2', data=result).fit()
# print(anova.anova_lm(model))
print(f'r2={model.rsquared}')

import re
results = {
  'LSE': model.params,
  'S.E.': SE,
  't': model.params / SE,
}
pattern = r'\[.*?\]'
names = [re.sub(pattern, '', s) for s in model.params.index]
df = pd.DataFrame(results)
df.index = names
significance = ['**' if abs(t) > 7 else '*' if abs(t) > 2.6 else ''
                for t in df['t']]
df['significance'] = significance

df.round(5)

style = df.iloc[1:,:].style
style = style.format(subset='LSE', precision=4)
style = style.format(subset='S.E.', precision=5)
style = style.format(subset='t', precision=2)
s = style.to_latex(column_format='lrrrl', hrules=True)
s = s.replace('significance', '')
print(s)
print(np.var(model.predict(result)))

mistat.mainEffectsPlot(result[['m', 's', 'v0', 'k', 't', 'seconds']], 'seconds')
plt.show()

mistat.interactionPlot(result[['m', 's', 'v0', 'k', 't', 'seconds']], 'seconds')
plt.show()

_, ax = plt.subplots(figsize=[10, 4])
mistat.marginalInteractionPlot(result[['m', 's', 'v0', 'k', 't', 'seconds']], 'seconds', ax=ax)
plt.show()

### $3^m$ Factorial Designs
def getStandardOrder(levels, labels):
    parameter = ''
    omega = 0
    for i, (level, label) in enumerate(zip(levels, labels), 1):
        omega += level * 3**(i-1)
        if level == 1:
            parameter = f'{parameter}{label}'
        elif level == 2:
            parameter = f'{parameter}{label}2'
    if parameter == '':
        parameter = 'Mean'
    return {'omega': omega, 'Parameter': parameter}

stress = mistat.load_data('STRESS')
standardOrder = pd.DataFrame(getStandardOrder(row[['A','B','C']], 'ABC') 
                             for _, row in stress.iterrows())

# add information to dataframe stress and sort in standard order 
stress.index = standardOrder['omega']
stress['Parameter'] = standardOrder['Parameter']
stress = stress.sort_index()

def get_psi3m(m):
    psi31 = np.array([[1, -1, 1], [1, 0, -2], [1, 1, 1]] )
    if m == 1:
        return psi31
    psi3m1 = get_psi3m(m-1)
    return np.kron(psi31, psi3m1)

Y_3m = stress['stress']

psi3m = get_psi3m(3)
delta3m = np.matmul(psi3m.transpose(), psi3m)
inv_delta3m = np.diag(1/np.diag(delta3m))
gamma_3m = np.matmul(inv_delta3m, np.matmul(psi3m.transpose(), Y_3m))

estimate = pd.DataFrame({
    'Parameter': stress['Parameter'],
    'LSE': gamma_3m,
})
# determine Lambda0 set as interactions that include quadratic terms
lambda0 = [term for term in stress['Parameter'] if '2' in term and len(term) > 2]
estimate['Significance'] = ['n.s.' if p in lambda0 else '' 
                            for p in estimate['Parameter']]

# estimate sigma2 using non-significant terms in lambda0
sigma2 = 0
for idx, row in estimate.iterrows():
    p = row['Parameter']
    if p not in lambda0:
        continue
    idx = int(idx)
    sigma2 += row['LSE']**2 * np.sum(psi3m[:, idx]**2)
K0 = len(lambda0)
sigma2 = sigma2 / K0
table = estimate.style.hide(axis='index')
table = table.format(precision=3)
s = table.to_latex(hrules=True, column_format='lrcr')
s = s.replace('A2', 'A$^2$').replace('B2', 'B$^2$').replace('C2', 'C$^2$')
print(s)
Y_3m = stress['stress']

psi3m = get_psi3m(3)
delta3m = np.matmul(psi3m.transpose(), psi3m)
inv_delta3m = np.diag(1/np.diag(delta3m))
gamma_3m = np.matmul(inv_delta3m, np.matmul(psi3m.transpose(), Y_3m))

estimate = pd.DataFrame({
    'Parameter': stress['Parameter'],
    'LSE': gamma_3m,
})

# determine Lambda0 set as interactions that include quadratic terms
lambda0 = [term for term in stress['Parameter'] if '2' in term and len(term) > 2]
print(f'lambda0 : {lambda0}')
estimate['Significance'] = ['n.s.' if p in lambda0 else '' 
                            for p in estimate['Parameter']]

# estimate sigma2 using non-significant terms in lambda0
sigma2 = 0
for idx, row in estimate.iterrows():
    p = row['Parameter']
    if p not in lambda0:
        continue
    idx = int(idx)
    sigma2 += row['LSE']**2 * np.sum(psi3m[:, idx]**2)
K0 = len(lambda0)
sigma2 = sigma2 / K0
print(f'K0 = {K0}')
print(f'sigma2 = {sigma2.round(2)}')

n = len(psi3m)
variance = sigma2 / (n * np.sum(psi3m**2, axis=0))
estimate['S.E.'] = np.sqrt(n * variance)

estimate.round(3)

stress = mistat.load_data('STRESS')

# convert factor levels from (0,1,2) to (-1,0,1)
stress['A'] = stress['A'] - 1
stress['B'] = stress['B'] - 1
stress['C'] = stress['C'] - 1

mistat.mainEffectsPlot(stress, 'stress')
plt.show()

mistat.interactionPlot(stress, 'stress')
plt.show()

_, ax = plt.subplots(figsize=[7, 4])
mistat.marginalInteractionPlot(stress[['A', 'B', 'C', 'stress']], 'stress', ax=ax)
ax.set_ylabel('stress')
plt.show()

stress = mistat.load_data('STRESS')

# convert factor levels from (0,1,2) to (-1,0,1)
stress['A'] = stress['A'] - 1
stress['B'] = stress['B'] - 1
stress['C'] = stress['C'] - 1

# train a model including interactions and quadratic terms
formula = ('stress ~ A + B + C + A:B + A:C + B:C + A:B:C + ' +
           'I(A**2) + I(B**2) + I(C**2)')
model = smf.ols(formula, data=stress).fit()
table = model.summary2().tables[1].style.hide(axis='index')
table = table.format(precision=3)
s = table.to_latex(hrules=True, column_format='lrrrrrr')
print(s)
stress = mistat.load_data('STRESS')

# convert factor levels from (0,1,2) to (-1,0,1)
stress['A'] = stress['A'] - 1
stress['B'] = stress['B'] - 1
stress['C'] = stress['C'] - 1

# train a model including interactions and quadratic terms
formula = ('stress ~ A + B + C + A:B + A:C + B:C + A:B:C + ' +
           'I(A**2) + I(B**2) + I(C**2)')
model = smf.ols(formula, data=stress).fit()
model.summary2()

## Blocking and Fractional Replications of $2^m$ Factorial Designs
def renderDesign(design, mainEffects, to_latex=False):
    design = mistat.addTreatments(design, mainEffects)
    defining = set(design.columns) - set(mainEffects) - {'Treatments'}
    columns = [('Treatments', '')]
    columns.extend(('Main Effects', effect) for effect in mainEffects)
    columns.extend(('Defining Parameter', effect) for effect in design.columns if effect in defining)
    design.columns = pd.MultiIndex.from_tuples(columns, names=["first", "second"])
    if not to_latex:
        return design
    style = design.style.hide(axis='index')
    style = style.format(precision=0)
    return style.to_latex(hrules=True, column_format='c'+'r'*(len(columns)-1))

from pyDOE3 import fracfact
design = pd.DataFrame(fracfact('A B C ABC'), columns='A B C ABC'.split())

renderDesign(design, 'A B C'.split())

print(renderDesign(design, 'A B C'.split(), to_latex=True))
from pyDOE3 import fracfact
# define the generator
generator = 'A B C ABC'
design = pd.DataFrame(fracfact(generator), columns=generator.split())
block_n = design[design['ABC'] == -1]
block_p = design[design['ABC'] == 1]

renderDesign(block_n, ['A', 'B', 'C'])

renderDesign(block_p, ['A', 'B', 'C'])

mistat.subgroupOfDefining(['ABCH', 'ABEFG', 'BDEFH'])

mainEffects = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
defining = ['ABCH', 'ABEFG', 'BDEFH', 'BCEFH']
design = pd.DataFrame(fracfact(' '.join(mainEffects)), columns=mainEffects)
design = mistat.addTreatments(design, mainEffects)
subgroup = mistat.subgroupOfDefining(defining, noTreatment='(1)')
block1 = design[design['Treatments'].isin(subgroup)]
block1

## Exploration of Response Surfaces
### Second Order Designs
### Some Specific Second Order Designs
#### $3^k$-Designs
#### Central Composite Designs
factors = {
  's': [0.01, 0.015],
  'v0': [0.00625, 0.00875],
  'k': [2000, 4000],
  't0': [345, 355],
}
Design = doe.central_composite(factors, alpha='r', center=[4, 4])

simulator = mistat.PistonSimulator(**Design, m=60, p0=110_000, t=296, 
                                   n_replicate=50, seed=2)
result = simulator.simulate()

# calculate mean and std of response by group
result = result.groupby(by='group')
result = result.agg({'s': 'mean', 'v0': 'mean', 'k': 'mean',
                     't0': 'mean', 'seconds':['mean','std']})
result.columns = ['s', 'v0', 'k', 't0', 'Ymean', 'Ystd']

# transformation between factors and code levels
factor2x = {factor: f'x{i}' for i, factor in enumerate(factors, 1)}
x2factor = {f'x{i}': factor for i, factor in enumerate(factors, 1)}
center = {factor: 0.5 * (max(values) + min(values))
          for factor, values in factors.items()}
unit = {factor: 0.5 * (max(values) - min(values))
        for factor, values in factors.items()}

# define helper function to convert code co-ordinates to factor co-ordinates
def toFactor(code, codeValue):
    ''' convert code to factor co-ordinates '''
    factor = x2factor[code]
    return center[factor] + codeValue * unit[factor]

# add code levels to table
for c in factors:
    result[factor2x[c]] =  (result[c] - center[c]) / unit[c]

table = result[['x1', 'x2', 'x3', 'x4', 'Ymean', 'Ystd']]
table.round(6)

table = result[['x1', 'x2', 'x3', 'x4', 'Ymean', 'Ystd']]
table = table.round({'Ymean': 3, 'Ystd': 4})
for column in ['x1', 'x2', 'x3', 'x4']:
  table[column] = table[column].astype(np.int64)
style = table.style
style = style.format(subset='Ymean', precision=3)
style = style.format(subset='Ystd', precision=4)
print(style.to_latex(hrules=True))
mistat.mainEffectsPlot(result, 'Ymean', factors=['x1', 'x2', 'x3', 'x4'],
                       height=3.75, aspect=0.5)

formula = ('Ymean ~ (x1+x2+x3+x4)**2 + ' +
           'I(x1**2) + I(x2**2) + I(x3**2) + I(x4**2)')
model = smf.ols(formula, data=result).fit()
print(model.summary2())

formula = ('Ymean ~ x1 + x2 + x3 + x1:x3 + x1:x2 + I(x1**2)')
reduced_model = smf.ols(formula, data=result).fit()
print(reduced_model.summary2())

def plotResponseSurface(model, center, unit, x1=None, x2=None,
                        x3=0, x4=0, ncontours=20):
    # predict in code co-ordinates
    x1 = x1 or (-2, 4)
    x2 = x2 or (-2, 2)
    x1 = np.linspace(*x1)
    x2 = np.linspace(*x2)
    X1, X2 = np.meshgrid(x1, x2)
    exog = pd.DataFrame({'x1': X1.ravel(), 'x2': X2.ravel(), 'x3': x3, 'x4': x4})
    responses = model.predict(exog=exog)

    # display in factor co-ordinates
    svalues = toFactor('x1', x1)
    v0values = toFactor('x2', x2)
    CS = plt.contour(svalues, v0values,
                responses.values.reshape(len(x2), len(x1)),
                ncontours, colors='gray')
    ax = plt.gca()
    ax.clabel(CS, inline=True, fontsize=10)
    ax.set_xlabel('s')
    ax.set_ylabel('v0')
    return ax

plotResponseSurface(model, center, unit)
plt.show()

### Approaching the Region of the Optimal Yield
# calculate gradient and create steps along gradient to descend
codes = ['x1', 'x2', 'x3', 'x4']
rsm = mistat.ResponseSurfaceMethod(model, codes)
distance = np.arange(0, 2.6, 0.5)
direction = {code: np.zeros(len(distance)) for code in codes}
x = pd.Series(np.zeros(4), index=codes)
for step in range(len(distance)):
    for code in codes:
        direction[code][step] = x[code]
    gradient = rsm.gradient(x)
    x = x - 0.5 * gradient

path = rsm.constrainedOptimization((0, 0, 0, 0), distances=(0.5, 1, 1.5, 2, 2.5), 
                                   maximize=False)
steps = pd.DataFrame({x2factor[code]: toFactor(code, path[code]) 
                      for code in rsm.codes})

# convert direction from code co-ordinates to factor co-ordinates
factor_direction = {x2factor[code]: toFactor(code, d)
                    for code, d in direction.items()}

# predict along the steps in direction and collect results
result = pd.DataFrame({
    'Distance': distance,
    **direction,
    **factor_direction,
    'yhat': model.get_prediction(exog=direction).predicted_mean,
})

ax = plotResponseSurface(model, center, unit, x1=(-2, 4), x2=(-3, 2))
ax.plot(result['s'], result['v0'], '-o',
        color='black', markerfacecolor='gray', markeredgecolor='gray')
ax.plot(steps['s'], steps['v0'], ':s',
        color='black', markerfacecolor='gray', markeredgecolor='gray')
ax.plot(center['s'], center['v0'], 'o', color='black')

plt.show()

table = result[['Distance', 'x1', 'x2', 'x3', 'x4', 's', 'v0', 'k', 't0', 'yhat']]

table.round(3)

for column in ['k', 't0']:
  table[column] = table[column].astype(np.int64)

style = table.style.hide(axis='index')
style = style.format(subset=['Distance'], precision=1)
style = style.format(subset=['s', 'v0'], precision=4)
style = style.format(subset=['yhat'], precision=3)
style = style.format(subset=['k', 't0'], precision=0)
style = style.format(subset=['x1', 'x2', 'x3', 'x4'], precision=2)
print(style.to_latex(hrules=True, column_format='cccccccccc'))
### Canonical Representation
import matplotlib.cm as cm

def surface1(x, y):
  return 83.57 + 9.39*x + 7.12*y - 7.44*x**2 - 3.71*y**2 - 5.80*x*y

def surface2(x, y):
  return 84.29 + 11.06*x + 4.05*y - 6.46*x**2 - 0.43*y**2 - 9.38*x*y

def surface3(x, y):
  return 83.93 + 10.23*x + 5.59*y - 6.95*x**2 - 2.07*y**2 - 7.59*x*y

def surface4(x, y):
  return 82.71 + 8.80*x + 8.19*y - 6.95*x**2 - 2.07*y**2 - 7.59*x*y

delta = 0.025
x = np.arange(-5.0, 5.0, delta)
y = np.arange(-6.0, 7.0, delta)
X, Y = np.meshgrid(x, y)
Z = surface2(X, Y)

plt.rcParams['contour.negative_linestyle'] = 'solid'
fig, axes = plt.subplots(figsize=(6, 6), ncols=4) #, nrows=2)
# axes = [*axes[0], *axes[1]]

for ax, surface in zip(axes, [surface1, surface2, surface3, surface4]):
    Z = surface(X, Y)
    im = ax.imshow(Z, interpolation='bilinear', origin='lower',
               cmap=cm.gray, extent=(min(x), max(x), min(y), max(y)))
    CS = ax.contour(X, Y, Z, 30, colors='black', linewidths=0.5)
    ax.axis('off')
axes[0].set_title('Simple maximum')
axes[1].set_title('Minimax')
axes[2].set_title('Stationary ridge')
axes[3].set_title('Rising ridge')
plt.tight_layout()
plt.show()

codes = ['x1', 'x2', 'x3', 'x4']
rsm = mistat.ResponseSurfaceMethod(model, codes)
stationary = rsm.stationary_point()

factor_stationary = pd.Series({x2factor[code]: toFactor(code, d)
                     for code, d in stationary.items()})
factor_stationary

path = rsm.constrainedOptimization(rsm.stationary_point(), maximize=False, 
                                   reverse=True)
steps = pd.DataFrame({x2factor[code]: toFactor(code, path[code]) 
                      for code in rsm.codes})
steps.head()

ax = plotResponseSurface(model, center, unit,
                         x1=(-2, 4), x2=(-3, 2),
                         x3=stationary['x3'], x4=stationary['x4'])
ax.plot(factor_stationary['s'], factor_stationary['v0'], 'o', color='black')
ax.plot(steps['s'], steps['v0'], '-o',
        color='black', markerfacecolor='gray', markeredgecolor='gray')
plt.show()

## Evaluating Designed Experiments
design35 = mistat.load_data('CUSTOMDESIGN_35')
design80 = mistat.load_data('CUSTOMDESIGN_80')
design169 = mistat.load_data('CUSTOMDESIGN_169')

design35

table = design35
style = table.style.hide(axis='index')
style = style.format(subset=['s', 'v0'], precision=3)
style = style.format(subset=['p0'], precision=4)
print(style.to_latex(hrules=True))
def plotCorrelation(design, mod=0, ax=None):
    mm = mistat.getModelMatrix(design, mod=mod)
    mm = mm.drop(columns='Intercept')
    corr = mm.corr().abs()
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(11, 7)
    sns.heatmap(corr, cmap='binary', ax=ax, square=True)
    return ax

fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
plotCorrelation(design35, mod=2, ax=axes[0])
plotCorrelation(design80, mod=2, ax=axes[1])
plotCorrelation(design169, mod=2, ax=axes[2])
plt.tight_layout()
plt.show()

ax = None
ax = mistat.FDS_Plot(design35, ax=ax, label='Design 35',
    plotkw={'linestyle': '-'})
ax = mistat.FDS_Plot(design80, ax=ax, label='Design 80',
    plotkw={'linestyle': '-.'})
ax = mistat.FDS_Plot(design169, maxscale=np.power(2, 7/4),
    ax=ax, label='Design 169', plotkw={'linestyle': '--'})
plt.show()

## Chapter Highlights
## Exercises
