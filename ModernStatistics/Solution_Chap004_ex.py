## Exercise chapter 4
#
# Modern Statistics: A Computer Based Approach with Python<br>
# by Ron Kenett, Shelemyahu Zacks, Peter Gedeck
# 
# Publisher: Springer International Publishing; 1st edition (September 15, 2022) <br>
# ISBN-13: 978-3031075650
# 
# (c) 2022 Ron Kenett, Shelemyahu Zacks, Peter Gedeck
# 
# The code needs to be executed in sequence.


import warnings
from outdated import OutdatedPackageWarning
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=OutdatedPackageWarning)

import random
import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats as sms
import seaborn as sns
import matplotlib.pyplot as plt

import mistat


################################################################################
# Exercise 1
################################################################################
car = mistat.load_data('CAR')
sns.pairplot(car[['turn', 'hp', 'mpg']])
plt.show()


################################################################################
# Exercise 2
################################################################################
car = mistat.load_data('CAR')

ax = car.boxplot(column='turn', by='origin')
ax.set_title('')
ax.get_figure().suptitle('')
ax.set_xlabel('origin')
plt.show()


################################################################################
# Exercise 3
################################################################################
hadpas = mistat.load_data('HADPAS')
ax = hadpas.boxplot(column='res3', by='hyb')
ax.set_title('')
ax.get_figure().suptitle('')
ax.set_xlabel('Hybrid number')
ax.set_ylabel('Res 3')
plt.show()

sns.pairplot(hadpas[['res3', 'res7', 'res18', 'res14', 'res20']])
plt.show()


################################################################################
# Exercise 4
################################################################################
car = mistat.load_data('CAR')
binned_car = pd.DataFrame({
  'hp': pd.cut(car['hp'], bins=np.arange(50, 275, 25)),
  'mpg': pd.cut(car['mpg'], bins=np.arange(10, 40, 5)),
})
freqDist = pd.crosstab(binned_car['hp'], binned_car['mpg'])
print(freqDist)
# You can get distributions for hp and mpg by summing along an axis
print(freqDist.sum(axis=0))
print(freqDist.sum(axis=1))


################################################################################
# Exercise 5
################################################################################
hadpas = mistat.load_data('HADPAS')
binned_hadpas = pd.DataFrame({
  'res3': pd.cut(hadpas['res3'], bins=np.arange(1580, 2780, 200)),
  'res14': pd.cut(hadpas['res14'], bins=np.arange(900, 3000, 300)),
})
pd.crosstab(binned_hadpas['res14'], binned_hadpas['res3'])


################################################################################
# Exercise 6
################################################################################
hadpas = mistat.load_data('HADPAS')
in_range = hadpas[hadpas['res14'].between(1300, 1500)]
pd.cut(in_range['res3'], bins=np.arange(1580, 2780, 200)).value_counts(sort=False)


################################################################################
# Exercise 7
################################################################################
hadpas = mistat.load_data('HADPAS')
bins = [900, 1200, 1500, 1800, 2100, 3000]
binned_res14 = pd.cut(hadpas['res14'], bins=bins)

results = []
for group, df in hadpas.groupby(binned_res14):
  res3 = df['res3']
  results.append({
    'res3': group,
    'N': len(res3),
    'mean': res3.mean(),
    'std': res3.std(),
  })
pd.DataFrame(results)


################################################################################
# Exercise 8
################################################################################
df = pd.DataFrame([
  [10.0, 8.04, 10.0, 9.14, 10.0, 7.46, 8.0, 6.58],
  [8.0, 6.95, 8.0, 8.14, 8.0, 6.77, 8.0, 5.76],
  [13.0, 7.58, 13.0, 8.74, 13.0, 12.74, 8.0, 7.71],
  [9.0, 8.81, 9.0, 8.77, 9.0, 7.11, 8.0, 8.84],
  [11.0, 8.33, 11.0, 9.26, 11.0, 7.81, 8.0, 8.47],
  [14.0, 9.96, 14.0, 8.10, 14.0, 8.84, 8.0, 7.04],
  [6.0, 7.24, 6.0, 6.13, 6.0, 6.08, 8.0, 5.25],
  [4.0, 4.26, 4.0, 3.10, 4.0, 5.39, 19.0, 12.50],
  [12.0, 10.84, 12.0, 9.13, 12.0, 8.15, 8.0, 5.56],
  [7.0, 4.82, 7.0, 7.26, 7.0, 6.42, 8.0, 7.91],
  [5.0, 5.68, 5.0, 4.74, 5.0, 5.73, 8.0, 6.89],
], columns=['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'])

results = []
for i in (1, 2, 3, 4):
  x = df[f'x{i}']
  y = df[f'y{i}']
  model = smf.ols(formula=f'y{i} ~ 1 + x{i}', data=df).fit()
  results.append({
    'Data Set': i,
    'Intercept': model.params['Intercept'],
    'Slope': model.params[f'x{i}'],
    'R2': model.rsquared,
  })
pd.DataFrame(results)

fig, axes = plt.subplots(figsize=[4, 4], ncols=2, nrows=2)
for i in range(4):
  ax = axes[i // 2, i % 2]
  df.plot.scatter(x=f'x{i+1}', y=f'y{i + 1}', ax=ax)
  ax.plot([0, 20], [3, 13], color='C1')
plt.tight_layout()
plt.show()


################################################################################
# Exercise 9
################################################################################
car = mistat.load_data('CAR')
car[['turn', 'hp', 'mpg']].corr()



################################################################################
# Exercise 11
################################################################################
car = mistat.load_data('CAR')

model = smf.ols(formula='mpg ~ 1 + hp + turn', data=car).fit()
print(model.summary2())


################################################################################
# Exercise 12
################################################################################
car = mistat.load_data('CAR')

# y: mpg, x1: cyl, x2: hp
model_1 = smf.ols(formula='mpg ~ cyl + 1', data=car).fit()
e_1 = model_1.resid

model_2 = smf.ols(formula='hp ~ cyl + 1', data=car).fit()
e_2 = model_2.resid

print(f'Partial correlation {stats.pearsonr(e_1, e_2)[0]:.5f}')


################################################################################
# Exercise 13
################################################################################
car = mistat.load_data('CAR')

# y: mpg, x1: hp, x2: turn
model_1 = smf.ols(formula='mpg ~ hp + 1', data=car).fit()
e_1 = model_1.resid
print('Model mpg ~ hp + 1:\n', model_1.params)

model_2 = smf.ols(formula='turn ~ hp + 1', data=car).fit()
e_2 = model_2.resid
print('Model turn ~ hp + 1:\n', model_2.params)

print('Partial correlation', stats.pearsonr(e_1, e_2)[0])
df = pd.DataFrame({'e1': e_1, 'e2': e_2})
model_partial = smf.ols(formula='e1 ~ e2 - 1', data=df).fit()
# print(model_partial.summary2())
print('Model e1 ~ e2:\n', model_partial.params)



################################################################################
# Exercise 15
################################################################################
almpin = mistat.load_data('ALMPIN')
model = smf.ols('capDiam ~ 1 + diam2 + diam3', data=almpin).fit()
model.summary2()


################################################################################
# Exercise 16
################################################################################
gasol = mistat.load_data('GASOL')
# rename column 'yield' to 'Yield' as 'yield' is a special keyword in Python
gasol = gasol.rename(columns={'yield': 'Yield'})
model = smf.ols(formula='Yield ~ x1 + x2 + astm + endPt',
                data=gasol).fit()
print(model.summary2())

model = smf.ols(formula='Yield ~ x1 + astm + endPt', data=gasol).fit()
print(model.summary2())

fig, ax = plt.subplots(figsize=[4, 4])
pg.qqplot(model.resid, ax=ax)
plt.show()








################################################################################
# Exercise 23
################################################################################
socell = mistat.load_data('SOCELL')

# combine the two datasets and add the additional columns z and w
socell_1 = socell[['t3', 't1']].copy()
socell_1.columns = ['t3', 't']
socell_1['z'] = 0
socell_2 = socell[['t3', 't2']].copy()
socell_2.columns = ['t3', 't']
socell_2['z'] = 1
combined = pd.concat([socell_1, socell_2])
combined['w'] = combined['z'] * combined['t']

# multiple linear regression model
model_test  = smf.ols(formula='t3 ~ t + z + w + 1',
                      data=combined).fit()
print(model_test.summary2())

model_combined  = smf.ols(formula='t3 ~ t + 1',
                          data=combined).fit()
print(model_combined.summary2())


################################################################################
# Exercise 24
################################################################################
df = mistat.load_data('CEMENT.csv')

# ignore UserWarning for Kurtosis-test due to small dataset
import warnings
warnings.simplefilter('ignore', category=UserWarning)

model1 = smf.ols('y ~ x1 + 1', data=df).fit()
print(model1.summary().tables[1])
r2 = model1.rsquared
print(f'R-sq: {r2:.3f}')

anova = sms.anova.anova_lm(model1)
print('Analysis of Variance\n', anova)

F = anova.F['x1']
SSE_1 = anova.sum_sq['Residual']

model2 = smf.ols('y ~ x1 + x2 + 1', data=df).fit()
r2 = model2.rsquared
print(model2.summary().tables[1])
print(f'R-sq: {r2:.3f}')
anova = sms.anova.anova_lm(model2)
print('Analysis of Variance\n', anova)
SEQ_SS_X2 = anova.sum_sq['x2']
SSE_2 = anova.sum_sq['Residual']
s2e2 = anova.mean_sq['Residual']
partialF = np.sum(anova.sum_sq) * (model2.rsquared - model1.rsquared) / s2e2

anova = sms.anova.anova_lm(model1, model2)
print('Comparing models\n', anova)
partialF = anova.F[1]

model3 = smf.ols('y ~ x1 + x2 + x3 + 1', data=df).fit()
r2 = model3.rsquared
print(model3.summary().tables[1])
print(f'R-sq: {r2:.3f}')
anova = sms.anova.anova_lm(model3)
print('Analysis of Variance\n', anova)
SEQ_SS_X3 = anova.sum_sq['x3']
SSE_3 = anova.sum_sq['Residual']
s2e3 = anova.mean_sq['Residual']

anova = sms.anova.anova_lm(model2, model3)
print('Comparing models\n', anova)
partialF = anova.F[1]

model4 = smf.ols('y ~ x1 + x2 + x3 + x4 + 1', data=df).fit()
r2 = model4.rsquared
print(model4.summary().tables[1])
print(f'R-sq: {r2:.3f}')
anova = sms.anova.anova_lm(model4)
print('Analysis of Variance\n', anova)
SEQ_SS_X4 = anova.sum_sq['x4']
SSE_4 = anova.sum_sq['Residual']
s2e4 = anova.mean_sq['Residual']

anova = sms.anova.anova_lm(model3, model4)
print('Comparing models\n', anova)
partialF = anova.F[1]

# restore default setting
warnings.simplefilter('default', category=UserWarning)


################################################################################
# Exercise 25
################################################################################
outcome = 'y'
all_vars = ['x1', 'x2', 'x3', 'x4']

included, model = mistat.stepwise_regression(outcome, all_vars, df)

formula = ' + '.join(included)
formula = f'{outcome} ~ 1 + {formula}'
print()
print('Final model')
print(formula)
print(model.params)


################################################################################
# Exercise 26
################################################################################
car = mistat.load_data('CAR')
car_3 = car[car['origin'] == 3]
print('Full dataset shape', car.shape)
print('Origin 3 dataset shape', car_3.shape)
model = smf.ols(formula='mpg ~ hp + 1', data=car_3).fit()
print(model.summary2())

influence = model.get_influence()
df = pd.DataFrame({
  'hp': car_3['hp'],
  'mpg': car_3['mpg'],
  'resi': model.resid,
  'sres': influence.resid_studentized_internal,
  'hi': influence.hat_matrix_diag,
  'D': influence.cooks_distance[0],
})
print(df.round(4))


################################################################################
# Exercise 27
################################################################################
np.random.seed(1)
settings = {'s': 0.005, 'v0': 0.002, 'k': 1000, 'p0': 90_000,
            't': 290, 't0': 340}
results = []
n_simulation = 5
for m in [30, 40, 50, 60]:
  simulator = mistat.PistonSimulator(m=m, n_simulation=n_simulation,
                                     **settings)
  sim_result = simulator.simulate()
  results.extend([m, s] for s in sim_result['seconds'])
results = pd.DataFrame(results, columns=['m', 'seconds'])

group_std = results.groupby('m').std()
pooled_std = np.sqrt(np.sum(group_std**2) / len(group_std))[0]
print('Pooled standard deviation', pooled_std)

group_mean = results.groupby('m').mean()
ax = results.plot.scatter(x='m', y='seconds', color='black')
ax.errorbar(group_mean.index, results.groupby('m').mean().values.flatten(),
            yerr=[pooled_std] * 4, color='grey')
plt.show()

model = smf.ols(formula='seconds ~ C(m)', data=results).fit()
aov_table = sm.stats.anova_lm(model)
aov_table


################################################################################
# Exercise 28
################################################################################
df = pd.DataFrame([
  [2.58, 2.62, 2.22],
  [2.48, 2.77, 1.73],
  [2.52, 2.69, 2.00],
  [2.50, 2.80, 1.86],
  [2.53, 2.87, 2.04],
  [2.46, 2.67, 2.15],
  [2.52, 2.71, 2.18],
  [2.49, 2.77, 1.86],
  [2.58, 2.87, 1.84],
  [2.51, 2.97, 1.86]
], columns=['Exp. 1', 'Exp. 2', 'Exp. 3'])
df.boxplot()

# Convert data frame to long format using melt
df = df.melt(var_name='Experiment', value_name='mu')

model = smf.ols(formula='mu ~ C(Experiment)', data=df).fit()
aov_table = sm.stats.anova_lm(model)
aov_table

experiment = df['Experiment']
mu = df['mu']
def onewayTest(x, verbose=False):
    df = pd.DataFrame({
        'value': x,
        'variable': experiment,
    })
    aov = pg.anova(dv='value', between='variable', data=df)
    return aov['F'].values[0]

B = pg.compute_bootci(mu, func=onewayTest, n_boot=1000,
    seed=1, return_dist=True)

Bt0 = onewayTest(mu)
print('Bt0', Bt0)
print('ratio', sum(B[1] >= Bt0)/len(B[1]))


################################################################################
# Exercise 29
################################################################################
df = pd.DataFrame({
  'Batch A': [103, 107, 104, 102, 95, 91, 107, 99, 105, 105],
  'Batch B': [104, 103, 106, 103, 107, 108, 104, 105, 105, 97],
})
df.boxplot()
plt.show()

dist = mistat.randomizationTest(df['Batch A'], df['Batch B'], np.mean,
                                aggregate_stats=lambda x: x[0] - x[1],
                                n_boot=10000, seed=1)
# ax = sns.distplot(dist)
# ax.axvline(np.mean(df['Batch A']) - np.mean(df['Batch B']))

# Convert data frame to long format using melt
df = df.melt(var_name='Batch', value_name='film_speed')

model = smf.ols(formula='film_speed ~ C(Batch)', data=df).fit()
aov_table = sm.stats.anova_lm(model)
aov_table


################################################################################
# Exercise 30
################################################################################
df = pd.DataFrame([
  [2.58, 2.62, 2.22],
  [2.48, 2.77, 1.73],
  [2.52, 2.69, 2.00],
  [2.50, 2.80, 1.86],
  [2.53, 2.87, 2.04],
  [2.46, 2.67, 2.15],
  [2.52, 2.71, 2.18],
  [2.49, 2.77, 1.86],
  [2.58, 2.87, 1.84],
  [2.51, 2.97, 1.86]
], columns=['Exp. 1', 'Exp. 2', 'Exp. 3'])

# Convert data frame to long format using melt
df = df.melt(var_name='Experiment', value_name='mu')

def func_stats(x):
    m = pd.Series(x).groupby(df['Experiment']).agg(['mean', 'count'])
    top = np.sum(m['count'] * m['mean'] ** 2) - len(x)*np.mean(x)**2
    return top / np.std(x) ** 2

Bt = []
mu = list(df['mu'])
for _ in range(1000):
    mu_star = random.sample(mu, len(mu))
    Bt.append(func_stats(mu_star))

Bt0 = func_stats(mu)
print('Bt0', Bt0)
print('ratio', sum(Bt >= Bt0)/len(Bt))


################################################################################
# Exercise 31
################################################################################
place = mistat.load_data('PLACE')
place.boxplot('xDev', by='crcBrd')
plt.show()

model = smf.ols(formula='xDev ~ C(crcBrd)', data=place).fit()
aov_table = sm.stats.anova_lm(model)
aov_table

G1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
G2 = [10, 11, 12]
G3 = [13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26]
G4 = [20]
place['group'] = 'G1'
place.loc[place['crcBrd'].isin(G2), 'group'] = 'G2'
place.loc[place['crcBrd'].isin(G3), 'group'] = 'G3'
place.loc[place['crcBrd'].isin(G4), 'group'] = 'G4'

statistics = place['xDev'].groupby(place['group']).agg(['mean', 'sem', 'count'])
statistics = statistics.sort_values(['mean'], ascending=False)
print(statistics.round(8))
statistics['Diff'] = 0
n = len(statistics)
print(statistics['mean'][:-1].values - statistics['mean'][1:].values)
print(statistics['mean'][:(n-1)].values - statistics['mean'][1:].values)
statistics.loc[1:, 'Diff'] = (statistics['mean'][:-1].values -
                              statistics['mean'][1:].values)
statistics['CR'] = 6.193 * statistics['Diff']
print(statistics.round(8))
# 0.001510 0.0022614 0.0010683
# 0.000757 0.000467 0.000486

sem = statistics['sem'].values
sem = sem**2
sem = np.sqrt(sem[:-1] + sem[1:])
print(sem * 6.193)
print(757/644, 467/387, 486/459)

from statsmodels.stats.multicomp import pairwise_tukeyhsd
m_comp = pairwise_tukeyhsd(endog=place['xDev'], groups=place['group'],
                           alpha=0.05)
print(m_comp)


################################################################################
# Exercise 32
################################################################################
df = pd.DataFrame({
    'US': [33, 25],
    'Europe': [7, 7],
    'Asia': [26, 11],
  })

print(df)

col_sums = df.sum(axis=0)
row_sums = df.sum(axis=1)
total = df.to_numpy().sum()

expected_frequencies = np.outer(row_sums, col_sums) / total

chi2 = (df - expected_frequencies) ** 2 / expected_frequencies
chi2 = chi2.to_numpy().sum()
print(f'chi2: {chi2:.3f}')
print(f'p-value: {1 - stats.chi2.cdf(chi2, 2):.3f}')

chi2 = stats.chi2_contingency(df)
print(f'chi2-statistic: {chi2[0]:.3f}')
print(f'p-value: {chi2[1]:.3f}')
print(f'd.f.: {chi2[2]}')


################################################################################
# Exercise 33
################################################################################
car = mistat.load_data('CAR')
binned_car = pd.DataFrame({
  'turn': pd.cut(car['turn'], bins=[27, 30.6, 34.2, 37.8, 45]), #np.arange(27, 50, 3.6)),
  'mpg': pd.cut(car['mpg'], bins=[12, 18, 24, 100]),
})
freqDist = pd.crosstab(binned_car['mpg'], binned_car['turn'])
print(freqDist)

chi2 = stats.chi2_contingency(freqDist)
print(f'chi2-statistic: {chi2[0]:.3f}')
print(f'p-value: {chi2[1]:.3f}')
print(f'd.f.: {chi2[2]}')


################################################################################
# Exercise 34
################################################################################
question_13 = pd.DataFrame({
  '1': [0,0,0,1,0],
  '2': [1,0,2,0,0],
  '3': [1,2,6,5,1],
  '4': [2,1,10,23,13],
  '5': [0,1,1,15,100],
  }, index = ['1', '2', '3', '4', '5']).transpose()
question_23 = pd.DataFrame({
  '1': [1,0,0,3,1],
  '2': [2,0,1,0,0],
  '3': [0,4,2,3,0],
  '4': [1,1,10,7,5],
  '5': [0,0,1,30,134],
  }, index = ['1', '2', '3', '4', '5']).transpose()

chi2_13 = stats.chi2_contingency(question_13)
chi2_23 = stats.chi2_contingency(question_23)

msc_13 = chi2_13[0] / question_13.to_numpy().sum()
tschuprov_13 = np.sqrt(msc_13 / (2 * 2)) # (4 * 4))
cramer_13 = np.sqrt(msc_13 / 2) # min(4, 4))

msc_23 = chi2_23[0] / question_23.to_numpy().sum()
tschuprov_23 = np.sqrt(msc_23 / 4) # (4 * 4))
cramer_23 = np.sqrt(msc_23 / 2) # min(4, 4))

print('Question 1 vs 3')
print(f'  Mean squared contingency : {msc_13:.3f}')
print(f'  Tschuprov : {tschuprov_13:.3f}')
print(f"  Cramer's index : {cramer_13:.3f}")
print('Question 2 vs 3')
print(f'  Mean squared contingency : {msc_23:.3f}')
print(f'  Tschuprov : {tschuprov_23:.3f}')
print(f"  Cramer's index : {cramer_23:.3f}")

