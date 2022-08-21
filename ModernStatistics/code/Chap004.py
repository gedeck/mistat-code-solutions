## Chapter 4
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
import os
os.environ['OUTDATED_IGNORE'] = '1'
import warnings
from outdated import OutdatedPackageWarning
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=OutdatedPackageWarning)

# Variability in Several Dimensions and Regression Models
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats as sms
from statsmodels.graphics.mosaicplot import mosaic
import seaborn as sns
import matplotlib.pyplot as plt

import mistat

place = mistat.load_data('PLACE')
ax = place.plot.scatter(x='xDev', y='yDev', marker = "$\u25ef$",
                        color='black')
ax.set_xlim(-0.0035, 0.005)
ax.set_ylim(-0.004, 0.0035)
plt.show()

## Graphical Display and Analysis
### Scatterplots
# The following command would be sufficient to create the scatterplot matrix
# matplotlib has however a problem with scaling xDev
sns.pairplot(place[['xDev', 'yDev', 'tDev']], markers=".",
             plot_kws={'facecolors': 'none', 'edgecolor': 'black'},
             diag_kws={'color': 'grey'})

#def panelPlot(x, y, **kwargs):
#    plt.scatter(x, y, **kwargs,
#                facecolors='none', edgecolor='black', s=20)
#    dx = 0.05*(max(x) - min(x))
#    plt.xlim(min(x)-dx, max(x) + dx)
#    dy = 0.05*(max(y) - min(y))
#    plt.ylim(min(y)-dy, max(y) + dy)
#g = sns.PairGrid(place[['xDev', 'yDev', 'tDev']])
#g = g.map_offdiag(panelPlot)
plt.show()

ax = sns.histplot(place['tDev'], kde=True, color='black')

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

fig = plt.figure(figsize=[6, 5])
ax = fig.add_subplot(111, projection='3d')

ax.scatter(place['tDev'], place['xDev'], place['yDev'], color='black')
ax.set_xlabel('tDev')
ax.set_ylabel('yDev')
ax.set_zlabel('zDev')

ax.set_yticks([-0.002, 0, 0.002, 0.004])
ax.yaxis.labelpad = 15
ax.zaxis.labelpad = 20
ax.tick_params(axis='z', which='major', pad=10)
ax.dist = 13
plt.show()

### Multiple Box-Plots
fig, ax = plt.subplots(figsize=(8, 5))
place.boxplot(column='xDev', by='crcBrd', color='black', ax=ax)
ax.set_title('')
ax.get_figure().suptitle('')
ax.set_xlabel('Board number')
ax.set_ylabel('xDev')
plt.show()


place['code'] = [*['lDev'] * 9 * 16,
                 *['mDev'] * 3 * 16,
                 *['hDev'] * 14 * 16, ]
fig, ax = plt.subplots(figsize=(6, 4))
marker = {'lDev': '^', 'mDev': '+', 'hDev': 'o'}
label = {'lDev': 'Code 1', 'mDev': 'Code 2', 'hDev': 'Code 3'}
common = {'x': 'xDev', 'y': 'yDev',
          'linestyle': 'none', 'fillstyle': 'none', 'color': 'black'}
for code in ('lDev', 'mDev', 'hDev'):
    subset = place[place['code'] == code]
    subset.plot(marker=marker[code], ax=ax, label=label[code], **common)
ax.set_xlabel('xDev')
ax.set_ylabel('yDev')
plt.show()

## Frequency Distributions in Several Dimensions
### Bivariate Joint Frequency Distributions
almpin = mistat.load_data('ALMPIN')
binned_almpin = pd.DataFrame({
  'lenWcp': pd.cut(almpin['lenWcp'], bins=np.arange(59.9, 60.2, 0.1)),
  'lenNocp': pd.cut(almpin['lenNocp'], bins=np.arange(49.8, 50.1, 0.1)),
})
join_frequencies = pd.crosstab(binned_almpin['lenNocp'],
                               binned_almpin['lenWcp'])
print(join_frequencies)

print('Row Totals', join_frequencies.sum(axis=1))
print('Column Totals', join_frequencies.sum(axis=0))

mosaic(binned_almpin, ['lenWcp', 'lenNocp'], labelizer=lambda x: '')
plt.show()

hadpas = mistat.load_data('HADPAS')
hadpas['res7fac'] = pd.cut(hadpas['res7'], bins=range(1300, 2500, 200))
hadpas['res3'].groupby(hadpas['res7fac']).agg(['count', 'mean', 'std'])

ax = hadpas.boxplot(column='res3', by='res7fac',
         color={'boxes':'grey', 'medians':'black', 'whiskers':'black'},
         patch_artist=True)
ax.set_title('')
ax.get_figure().suptitle('')
ax.set_xlabel('res7fac')
plt.show()

### Conditional Distributions
hadpas = mistat.load_data('HADPAS')
binned_hadpas = pd.DataFrame({
  'res3': pd.cut(hadpas['res3'], bins=np.arange(1500, 2700, 200)),
  'res7': pd.cut(hadpas['res7'], bins=np.arange(1300, 2500, 200)),
})
res3_res7 = pd.crosstab(binned_hadpas['res3'], binned_hadpas['res7'])
cond_dist = 100 * res3_res7 / res3_res7.sum(axis=0)

## Correlation and Regression Analysis
### Covariances and Correlations
almpin = mistat.load_data('ALMPIN') 
sns.pairplot(almpin, plot_kws={'color': 'black'}, diag_kws={'color': 'grey'}, height=1.4)
plt.show()

### Fitting Simple Regression Lines to Data
#### The Least Squares Method
socell = mistat.load_data('SOCELL')
socell.plot.scatter(x='t1', y='t2', color='black')
plt.show()

# ignore UserWarning for Kurtosis-test due to small dataset
import warnings
warnings.simplefilter('ignore', category=UserWarning)

socell = mistat.load_data('SOCELL')
model = smf.ols(formula='t2 ~ 1 + t1', data=socell).fit()
print(model.summary2())

# restore default setting
warnings.simplefilter('default', category=UserWarning)

sns.residplot(x=model.predict(socell), y=socell['t2'], lowess=False, color='black')
plt.show()

#### Regression and Prediction Intervals
result = model.get_prediction(pd.DataFrame({'t1': [4.0,4.4,4.8,5.2]}))
columns = ['mean', 'obs_ci_lower', 'obs_ci_upper']
print(0.01)
print(result.summary_frame(alpha=0.01)[columns].round(3))
print(0.05)
print(result.summary_frame(alpha=0.05)[columns].round(3))

result = model.get_prediction(pd.DataFrame({'t1': [4.0,4.4,4.8,5.2]}))
print(str(result.summary_frame(alpha=0.01)).replace('\\', '\\\\'))

hadpas = mistat.load_data('HADPAS')
ax = hadpas.plot.scatter(x='res7', y='res3', color='darkgrey')

model = smf.ols(formula='res3 ~ 1 + res7', data=hadpas).fit()
sm.graphics.abline_plot(model_results=model, ax=ax, color='black')

newdata = pd.DataFrame({'res7': np.linspace(1300, 2300, 200)})
predictions = model.get_prediction(newdata)
predIntervals = predictions.summary_frame(alpha=0.05)
ax.plot(newdata['res7'], predIntervals['obs_ci_upper'], color='grey', linestyle='--')
ax.plot(newdata['res7'], predIntervals['obs_ci_lower'], color='grey', linestyle='--')
predIntervals = predictions.summary_frame(alpha=0.01)
ax.plot(newdata['res7'], predIntervals['obs_ci_upper'], color='red', linestyle='--')
ax.plot(newdata['res7'], predIntervals['obs_ci_lower'], color='red', linestyle='--')
plt.show()

## Multiple Regression
### Regression on Two Variables
gasol = mistat.load_data('GASOL')
# rename column 'yield' to 'Yield' as 'yield' is a special keyword in Python
gasol = gasol.rename(columns={'yield': 'Yield'})
model = smf.ols(formula='Yield ~ astm + endPt + 1', data=gasol).fit()
print(model.summary2())

# Covariance
gasol[['astm', 'endPt', 'Yield']].cov()
# Means
gasol[['astm', 'endPt', 'Yield']].mean()

ax = sns.residplot(x=model.predict(gasol), y=gasol['Yield'], lowess=False, color='black')
ax.set_xlabel('Fitted value')
ax.set_ylabel('Residuals')
plt.show()

### Partial Regression and Correlation
stage1 = smf.ols(formula='Yield ~ 1 + astm', data=gasol).fit()
print(stage1.params)
print('R2(y, astm)', stage1.rsquared)

stage2 = smf.ols(formula='endPt ~ 1 + astm', data=gasol).fit()
print(stage2.params)
print('R2(endPt, astm)', stage2.rsquared)

residuals = pd.DataFrame({
  'e1': stage1.resid,
  'e2': stage2.resid,
})
print(np.corrcoef(stage1.resid, stage2.resid))

# use -1 in the formula to fix intercept to 0
stage3 = smf.ols(formula='e1 ~ e2 - 1', data=residuals).fit()
print(stage3.params)
print('R2(e1, e2)', stage3.rsquared)

plt.scatter(stage2.resid, stage1.resid, color='grey')
ax = plt.gca()
ax.set_xlabel(r'$e_2$')
ax.set_ylabel(r'$e_1$')
xlim = np.array(ax.get_xlim())
ax.plot(xlim, stage3.params[0] * xlim, color='black')
plt.show()

### Multiple Linear Regression
almpin = mistat.load_data('ALMPIN')
# create the X matrix
X = almpin[['diam1', 'diam2', 'diam3']]
X = np.hstack((np.ones((len(X), 1)), X))
# calculate the inverse of XtX
np.linalg.inv(np.matmul(X.transpose(), X))

almpin = mistat.load_data('ALMPIN')
model = smf.ols('capDiam ~ 1 + diam1 + diam2 + diam3', data=almpin).fit()
print(model.summary2())
print()
print(sms.anova.anova_lm(model))

ax = sns.residplot(x=model.predict(almpin), y=almpin['capDiam'], lowess=False,
                   color='grey', line_kws={'color': 'black', 'linestyle': '--'})
ax.set_xlabel('Fitted value')
ax.set_ylabel('Residuals')
plt.show()

# load dataset and split into data for US and Asia
car = mistat.load_data('CAR.csv')
car_US = car[car['origin'] == 1].copy()
car_Asia = car[car['origin'] == 3].copy()
# add the indicator variable z
car_US['z'] = 0
car_Asia['z'] = 1
# combine datasets and add variable w
car_combined = pd.concat([car_US, car_Asia])
car_combined['w'] = car_combined['z'] * car_combined['turn']

model_US = smf.ols('mpg ~ 1 + turn', data=car_US).fit()
model_Asia = smf.ols('mpg ~ 1 + turn', data=car_Asia).fit()
model_combined = smf.ols('mpg ~ 1 + turn+z+w', data=car_combined).fit()
model_simple = smf.ols('mpg ~ 1 + turn', data=car_combined).fit()
print('US\n', model_US.params)
print('Europe\n', model_Asia.params)
print(model_combined.summary2())

# create visualization
ax = car_US.plot.scatter(x='turn', y='mpg', color='gray', marker='o')
car_Asia.plot.scatter(x='turn', y='mpg', ax=ax, color='gray', marker='^')

car_combined = car_combined.sort_values(['turn'])
ax.plot(car_combined['turn'], model_US.predict(car_combined),
        color='gray', linestyle='--')
ax.plot(car_combined['turn'], model_Asia.predict(car_combined),
        color='gray', linestyle=':')
ax.plot(car_combined['turn'], model_simple.predict(car_combined),
        color='black', linestyle='-')
plt.show()


### Partial $F$-Tests and The Sequential SS
import warnings
almpin = mistat.load_data('ALMPIN')
model3 = smf.ols('capDiam ~ 1 + diam1+diam2+diam3', data=almpin).fit()
model2 = smf.ols('capDiam ~ 1 + diam1+diam2', data=almpin).fit()
model1 = smf.ols('capDiam ~ 1 + diam1', data=almpin).fit()
model0 = smf.ols('capDiam ~ 1', data=almpin).fit()

print('Full model\n', sms.anova.anova_lm(model))
print(f'SSE: diam1: {model1.ssr:.6f}')
print(f'     diam2: {model2.ssr:.6f}')
print(f'     diam3: {model3.ssr:.6f}')

# we capture a few irrelevant warnings here -
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print('diam1:\n', sms.anova.anova_lm(model0, model1))
    print('diam2:\n', sms.anova.anova_lm(model1, model2))
    print('diam3:\n', sms.anova.anova_lm(model2, model3))

### Model Construction:  Step-Wise Regression
gasol = mistat.load_data('GASOL')
gasol = gasol.rename(columns={'yield': 'Yield'})

outcome = 'Yield'
all_vars = set(gasol.columns)
all_vars.remove(outcome)

include, model = mistat.stepwise_regression(outcome, all_vars, gasol)

formula = ' + '.join(include)
formula = f'{outcome} ~ 1 + {formula}'
print()
print('Final model')
print(formula)
print(model.params)

### Regression Diagnostics
# load data and create modified dataset
socell = mistat.load_data('SOCELL')
socell = socell.sort_values(['t1'])
socell1 = socell.copy()
t1_1 = socell1.loc[8, 't1']
t2_1 = socell1.loc[8, 't2']
socell1.loc[8, 't2'] = t2_1 - 1

socell2 = socell.copy()
t1_2 = socell2.loc[5, 't1']
t2_2 = socell2.loc[5, 't2']
socell2.loc[5, 't2'] = t2_2 + 1

model = smf.ols('t2 ~ 1 + t1', data=socell).fit()
model1 = smf.ols('t2 ~ 1 + t1', data=socell1).fit()
model2 = smf.ols('t2 ~ 1 + t1', data=socell2).fit()

ax = socell.plot.scatter(x='t1', y='t2', color='grey')
ax.plot(socell['t1'], model.predict(socell), color='black')
ax.plot(socell['t1'], model1.predict(socell), color='black', linestyle='--')
ax.plot(socell['t1'], model2.predict(socell), color='black', linestyle=':')

prop = {'arrowstyle': "->,head_width=0.4,head_length=0.8",
        'shrinkA': 0, 'shrinkB': 0, 'color': 'grey', 'linewidth': 1}
d = 0.05
ax.scatter([t1_1], [t2_1], color='black')
ax.scatter([t1_1], [t2_1-1], facecolors='none', edgecolors='black')
plt.annotate("", xytext=(t1_1, t2_1-d), xy=(t1_1, t2_1-1+d),
       arrowprops={'linestyle': '--', **prop})
ax.scatter([t1_2], [t2_2], color='black', marker='s')
ax.scatter([t1_2], [t2_2+1], facecolors='none', edgecolors='black', marker='s')
plt.annotate("", xytext=(t1_2, t2_2+d), xy=(t1_2, t2_2+1-d),
       arrowprops={'linestyle': ':', **prop})
plt.show()

socell = mistat.load_data('SOCELL')

model = smf.ols(formula='t2 ~ 1 + t1', data=socell).fit()
influence = model.get_influence()
# leverage: influence.hat_matrix_diag
# std. residuals: influence.resid_studentized
# Cook-s distance: influence.cooks_distance[0]
# DFIT: influence.dffits[0]

sm.graphics.influence_plot(model)
ax = plt.gca()
ax.set_ylim(-2.4, 2.8)
plt.show()

leverage = influence.hat_matrix_diag
print(f'average leverage: {np.mean(leverage):.3f}')
print(f'point #8: {leverage[8]:.3f}')
print(f'point #5: {leverage[5]:.3f}')

influence = model.get_influence()
plt.scatter(influence.hat_matrix_diag, influence.summary_frame()['cooks_d'])
ax = plt.gca()
ax.set_xlabel('Leverage $h_{ii}$')
ax.set_ylabel('Cook''s distance')
plt.show()

## Quantal Response Analysis:  Logistic Regression
## The Analysis of Variance:  The Comparison of Means
### The Statistical Model
### The One-Way Analysis of Variance (ANOVA)
vendor = mistat.load_data('VENDOR')
vendor_long = pd.melt(vendor, value_vars=vendor.columns)
vendor_long['value'] = np.sqrt(vendor_long['value'])

ax = vendor_long.boxplot(column='value', by='variable',
                    color={'boxes': 'grey', 'medians': 'black', 'whiskers': 'black'},
                    patch_artist=True)
ax.set_title('')
ax.get_figure().suptitle('')
model = smf.ols('value ~ -1 + variable', data=vendor_long).fit()
ci = model.conf_int()
print(ci)
err = 0.5 * (ci[1] - ci[0])
print(err)
#err = model.conf_int().transpose().values - model.params
# print(err)
ax.errorbar([1.2, 2.2, 3.2], model.params, yerr=err,
  fmt='o', color='darkgrey')
plt.show()

model = smf.ols('value ~ variable', data=vendor_long).fit()
table = sm.stats.anova_lm(model, typ=1)
print(table)

model = smf.ols('value ~ -1 + variable', data=vendor_long).fit()
print(model.conf_int())

## Simultaneous Confidence Intervals:  Multiple Comparisons
hadpas = mistat.load_data('HADPAS')
ax = hadpas.boxplot(column='res3', by='hyb',
                    color={'boxes': 'grey', 'medians': 'black', 'whiskers': 'black'},
                    patch_artist=True)
ax.set_title('')
ax.get_figure().suptitle('')
ax.set_xlabel('hyb')
plt.show()

from statsmodels.stats.multicomp import pairwise_tukeyhsd
hadpas = mistat.load_data('HADPAS')
mod = pairwise_tukeyhsd(hadpas['res3'], hadpas['hyb'])
print(mod)

## Contingency Tables
### The Structure of Contingency Tables
insertion = mistat.load_data('INSERTION')
insertion['percentage'] = 100 * insertion['fail'] / (insertion['fail']+insertion['succ'])
ax = insertion.plot.bar(x='comp', y='percentage', color='grey')
ax.set_ylabel('%')
plt.show()

car = mistat.load_data('CAR')
count_table = car[['cyl', 'origin']].pivot_table(
             index='cyl', columns='origin', aggfunc=len, fill_value=0)
print(count_table)

### Indices of Association For Contingency Tables
#### Two Interval Scaled Variables
# create binned data set
bins_turn = np.array([27, 30.6, 34.2, 37.8, car['turn'].max()])
bins_mpg = np.array([12, 18, 24, car['mpg'].max()])
binned_car = pd.DataFrame({
  'turn': pd.cut(car['turn'], bins=bins_turn),
  'mpg': pd.cut(car['mpg'], bins=(12,18,24,car['mpg'].max())),
})
# calculate proportional frequency and marginals
freqDist = pd.crosstab(binned_car['turn'], binned_car['mpg'])
pij = freqDist/len(car)
p_mpg = np.sum(pij, axis=0)
p_turn = np.sum(pij, axis=1)

# calculate average turn and mpg
center_turn = 0.5 * (bins_turn[1:] + bins_turn[:-1])
center_mpg = 0.5 * (bins_mpg[1:] + bins_mpg[:-1])
mean_turn = np.sum(center_turn * p_turn)
mean_mpg = np.sum(center_mpg * p_mpg)

# calculate estimate of coefficient of correlation
rho1 = np.sum((pij.values * np.outer(center_turn - mean_turn,
                                     center_mpg - mean_mpg)))
rho2a = np.sqrt(np.sum(p_turn*(center_turn-mean_turn)**2))
rho2b = np.sqrt(np.sum(p_mpg*(center_mpg-mean_mpg)**2))
rho = rho1 / (rho2a * rho2b)
print(f"r_XY   {np.corrcoef(car['turn'], car['mpg'])[0][1]:.3f}")
print(f'rho_XY {rho:.3f}')

#### Indices of Association for Categorical Variables
chi2 = stats.chi2_contingency(count_table)
print(f'chi2 statistic {chi2[0]:.2f}')

## Categorical Data Analysis
### Comparison of Binomial Experiments
df = pd.DataFrame({
  'i': [1, 2, 3, 4, 5, 6, 7, 8, 9],
  'Ji': [61, 34, 10, 23, 25, 9, 12, 3, 13],
  'ni': [108119,136640,107338,105065,108854,96873,107391,105854,180630],
  })
df['Yi'] = 2*np.arcsin(np.sqrt((df['Ji'] + 3/8)/(df['ni'] + 3/4)))

Ybar = np.sum(df['ni'] * df['Yi']) / np.sum(df['ni'])
Q = np.sum(df['ni'] * (df['Yi'] - Ybar) ** 2)
print(Q)

stats.chi2.cdf(105.43, df=8)

## Chapter Highlights
## Exercises
