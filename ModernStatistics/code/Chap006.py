## Chapter 6
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

# Time Series Analysis and Prediction
import datetime
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ValueWarning
import pandas as pd

import random
import numpy as np
import pingouin as pg
from scipy import stats
import matplotlib.pyplot as plt
import mistat

## The Components of a Time Series
### The Trend and Covariances
dow1941 = mistat.load_data('DOW1941')
t = np.arange(1, len(dow1941) + 1)
x = (t - 151) / 302
omega = 4 * np.pi * t / 302
ft = (123.34 + 27.73 * x - 15.83* x ** 2 - 237.00 * x**3
      + 0.1512 * np.cos(omega) + 1.738 * np.sin(omega)
      + 1.770 * np.cos(2 * omega) - 0.208 * np.sin(2 * omega)
      - 0.729 * np.cos(3 * omega) + 0.748 * np.sin(3 * omega))

fig, ax = plt.subplots(figsize=[4, 4])
ax.scatter(dow1941.index, dow1941, facecolors='none', edgecolors='grey')
ax.plot(t, ft, color='black')
ax.set_xlabel('Working day')
ax.set_ylabel('DOW1941')
plt.show()

### Analyzing Time Series With Python
dow1941 = mistat.load_data('DOW1941_DATE')

# convert Date column to Python datetime
dates = pd.to_datetime(dow1941['Date'], format='%Y-%m-%d')
dow1941_ts = pd.Series(dow1941['Open'], name='Dow_Jones_Index')
dow1941_ts.index = pd.DatetimeIndex(dates)

dow1941_ts.head()

from statsmodels.tsa import tsatools
dow1941_df = tsatools.add_trend(dow1941_ts, trend='ct')
dow1941_df.head()

from statsmodels.tsa import tsatools
dow1941_df = tsatools.add_trend(dow1941_ts, trend='ct')
model_1 = smf.ols(formula='Dow_Jones_Index ~ trend + 1', data=dow1941_df).fit()
print(model_1.params)
print(f'r2-adj: {model_1.rsquared_adj:.3f}')

ax = dow1941_ts.plot(color='grey')
ax.set_xlabel('Time')
ax.set_ylabel('Dow Jones index')
model_1.predict(dow1941_df).plot(ax=ax, color='black')
plt.show()

dow1941_df = tsatools.add_trend(dow1941_ts, trend='ct')
formula = 'Dow_Jones_Index ~ I(trend**3) + I(trend**2) + trend + 1'
model_2 = smf.ols(formula=formula, data=dow1941_df).fit()
print(model_2.params)
print(f'r2-adj: {model_2.rsquared_adj:.3f}')

ax = dow1941_ts.plot(color='grey')
ax.set_xlabel('Time')
ax.set_ylabel('Dow Jones index')
model_2.predict(dow1941_df).plot(ax=ax, color='black')
plt.show()

dow1941_df = tsatools.add_trend(dow1941_ts, trend='ct')
dow1941_df['month'] = dow1941_df.index.month
poly_formula = 'Dow_Jones_Index ~ C(month) + I(trend**3) + I(trend**2) + trend + 1'
model_3 = smf.ols(formula=poly_formula, data=dow1941_df).fit()
print(model_3.params)
print(f'r2-adj: {model_3.rsquared_adj:.3f}')

ax = dow1941_ts.plot(color='grey')
ax.set_xlabel('Time')
ax.set_ylabel('Dow Jones index')
model_3.predict(dow1941_df).plot(ax=ax, color='black')
plt.show()

fig, axes = plt.subplots(figsize=[4, 5], nrows=3)
def residual_plot(model, ax, title):
  model.resid.plot(color='grey', ax=ax)
  ax.set_xlabel('')
  ax.set_ylabel(f'Residuals\n{title}')
  ax.axhline(0, color='black')
residual_plot(model_1, axes[0], 'Model 1')
residual_plot(model_2, axes[1], 'Model 2')
residual_plot(model_3, axes[2], 'Model 3')
axes[2].set_xlabel('Time')
plt.tight_layout()
plt.show()

def plotLag(ts, lag, ax, limits):
  ax.scatter(ts[:-lag], ts[lag:], facecolors='none', edgecolors='black')
  ax.set_title(f'Lag {lag}')
  ax.set_xlim(*limits)
  ax.set_ylim(*limits)

fig, axes = plt.subplots(figsize=[6, 6], nrows=2, ncols=2)
limits = [dow1941_ts.min(), dow1941_ts.max()]
plotLag(dow1941_ts, 1, axes[0][0], limits)
plotLag(dow1941_ts, 5, axes[0][1], limits)
plotLag(dow1941_ts, 15, axes[1][0], limits)
plotLag(dow1941_ts, 60, axes[1][1], limits)

plt.tight_layout()
plt.show()

## Covariance Stationary Time Series
### Moving Averages
### Auto-Regressive Time Series
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

dow_acf = acf(dow1941_ts, nlags=15, fft=True)
dow_pacf = pacf(dow1941_ts, nlags=15)

fig, axes = plt.subplots(ncols=2, figsize=[8, 3.2])
plot_acf(dow1941_ts, lags=15, ax=axes[0])
plot_pacf(dow1941_ts, lags=15, method='ywm', ax=axes[1])
plt.tight_layout()
plt.show()

print(f'  k     rho(k)    rho*(k)')
for i, (v1, v2) in enumerate(zip(dow_acf, dow_pacf)):
  print(f'{i:3d}   {v1:8.4f}   {v2:8.4f}')

for i, (v1, v2) in enumerate(zip(dow_acf, dow_pacf)):
  print(f'{i} & {v1:.4f} & {v2:.4f} \\\\')
dow_acf = acf(model_3.resid, nlags=15, fft=True)
fig, axes = plt.subplots(ncols=2, figsize=[8, 3.2])
plot_acf(model_3.resid, lags=15, ax=axes[0])
plot_pacf(model_3.resid, lags=15, method='ywm', ax=axes[1])
plt.show()

print(f'  k     rho(k)')
for i, v1 in enumerate(dow_acf):
  print(f'{i:3d}   {v1:8.4f}')

for i, v1 in enumerate(dow_acf):
  print(f'{i} & {v1:.4f}  \\\\')
### Auto-Regressive Moving Averages Time Series
### Integrated Auto-Regressive Moving Average Time Series
### Applications with Python
# ignore ValueWarning for Kurtosis-test due to small dataset
import warnings
warnings.simplefilter('ignore', category=(ValueWarning, UserWarning))

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_predict

# Identify optimal ARMA options using the AIC score
bestModel = None
bestAIC = None
for ar in range(0, 5):
  for ma in range(0, 5):
    model = ARIMA(model_3.resid, order=(ar, 0, ma)).fit()
    if bestAIC is None or bestAIC > model.aic:
      bestAIC = model.aic
      bestModel = (ar, 0, ma)
print(f'Best model: {bestModel}')

model = ARIMA(model_3.resid, order=bestModel).fit()

prediction = model.get_forecast(30).summary_frame()
prediction['date'] = [max(dow1941_ts.index) + datetime.timedelta(days=i)
                      for i in range(1, len(prediction) + 1)]

plot_predict(model)
ax = plt.gca()

ax.plot(prediction['date'], prediction['mean'])
ax.fill_between(prediction['date'],
                prediction['mean_ci_lower'], prediction['mean_ci_upper'],
                color='lightgrey')
plt.show()

# restore default setting
warnings.simplefilter('default', category=ValueWarning)

## Linear Predictors for Covariance Stationary Time Series
### Optimal Linear Predictors
predictedError = mistat.optimalLinearPredictor(model_2.resid,11,nlags=10)
predictedTrend = model_2.predict(dow1941_df)
correctedTrend = predictedTrend + predictedError

fig, ax = plt.subplots()
ax.scatter(dow1941_ts.index, dow1941_ts,
           facecolors='none', edgecolors='grey')
predictedTrend.plot(ax=ax, color='grey')
correctedTrend.plot(ax=ax, color='black')
ax.set_xlabel('Time')
ax.set_ylabel('Dow Jones index')
plt.show()

print(f'PMSE(trend) = {np.mean((predictedTrend - dow1941_ts)**2):.4f}')
print(f'PMSE(corrected) = {np.mean((correctedTrend-dow1941_ts)**2):.4f}')

## Predictors for Non-Stationary Time Series
### Quadratic LSE Predictors
quadPrediction = mistat.quadraticPredictor(dow1941_ts, 20, 1)

fig, ax = plt.subplots()
ax.scatter(dow1941_ts.index, dow1941_ts,
           facecolors='none', edgecolors='grey')
ax.plot(dow1941_ts.index, quadPrediction, color='black')
ax.set_xlabel('Time')
ax.set_ylabel('Dow Jones index')
plt.show()

print(f'PMSE(quadratic) = {np.mean((quadPrediction-dow1941_ts)**2):.4f}')

### Moving Average Smoothing Predictors
masPrediction = mistat.masPredictor(dow1941_ts, 3, 1)

fig, ax = plt.subplots()
ax.scatter(dow1941_ts.index, dow1941_ts,
           facecolors='none', edgecolors='grey')
ax.plot(dow1941_ts.index, masPrediction, color='black')
ax.set_xlabel('Time')
ax.set_ylabel('Dow Jones index')
plt.show()

print(f'PMSE(MAS) = {np.mean((masPrediction - dow1941_ts)**2):.4f}')

## Dynamic Linear Models
### Some Special Cases
#### The Normal Random Walk
res = mistat.normRandomWalk(100, 3, 1, 1, seed=2)

fig, ax = plt.subplots()
ax.scatter(res.t, res.X, facecolors='none', edgecolors='grey')
ax.plot(res.t, res.predicted, color='black')
ax.set_xlabel('Time')
ax.set_ylabel('TS')

plt.show()

#### Dynamic Linear Model With Linear Growth
C0 = np.array([[0.22325, -0.00668], [-0.00668, 0.00032]])
M0 = np.array([134.234, -0.3115])
W = np.array([[0.3191, -0.0095], [-0.0095, 0.0004]])
v = 1

dow1941 = mistat.load_data('DOW1941.csv')
predicted = mistat.dlmLinearGrowth(dow1941, C0, v, W, M0)

fig, ax = plt.subplots()
ax.scatter(dow1941.index, dow1941, facecolors='none', edgecolors='grey')
ax.plot(dow1941.index, predicted, color='black')
ax.set_xlabel('Time')
ax.set_ylabel('Dow Jones index')
plt.show()

#### Dynamic Linear Model for ARMA(p,q)
a = [0.5, 0.3, 0.1]
b = [0.3, 0.5]
ts = pd.Series(mistat.simulateARMA(100, a, b, seed=1))
predicted = mistat.predictARMA(ts, a)

fig, ax = plt.subplots()
ax.scatter(ts.index, ts, facecolors='none', edgecolors='grey')
ax.plot(ts.index, predicted, color='black')
ax.set_xlabel('Time')
ax.set_ylabel('TS')
plt.show()

print(f'PMSE(ARMA) = {np.mean((predicted - ts)**2):.4f}')

## Chapter Highlights
## Exercises
