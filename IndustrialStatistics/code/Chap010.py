## Chapter 
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

# Bayesian Reliability Estimation and Prediction
import random
import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import gamma
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
import mistat
import lifelines

## Prior and Posterior Distributions
from mistat import bayes

betaMixture = bayes.Mixture(
    probabilities=[0.5, 0.5],
    distributions=[bayes.BetaDistribution(a=1, b=1),
                   bayes.BetaDistribution(a=15, b=2)])
data = [10, 2]
result = bayes.updateBetaMixture(betaMixture, data)
thetas = [round(d.theta(), 2) for d in result.distributions]

print(f'A posteriori: {result.probabilities}')
print(f'Updated beta distributions:\n{result.distributions}')
print(f'Update theta values:\n{thetas}')

mixture = bayes.Mixture(
    probabilities=[0.5, 0.5],
    distributions=[
        bayes.GammaDistribution(shape=1, rate=1),
        bayes.GammaDistribution(shape=15, rate=2),
    ]
)
data = {'y': [5], 't': [1]}
result = bayes.updateGammaMixture(mixture, data)

print(f'A posteriori: {result.probabilities}')
print(f'Updated beta distributions:\n{result.distributions}')

## Loss Functions and Bayes Estimators
### Distribution-Free Bayes Estimator of Reliability
### Bayes Estimator of Reliability for Exponential Life Distributions
## Bayesian Credibility and Prediction Intervals
### Distribution-Free Reliability Estimation
### Exponential Reliability Estimation
### Prediction Intervals
### \remove
## Credibility Intervals for the Asymptotic Availability of Repairable Systems: The Exponential Case
## Empirical Bayes Method
## Chapter Highlights
## Exercises
