## Exercise chapter 2
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

import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt



################################################################################
# Exercise 2
################################################################################
fig, ax = plt.subplots(figsize=[3, 3])
ax.plot([10, 15, 10, 10], [10, 10, 15, 10], color='black')
ax.plot([10, 15], [15, 10], color='black', linewidth=3)
ax.set_xlim(0, 25)
ax.set_ylim(0, 25)
ax.text(11.2, 11.2, 'B')
ax.text(13, 13, 'A')
ax.set_xlabel('$\omega_1$')
ax.set_ylabel('$\omega_2$')
plt.show()









































################################################################################
# Exercise 42
################################################################################
x = list(range(15))
table = pd.DataFrame({
  'x': x,
  'p.d.f.': [stats.binom(20, 0.17).pmf(x) for x in x],
  'c.d.f.': [stats.binom(20, 0.17).cdf(x) for x in x],
})
print(table)














################################################################################
# Exercise 55
################################################################################
stats.nbinom.ppf([0.25, 0.5, 0.75], 3, 0.01)









################################################################################
# Exercise 63
################################################################################
rv = stats.norm(100, 15)
print('(i)', rv.cdf(108) - rv.cdf(92))
print('(ii)', 1 - rv.cdf(105))
print('(iii)', rv.cdf((200 - 5)/2))













################################################################################
# Exercise 75
################################################################################
from scipy.special import gamma
print(gamma(1.17), gamma(1 / 2), gamma(3 / 2))































################################################################################
# Exercise 105
################################################################################
print(stats.t.ppf(0.95, 10))
print(stats.t.ppf(0.95, 15))
print(stats.t.ppf(0.95, 20))


################################################################################
# Exercise 106
################################################################################
print(stats.f.ppf(0.95, 10, 30))
print(stats.f.ppf(0.95, 15, 30))
print(stats.f.ppf(0.95, 20, 30))





