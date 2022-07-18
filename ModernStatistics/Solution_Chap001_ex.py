## Exercise chapter 1
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

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mistat
from scipy import stats

def trim_std(data, alpha):
  """ Calculate trimmed standard deviation """
  data = np.array(data)
  data.sort()
  n = len(data)
  low = int(n * alpha) + 1
  high = int(n * (1 - alpha))
  return data[low:(high + 1)].std()


################################################################################
# Exercise 1
################################################################################
import random
random.seed(1)
values = random.choices([1, 2, 3, 4, 5, 6], k=50)

from collections import Counter
Counter(values)


################################################################################
# Exercise 2
################################################################################
random.seed(1)
x = list(range(50))
y = [5 + 2.5 * xi for xi in x]
y = [yi + random.uniform(-10, 10) for yi in y]

x = list(range(50))
y = [5 + 2.5 * xi for xi in x]
y = [yi + random.uniform(-10, 10) for yi in y]
pd.DataFrame({'x': x, 'y': y}).plot.scatter(x='x', y='y')
plt.show()


################################################################################
# Exercise 3
################################################################################
from scipy.stats import binom
np.random.seed(1)

for p in (0.1, 0.3, 0.7, 0.9):
  X = binom.rvs(1, p, size=50)
  print(p, sum(X))


################################################################################
# Exercise 4
################################################################################
inst1 = [9.490950, 10.436813, 9.681357, 10.996083, 10.226101, 10.253741,
         10.458926, 9.247097, 8.287045, 10.145414, 11.373981, 10.144389,
         11.265351, 7.956107, 10.166610, 10.800805, 9.372905, 10.199018,
         9.742579, 10.428091]
inst2 = [11.771486, 10.697693, 10.687212, 11.097567, 11.676099,
         10.583907, 10.505690, 9.958557, 10.938350, 11.718334,
         11.308556, 10.957640, 11.250546, 10.195894, 11.804038,
         11.825099, 10.677206, 10.249831, 10.729174, 11.027622]
ax = pd.Series(inst1).plot(marker='o', linestyle='none',
                           fillstyle='none', color='black')
pd.Series(inst2).plot(marker='+', linestyle='none', ax=ax,
                      fillstyle='none', color='black')
plt.show()

print('mean inst1', np.mean(inst1))
print('stdev inst1', np.std(inst1, ddof=1))
print('mean inst2', np.mean(inst2))
print('stdev inst2', np.std(inst2, ddof=1))



################################################################################
# Exercise 6
################################################################################
import random
random.choices(range(1, 101), k=20)


################################################################################
# Exercise 7
################################################################################
import random
random.sample(range(11, 31), 10)




################################################################################
# Exercise 10
################################################################################
filmsp = mistat.load_data('FILMSP')
filmsp.plot.hist()
plt.show()


################################################################################
# Exercise 11
################################################################################
coal = mistat.load_data('COAL')
pd.DataFrame(coal.value_counts(sort=False))


################################################################################
# Exercise 12
################################################################################
car = mistat.load_data('CAR')
car['cyl'].value_counts(sort=False)

pd.cut(car['turn'], bins=range(28, 46, 2)).value_counts(sort=False)

pd.cut(car['hp'], bins=range(50, 275, 25)).value_counts(sort=False)

pd.cut(car['mpg'], bins=range(9, 38, 5)).value_counts(sort=False)


################################################################################
# Exercise 13
################################################################################
filmsp = mistat.load_data('FILMSP')
filmsp = filmsp.sort_values(ignore_index=True)  # sort and reset index

print(filmsp.quantile(q=[0, 0.25, 0.5, 0.75, 1.0]))
print(filmsp.quantile(q=[0.8, 0.9, 0.99]))

def calculate_quantile(x, q):
  idx = (len(x) - 1) * q
  left = math.floor(idx)
  right = math.ceil(idx)
  return 0.5 * (x[left] + x[right])

for q in (0, 0.25, 0.5, 0.75, 0.8, 0.9, 0.99, 1.0):
  print(q, calculate_quantile(filmsp, q))


################################################################################
# Exercise 14
################################################################################
filmsp = mistat.load_data('FILMSP')
n = len(filmsp)
mean = filmsp.mean()
deviations = [film - mean for film in filmsp]
S = math.sqrt(sum(deviation**2 for deviation in deviations) / n)

skewness = sum(deviation**3 for deviation in deviations) / n / (S**3)
kurtosis = sum(deviation**4 for deviation in deviations) / n / (S**4)
print('Python:\n',
      f'Skewness: {skewness}, Kurtosis: {kurtosis}')

print('Pandas:\n',
      f'Skewness: {filmsp.skew()}, Kurtosis: {filmsp.kurtosis()}')


################################################################################
# Exercise 15
################################################################################
car = mistat.load_data('CAR')
car['mpg'].groupby(by=car['origin']).mean()
car['mpg'].groupby(by=car['origin']).std()
# calculate both at the same time
print(car['mpg'].groupby(by=car['origin']).agg(['mean', 'std']))


################################################################################
# Exercise 16
################################################################################
car = mistat.load_data('CAR')
car_US = car[car['origin'] == 1]
gamma = car_US['turn'].std() / car_US['turn'].mean()


################################################################################
# Exercise 17
################################################################################
car = mistat.load_data('CAR')

car_US = car[car['origin'] == 1]
car_Asia = car[car['origin'] == 3]
print('US')
print('mean', car_US['turn'].mean())
print('geometric mean', stats.gmean(car_US['turn']))
print('Japanese')
print('mean', car_Asia['turn'].mean())
print('geometric mean', stats.gmean(car_Asia['turn']))


################################################################################
# Exercise 18
################################################################################
filmsp = mistat.load_data('FILMSP')

Xbar = filmsp.mean()
S = filmsp.std()
print(f'mean: {Xbar}, stddev: {S}')
expected = {1: 0.68, 2: 0.95, 3: 0.997}
for k in (1, 2, 3):
  left = Xbar - k * S
  right = Xbar + k * S
  proportion = sum(left < film < right for film in filmsp)
  print(f'X +/- {k}S: ',
        f'actual freq. {proportion}, ',
        f'pred. freq. {expected[k] * len(filmsp):.2f}')


################################################################################
# Exercise 19
################################################################################
car = mistat.load_data('CAR')
car.boxplot(column='mpg', by='origin')
plt.show()


################################################################################
# Exercise 20
################################################################################
oturb = mistat.load_data('OTURB')
mistat.stemLeafDiagram(oturb, 2, leafUnit=0.01)

oturb = mistat.load_data('OTURB')
mistat.stemLeafDiagram(oturb, 2, leafUnit=0.01, latex=True)


################################################################################
# Exercise 21
################################################################################
from scipy.stats import trim_mean

oturb = mistat.load_data('OTURB')
print(f'T(0.1) = {trim_mean(oturb, 0.1)}')
print(f'S(0.1) = {trim_std(oturb, 0.1)}')


################################################################################
# Exercise 22
################################################################################
germanCars = [10, 10.9, 4.8, 6.4, 7.9, 8.9, 8.5, 6.9, 7.1,
              5.5, 6.4, 8.7, 5.1, 6.0, 7.5]
japaneseCars = [9.4, 9.5, 7.1, 8.0, 8.9, 7.7, 10.5, 6.5, 6.7,
                9.3, 5.7, 12.5, 7.2, 9.1, 8.3, 8.2, 8.5, 6.8, 9.5, 9.7]
# convert to pandas Series
germanCars = pd.Series(germanCars)
japaneseCars = pd.Series(japaneseCars)
# use describe to calculate statistics
comparison = pd.DataFrame({
  'German': germanCars.describe(),
  'Japanese': japaneseCars.describe(),
})
print(comparison)


################################################################################
# Exercise 23
################################################################################
hadpas = mistat.load_data('HADPAS')
sampleStatistics = pd.DataFrame({
  'res3': hadpas['res3'].describe(),
  'res7': hadpas['res7'].describe(),
})
print(sampleStatistics)

ax = hadpas.hist(column='res3', alpha=0.5)
hadpas.hist(column='res7', alpha=0.5, ax=ax)
plt.show()

print('res3')
mistat.stemLeafDiagram(hadpas['res3'], 2, leafUnit=10)
print('res7')
mistat.stemLeafDiagram(hadpas['res7'], 2, leafUnit=10)


################################################################################
# Exercise 24
################################################################################
hadpas.boxplot(column='res3')
plt.show()

