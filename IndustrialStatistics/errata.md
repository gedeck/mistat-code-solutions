# Errata

The errata list is a list of errors and their corrections that were found after the product was released. Use the [Github issue tracker](https://github.com/gedeck/mistat-code-solutions/issues/new?assignees=&labels=&template=modern-statistics.md) to submit new errors.

## Chapter 7
- p. 283-284, mean values for sample sizes of 100 and 1000 are different from the values in Figure 7.7 (note that the boxplots show the median while the text shows mean values)

  $\text{mean}(\exp\{X+Y\}, n=100) = 5.014902$
  
  $\text{mean}(\exp\{X+Y\}, n=1000) = 6.418258$


## Chapter 10
- p. 384, due to change in `pymc` the function `weibull_log_sf` needs to be changed (download notebooks to get more recent versions of code).
```
  def weibull_log_sf(y, nu, beta):
      return - (y / beta) ** nu
```
