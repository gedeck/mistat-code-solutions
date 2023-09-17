# Errata

The errata list is a list of errors and their corrections that were found after the product was released. Use the [Github issue tracker](https://github.com/gedeck/mistat-code-solutions/issues/new?assignees=&labels=&template=modern-statistics.md) to submit new errors.


## Chapter 4
- p. 135, error at end of page: The sentence starting with "Abadie et al."  should read:
> Abadie et al. (2015) estimated the weights using $𝑊 \gg 0$ and $\sum_1^{m} w_j = 1$ as an additional constraint.

## Chapter 5
- p. 185, clarify use rendering of interactions
>  For the interaction part, lower factor levels are identified 
as red and higher factor levels as black half squares. _The left half of the square 
corresponds to the first factor in the interaction, and the right half 
to the second factor, respectively._

## Chapter 7
- p. 283-284, mean values for sample sizes of 100 and 1000 are different from the values in Figure 7.7 (note that the boxplots show the median while the text shows mean values)

  $\text{mean}(\exp\{X+Y\}, n=100) = 5.014902$
  
  $\text{mean}(\exp\{X+Y\}, n=1000) = 6.418258$


## Chapter 9
- p. 351, the first line should read: 
> Example 9.17 Using the censored data from **Example** 9.16, we estimate the ...


## Chapter 10
- p. 384, due to change in `pymc` the function `weibull_log_sf` needs to be changed (download notebooks to get more recent versions of code).
```
  def weibull_log_sf(y, nu, beta):
      return - (y / beta) ** nu
```
