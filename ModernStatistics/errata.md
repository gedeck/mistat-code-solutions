# Errata

The errata list is a list of errors and their corrections that were found after the product was released. Use the [Github issue tracker](https://github.com/gedeck/mistat-code-solutions/issues/new?assignees=&labels=&template=modern-statistics.md) to submit new errors.

## Chapter 1
- p. 12-13, code for Figure 1.7 and Figure 1.8 - `value_counts` creates now column named `proportion`:
  ```
  ...
  X.loc[4, 'proportion'] = 0  # there are no samples with 4 blemishes add a row
  X = X.sort_index()  # sort by number of blemishes
  ax = X['proportion'].plot.bar(color='grey', legend=False)
  ...
  X['Cumulative Frequency'] = X['proportion'].cumsum()
  ...
  ```

## Chapter 2
- p. 55, Section 2.2.1.1 - Clarification: The `scipy` package uses the term _probability mass function_
  instead of _probability distribution function_ for discrete distributions. Both terms are used 
  in the literature. In _Modern Statistics_, we use the term _probability distribution function_ 
  for both discrete and continuous distributions.
  
## Chapter 3
- p. 162, Equation 3.30 - Replace percentile for the first $\chi^2$:

  $\left(\frac{(n-1)S^2}{\chi^2_{1-\alpha/2}[n-1]},
         \frac{(n-1)S^2}{\chi^2_{\alpha/2}[n-1]}\right)$

- p. 186, Figure 3.15 - Incorrect distribution used to create the Figure. The updated code creates the following Figure. The additional dashed lines show the approximated HPD.
  <img src='../img/MS-Fig-3.15.png'>
- p. 196 - Code sample, the print statement needs to contain f-strings. Replace
  ```
  print('Xbar {Xbar:.2f} / SX {SX:.3f}')
  print('Ybar {Xbar:.2f} / SY {SX:.3f}')
  ```
    with
  ```
  print(f'Xbar {Xbar:.2f} / SX {SX:.3f}')
  print(f'Ybar {Ybar:.2f} / SY {SY:.3f}')
  ```

## Chapter 4
- Exercise 4.27: effects => affects

## Chapter 6
- p. 357 - Correct value of PMSE
  > The empirical PMSE is 0.8505.

## Chapter 7
- p. 379 - iteritems deprecated in pandas package; code change required:
  ```
  df = pd.DataFrame([
    {satisfaction: counts for satisfaction, counts
      in response.value_counts().items()},
    {satisfaction: counts for satisfaction, counts
      in response[q1_5].value_counts().items()},
  ])
  ```


## Chapter 8
- p. 404 - In equation 8.2, the matrix $V$ needs to be transposed:<br>
  $DTM \approx U * S * V'$

## Index
- P-value, 152, 215
