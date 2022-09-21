<style>
  table {
    border: 0px;
  }
  td {
    border: 0px;
    vertical-align: top;
  }
  .inner {
    max-width: 800px;
  }
</style>
![Python](https://github.com/gedeck/mistat-code-solutions/actions/workflows/run-notebooks.yml/badge.svg)

# Code repository
<table>
<tr>
<td><img src="../img/ModernStatistics.png" width=250></td>
<td>
  <b>Modern Statistics: A Computer Based Approach with Python</b>

by Ron Kenett, Shelemyahu Zacks, Peter Gedeck

Publisher: Springer International Publishing; 1st edition (September 15, 2022)
ISBN-13: 978-3031075650
Buy on 
<a href="https://www.amazon.com/Modern-Statistics-Computer-Based-Technology-Engineering/dp/303107565X/">Amazon</a>, 
<a href="https://www.barnesandnoble.com/w/modern-statistics-ron-kenett/1141391736">Barnes & Noble</a>

<!-- Errata: http://oreilly.com/catalog/errata.csp?isbn=9781492072942 -->
</td>
</tr>
</table>
<p><i>Modern Statistics: A Computer Based Approach with Python</i> is a companion volume to the book <a href="../IndustrialStatistics"><i>Industrial Statistics: A Computer Based Approach with Python.</i></a></p>

This part of the repository contains:

- `notebooks`: Python code of individual chapters in 
  [Jupyter notebooks](https://github.com/gedeck/mistat-code-solutions/new/main/ModernStatistics/notebooks) - 
  [download all as notebooks.zip](notebooks.zip)
- `code`: Python code for solutions as plain 
  [Python files](https://github.com/gedeck/mistat-code-solutions/tree/main/ModernStatistics/code) - 
  [download all as code.zip](code.zip)
- `solutions manual`: [Solutions_Modernstatistics.pdf](Solutions_Modernstatistics.pdf): solutions of exercises
- `solutions`: Python code for solutions in Jupyter 
  [notebooks](https://github.com/gedeck/mistat-code-solutions/tree/main/ModernStatistics/solutions) - 
  [download all as solutions.zip](solutions.zip)
- `all`: zip file with all files combined - [download all as all.zip](all.zip)

All the Python applications referred to in this book are contained in a package called `mistat` available 
for installation from the Python package index [https://pypi.org/project/mistat/](https://pypi.org/project/mistat/).
The `mistat` packages is maintained in a GitHub repository at [https://github.com/gedeck/mistat](https://github.com/gedeck/mistat).

# Installation instructions
Instructions on installing Python and required packages are <a href="../doc/installPython">here</a>.

These Python packages are used in the code examples of Modern Statistics: 
- mistat
- numpy 
- scipy 
- scikit-learn
- statsmodels
- pingouin
- xgboost
- KDEpy
- networkx
- scikit-fda
- pgmpy
- dtreeviz
- svglib
- pydotplus


# Table of contents

<a href='#chapter-1-analyzing-variability-descriptive-statistics'>Chapter 1: Analyzing Variability: Descriptive Statistics</a><br>
<a href='#chapter-2-probability-models-and-distribution-functions'>Chapter 2: Probability Models and Distribution Functions</a><br>
<a href='#chapter-3-statistical-inference-and-bootstrapping'>Chapter 3: Statistical Inference and Bootstrapping</a><br>
<a href='#chapter-4-variability-in-several-dimensions-and-regression-models'>Chapter 4: Variability in Several Dimensions and Regression Models</a><br>
<a href='#chapter-5-sampling-for-estimation-of-finite-population-quantities'>Chapter 5: Sampling for Estimation of Finite Population Quantities</a><br>
<a href='#chapter-6-time-series-analysis-and-prediction'>Chapter 6: Time Series Analysis and Prediction</a><br>
<a href='#chapter-7-and-8-modern-analytic-methods-part-i-and-ii'>Chapter 7: Modern analytic methods: Part I</a><br>
<a href='#chapter-7-and-8-modern-analytic-methods-part-i-and-ii'>Chapter 8: Modern analytic methods: Part II</a><br>


# Introductory videos

## Chapter 1: Analyzing Variability: Descriptive Statistics
The chapter focuses on statistical variability and on various methods of
analyzing random data.  Random
results of experiments are illustrated with distinction between
deterministic and random components of variability.  The difference between
accuracy and precision is explained.  Frequency distributions are defined to
represent random phenomena.  Various characteristics of location and
dispersion of frequency distributions are defined.  The elements of
exploratory data analysis are presented.

<video src="https://user-images.githubusercontent.com/8720575/180794305-25b0ee4d-6b15-4cd8-86cc-a60d555add9b.mp4" controls="controls" style="max-width: 730px;">
</video>

## Chapter 2: Probability Models and Distribution Functions
The chapter provides the basics of probability theory and of the theory of 
distribution functions.  The probability model for random	sampling is 
discussed.  This is fundamental for	statistical inference discussed in 
Chapter 3 and sampling procedures in Chapter 5.
Bayes' theorem also presented here has important ramifications in 
statistical inference, including Bayesian process monitoring and 
Bayesian reliability presented in Chapter 3 and Chapter 9, 
respectively (in the Industrial Statistics book).

<video src="https://user-images.githubusercontent.com/8720575/180794620-207dc1ca-3932-41d2-8140-087984ec5595.mp4" controls="controls" style="max-width: 730px;">
</video>

## Chapter 3: Statistical Inference and Bootstrapping
In this chapter we introduce basic concepts and methods of statistical inference.  The focus is on
estimating the parameters of statistical distributions
and of testing hypotheses about them. Problems of
testing if certain distributions fit observed data
are also considered.

<video src="https://user-images.githubusercontent.com/8720575/180794659-7e870498-7b03-4754-a18b-80ebee7fc4f3.mp4" controls="controls" style="max-width: 730px;">
</video>

## Chapter 4: Variability in Several Dimensions and Regression Models
When surveys or experiments are performed, measurements are
usually taken on several characteristics of the observation
elements in the sample.  In such cases we have multi-variate
observations, and the statistical methods which are used to
analyze the relationships between the values observed on different
variables are called multivariate methods.  In this chapter we
introduce some of these methods.  In particular, we focus
attention on graphical methods,
linear regression methods and the analysis of contingency
tables.  The linear regression methods explore the linear relationship
between a variable of interest and a set of variables, by which we try to
predict the values of the variable of interest.  Contingency tables
analysis studies the association between qualitative
(categorical) variables, on which
we cannot apply the usual regression methods.

<video src="https://user-images.githubusercontent.com/8720575/180794686-7fed67fa-207c-4f05-84be-658ac8fa2e0e.mp4" controls="controls" style="max-width: 730px;">
</video>

## Chapter 5: Sampling for Estimation of Finite Population Quantities
Techniques for sampling finite populations and estimating population
parameters are presented.  Formulas are given for the expected value and
variance of the sample mean and sample variance of simple random samples
with and without replacement.  Stratification is studied as a method to
increase the precision of estimators.  Formulas for proportional and
optimal allocation are provided and demonstrated with case studies.
The chapter is concluded with a section on
prediction models with known covariates.

<video src="https://user-images.githubusercontent.com/8720575/180794750-f21c7beb-3086-4f51-87d9-3dd1970ff540.mp4" controls="controls" style="max-width: 730px;">
</video>

## Chapter 6: Time Series Analysis and Prediction
In this chapter, we present essential parts of time series analysis,
with the objective of predicting or forecasting its future development.
Predicting future behavior is generally more successful for stationary
series, which do not change their stochastic characteristics as time proceeds.
We develop and illustrate time series which are of both types, namely
covariance stationary, and non-stationary.

<video src="https://user-images.githubusercontent.com/8720575/180794728-102532ae-be22-41ff-96de-4c7e7c11b91e.mp4" controls="controls" style="max-width: 730px;">
</video>


## Chapter 7 and 8: Modern analytic methods: Part I and II
Chapter 7 is a door opener to computer age statistics. It covers a 
range of supervised and unsupervised learning methods and
demonstrates their use in various applications.

Chapter 8 includes
tip of the iceberg examples with what we thought were interesting
insights, not always available in standard texts.
The chapter covers functional data analysis, text analytics,
reinforcement learning, Bayesian networks, and causality models.
 
<video src="https://user-images.githubusercontent.com/8720575/180794703-c6f05f40-eefd-4e1a-93f9-42cb78e6a6b4.mp4" controls="controls" style="max-width: 730px;">
</video>

