![Python](https://github.com/gedeck/mistat-code-solutions/actions/workflows/run-notebooks.yml/badge.svg)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gedeck/mistat-code-solutions/binder-modern-statistics?labpath=BioMedicalCourse%2Fnotebooks%2Findex.ipynb)

# Code repository
<table>
<tr>
<td><img src="../img/ModernStatistics.png" width=250></td>
<td>
  <p>
    <b>BioMedical Statistics Course based on</b><br>
    <i><a href="../ModernStatistics/">Modern Statistics: A Computer Based Approach with Python</a></i><br>
    by Ron Kenett, Shelemyahu Zacks, Peter Gedeck
  </p>
</td>
</tr>
</table>
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
- `datafiles`: zip file with all data files - [download all as data_files.zip](data_files.zip) - the `mistat` 
  package gives you already access to all datafiles, you only need to download this file if you want to use it with 
  different software

All the Python applications referred to in this book are contained in a package called `mistat` available 
for installation from the Python package index [https://pypi.org/project/mistat/](https://pypi.org/project/mistat/).
The `mistat` packages is maintained in a GitHub repository at [https://github.com/gedeck/mistat](https://github.com/gedeck/mistat).

# Try the code
You can explore the code on <a href="https://mybinder.org/v2/gh/gedeck/mistat-code-solutions/binder-modern-statistics?labpath=ModernStatistics%2Fnotebooks%2Findex.ipynb" target="_blank">Binder <img src="https://mybinder.org/badge_logo.svg"></a>.

# Installation instructions
Instructions on installing Python and required packages are <a href="../doc/installPython">here</a>.

These Python packages are used in the code examples of _Modern Statistics_: 
- mistat (for access to data sets and additional functionality)
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

The notebook [InstallPackages.ipynb](InstallPackages.ipynb) contains the pip command to install the required packages. Note that some of the packages may need to be pinned to specific versions.

If you have a problem with visualizing the decision tree or creating a network graph, follow the [installation instructions for graphviz in the dtreeviz github site](https://github.com/parrt/dtreeviz). On Windows, the problem is usually resolved by adding the path to the graphviz binaries to the PATH system variable.

# Table of contents

<a href='#chapter-1-analyzing-variability-descriptive-statistics'>Chapter 1: Analyzing Variability: Descriptive Statistics</a><br>
<a href='#chapter-2-probability-models-and-distribution-functions'>Chapter 2: Probability Models and Distribution Functions</a><br>
<a href='#chapter-3-statistical-inference-and-bootstrapping'>Chapter 3: Statistical Inference and Bootstrapping</a><br>
<a href='#chapter-4-variability-in-several-dimensions-and-regression-models'>Chapter 4: Variability in Several Dimensions and Regression Models</a><br>
<a href='#chapter-5-sampling-for-estimation-of-finite-population-quantities'>Chapter 5: Sampling for Estimation of Finite Population Quantities</a><br>
<a href='#chapter-6-time-series-analysis-and-prediction'>Chapter 6: Time Series Analysis and Prediction</a><br>
<a href='#chapter-7-and-8-modern-analytic-methods-part-i-and-ii'>Chapter 7: Modern analytic methods: Part I</a><br>
<a href='#chapter-7-and-8-modern-analytic-methods-part-i-and-ii'>Chapter 8: Modern analytic methods: Part II</a><br>


# Slides

1. <a href="slides/1. Kenett Biomed intro.pdf">Introduction</a>
2. <a href="slides/2. Kenett Biomed Data Integration.pdf">Data types and data integration</a>
3. <a href="slides/3. Kenett Biomed Supervised Learning.pdf">Supervised learning</a>
4. <a href="slides/4. Kenett Biomed Model Performance.pdf">Model performance</a>
5. <a href="slides/5. Kenett Biomed Time Series.pdf">Time series</a>
6. <a href="slides/6. Kenett Biomed Data Visualization.pdf">Data visualization</a>
7. <a href="slides/7. Kenett Biomed Causality and DOE.pdf">Causality and experimental design</a>
