<!-- TODO: 
- Notebook 7: Shapley values, reference and description dataset, warnings on column names 
- Notebook 8: Text mining to use to two or three other datasets, Bayesian Network require Bioreactor dataset
-->

![Python](https://github.com/gedeck/mistat-code-solutions/actions/workflows/run-notebooks.yml/badge.svg)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gedeck/mistat-code-solutions/binder-modern-statistics?labpath=BioMed_DataAnalyst_Course%2Fnotebooks%2Findex.ipynb)

# A Biomed Data Analyst Training Program
<table>
<tr>
<td>
  <a href="../ModernStatistics/"><img src="../img/ModernStatistics.png" width=250></a>
</td>
<td>
  <p>
    <b><a href="../ModernStatistics/">Modern Statistics: A Computer Based Approach with Python</a></b><br>
    by Ron Kenett, Shelemyahu Zacks, Peter Gedeck
  </p>


  <p>
    Publisher: <a href="https://link.springer.com/book/10.1007/978-3-031-07566-7">Springer International Publishing; 1st edition (September 15, 2022)</a><br>
    ISBN-13: 978-3-031-07565-0 (hardcover)<br>
    ISBN-13: 978-3-031-07568-1 (softcover)<br>
    ISBN-13: 978-3-031-28482-3 (eBook).<br>
    Buy at 
    <a href="https://www.amazon.com/Modern-Statistics-Computer-Based-Technology-Engineering/dp/303107565X/?&_encoding=UTF8&tag=petergedeck-20&linkCode=ur2&linkId=84f11d31518ed35288c8fcd790f5516f&camp=1789&creative=9325">Amazon</a>,
    <a href="https://link.springer.com/book/10.1007/978-3-031-07566-7">Springer</a>, 
    <a href="https://www.barnesandnoble.com/w/modern-statistics-ron-kenett/1141391736">Barnes & Noble</a>
  </p>

  <p>Errata: <a href="../ModernStatistics/errata">See known errata here</a></p>
</td>
</tr>
</table>

# Slides

1. <a href="slides/1. Kenett Biomed intro.pdf">Introduction</a>
2. <a href="slides/2. Kenett Biomed Data Integration.pdf">Data types and data integration</a>
3. <a href="slides/3. Kenett Biomed Supervised Learning.pdf">Supervised learning</a>
4. <a href="slides/4. Kenett Biomed Model Performance.pdf">Model performance</a>
5. <a href="slides/5. Kenett Biomed Time Series.pdf">Time series</a>
6. <a href="slides/6. Kenett Biomed Data Visualization.pdf">Data visualization</a>
7. <a href="slides/7. Kenett Biomed Causality and DOE.pdf">Causality and experimental design</a>

# Code and data files

This part of the repository contains:

- `notebooks`: Python code of individual chapters in 
  [Jupyter notebooks](https://github.com/gedeck/mistat-code-solutions/tree/main/BioMed_DataAnalyst_Course/notebooks) - 
  [download notebooks and data as notebooks.zip](notebooks.zip)

The Python package `mistat` contains datafiles and utility functions referred to in the <a href="ModernStatistics">Modern Statistics</a> book. It is available 
for installation from the Python package index [https://pypi.org/project/mistat/](https://pypi.org/project/mistat/).
The `mistat` packages is maintained in a GitHub repository at [https://github.com/gedeck/mistat](https://github.com/gedeck/mistat).

# Try the code
You can explore the code on <a href="https://mybinder.org/v2/gh/gedeck/mistat-code-solutions/binder-modern-statistics?labpath=BioMed_DataAnalyst_Course%2Fnotebooks%2Findex.ipynb" target="_blank">Binder <img src="https://mybinder.org/badge_logo.svg"></a>.

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

The notebook [InstallPackages.ipynb](../ModernStatistics/InstallPackages.ipynb) contains the pip command to install the required packages. Note that some of the packages may need to be pinned to specific versions.

If you have a problem with visualizing the decision tree or creating a network graph, follow the [installation instructions for graphviz in the dtreeviz github site](https://github.com/parrt/dtreeviz). On Windows, the problem is usually resolved by adding the path to the graphviz binaries to the PATH system variable.



