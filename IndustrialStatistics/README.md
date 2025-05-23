![Python](https://github.com/gedeck/mistat-code-solutions/actions/workflows/run-notebooks.yml/badge.svg)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gedeck/mistat-code-solutions/binder-industrial-statistics)
<a href="https://colab.research.google.com/github/gedeck/mistat-code-solutions/blob/main/IndustrialStatistics/notebooks/index.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Code repository
<table>
<tr>
<td><img src="../img/IndustrialStatistics.png" width=250></td>
<td>
  <p>
    <b>Industrial Statistics: A Computer Based Approach with Python</b><br>
    by Ron Kenett, Shelemyahu Zacks, Peter Gedeck
  </p>

  <p>
    Publisher: <a href="https://link.springer.com/book/10.1007/978-3-031-28482-3">Springer International Publishing; 
    1st edition (August 5, 2023)</a><br>
    ISBN-13: 978-3-031-28481-6 (hardcover)<br>
    ISBN-13: 978-3-031-28484-7 (softcover)<br>
    ISBN-13: 978-3-031-28482-3 (eBook).<br>
    Buy at
    <a href="https://www.amazon.com/Industrial-Statistics-Computer-Based-Technology-Engineering/dp/303128481X/?&_encoding=UTF8&tag=petergedeck-20&linkCode=ur2&linkId=189b1ce486c141101cec8046039d2120&camp=1789&creative=9325">Amazon</a>,
    <a href="https://link.springer.com/book/10.1007/978-3-031-28482-3">Springer</a>
<!--    <a href="https://www.barnesandnoble.com/w/modern-statistics-ron-kenett/1141391736">Barnes & Noble</a>-->
  </p>

  <p>Errata: <a href="errata">See known errata here</a></p>
</td>
</tr>
</table>


<p><i>Industrial Statistics: A Computer Based Approach with Python</i> is a companion volume to the book <a href="../ModernStatistics"><i>Modern Statistics: A Computer Based Approach with Python.</i></a></p>

This part of the repository contains:

- `notebooks`: Python code of individual chapters in 
  [Jupyter notebooks](https://github.com/gedeck/mistat-code-solutions/tree/main/IndustrialStatistics/notebooks) - 
  [download notebooks and data as notebooks.zip](notebooks.zip)
- `code`: Python code for solutions as plain 
  [Python files](https://github.com/gedeck/mistat-code-solutions/tree/main/IndustrialStatistics/code) - 
  [download all as code.zip](code.zip)
- `solutions manual`: [Solutions_IndustrialStatistics.pdf](Solutions_IndustrialStatistics.pdf): solutions of exercises
- `solutions`: Python code for solutions in Jupyter 
  [notebooks](https://github.com/gedeck/mistat-code-solutions/tree/main/IndustrialStatistics/solutions) - 
  [download all as solutions.zip](solutions.zip)
- `all`: zip file with all files combined - [download all as all.zip](all.zip)
- `datafiles`: zip file with all data files - [download all as data_files.zip](data_files.zip) - the `mistat`
  package gives you already access to all datafiles, you only need to download this file if you want to use it with 
  different software

All the Python applications referred to in this book are contained in a package called `mistat` available 
for installation from the Python package index [https://pypi.org/project/mistat/](https://pypi.org/project/mistat/).
The `mistat` packages is maintained in a GitHub repository at [https://github.com/gedeck/mistat](https://github.com/gedeck/mistat).

# Try the code
You can explore the code on 
- <a href="https://colab.research.google.com/github/gedeck/mistat-code-solutions/blob/main/IndustrialStatistics/notebooks/index.ipynb" target="_parent">Colab <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- <a href="https://mybinder.org/v2/gh/gedeck/mistat-code-solutions/binder-industrial-statistics?labpath=IndustrialStatistics%2Fnotebooks%2Findex.ipynb" target="_blank">Binder <img src="https://mybinder.org/badge_logo.svg"></a>.




# Installation instructions
Instructions on installing Python and required packages are <a href="../doc/installPython">here</a>.

These Python packages are used in the code of _Industrial Statistics_: 

- mistat (for access to data sets and additional functionality)
- matplotlib 
- numpy 
- pandas 
- scipy 
- statsmodels 
- seaborn 
- pingouin 
- lifelines 
- dtreeviz 
- svglib 
- pwlf 
- pyDOE3
- diversipy 
- pylibkriging 
- inspyred
- pymc
- arviz
- aesara

The notebook [InstallPackages.ipynb](InstallPackages.ipynb) contains the pip command to install the required packages. Note that some of the packages may need to be pinned to specific versions.

If you have a problem with visualizing the decision tree or creating a network graph, follow the [installation instructions for graphviz in the dtreeviz github site](https://github.com/parrt/dtreeviz). On Windows, the problem is usually resolved by adding the path to the graphviz binaries to the PATH system variable.




# Table of contents (with sample excerpts from chapters)

Chapter 1: Introduction to Industrial Statistics (<a href="blogs/Chap001">sample 1</a>)<br>
Chapter 2: Basic Tools and Principles of Process Control (<a href="blogs/Chap002">sample 2</a>)<br>
Chapter 3: Advanced Methods of Statistical Process Control (<a href="blogs/Chap003">sample 3</a>)<br>
Chapter 4: Multivariate Statistical Process Control (<a href="blogs/Chap004">sample 4</a>)<br>
Chapter 5: Classical Design and Analysis of Experiments (<a href="blogs/Chap005">sample 5</a>)<br>
Chapter 6: Quality by Design (<a href="blogs/Chap006">sample 6</a>)<br>
Chapter 7: Computer Experiments (<a href="blogs/Chap007">sample 7</a>)<br>
Chapter 8: Cybermanufacturing and Digital Twins (<a href="blogs/Chap008">sample 8</a>)<br>
Chapter 9: Reliability Analysis (<a href="blogs/Chap009">sample 9</a>)<br>
Chapter 10: Bayesian Reliability Estimation and Prediction (<a href="blogs/Chap010">sample 10</a>)<br>
Chapter 11: Sampling Plans for Batch and Sequential Inspection (<a href="blogs/Chap011">sample 11</a>)<br>

