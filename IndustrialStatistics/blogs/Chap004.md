# Multivariate Control Charts Scenarios

The Hotelling $T^2$ chart 
plots the $T^2$ statistic, which is the squared standardized distance of a
vector from a target point.
Values of $T^2$ represent equidistant vectors along a multidimensional ellipse
centered at the target vector point. The chart has an upper control limit
$(UCL)$ determined by the $F$ distribution.
Points exceeding $UCL$ are regarded as an out-of-control signal. The
charted $T^2$ statistic is a function that reduces multivariate observations
into a single value while accounting for the covariance matrix.
Out-of-control signals on the $T^2$ chart trigger an
investigation to uncover the causes for the signal.

The setup of an MSPC chart is performed by a process capability study.
The process capability study period is sometimes referred to as
**phase I**. The ongoing control using control limits determined in
phase I is then called **phase II**. The distinction between these
two phases is important.

In setting MSPC charts, one meets several alternative scenarios derived
from the characteristics of the reference
sample and the appropriate control procedure. These include:

1. internally derived target
2. using an external reference sample
3. externally assigned target
4. measurements units considered as batches.

## Internally Derived Target

Internally derived targets are a typical scenario for process capability
studies. The parameters to be estimated include the vector of process means,
the process covariance matrix, and the control limit for the control chart.
Consider a process capability study with a base sample of size $n$ of
$p$-dimensional observations, $X_1,X_2,\dots,X_n$. When
the data are grouped and $k$ subgroups of observations of size $m$ are
being monitored, $n = km$, the covariance matrix estimator, $S_p$ can be
calculated as the pooled covariances of the subgroups. 

## Using an External Reference Sample

We now assume
that we have a _reference_ sample $X_1,\dots,X_n$ of $F$ from an
in-control period. To control the quality of the produced items,
multivariate data is monitored for potential change in the distribution
of $\mathbf X$, by sequentially collecting and analyzing the observations
$X_i$. At some time $t = n + k$, $k$ time units after $n$, the process may
run out of control and the distribution of the
$X_i$'s changes to $G$. Our aim is to detect, in phase II, the change
in the distribution of subsequent observations $X_{n+k}$, $k \geq 1$, as
quickly as possible, subject to a bound $\alpha \in (0, 1)$ on the
probability of raising a false alarm at each time point $t = n + k$ (that is,
the probability of erroneously deciding that the distribution of
$X_{n+k}$ is not $F$).  The reference sample $X_1,\dots,X_n$ does not
incorporate the observations $X_{n+k}$ taken after the _reference_ stage, even
if no alarm is raised, so that the rule is conditional only on the reference sample.

When the data in phase II is grouped, and the reference sample from historical
data includes $k$ subgroups of observations of size $m$, $n = km$, with the
covariance matrix estimator $S_p$ calculated as the pooled covariances
of the subgroups.

## Externally Assigned Target

If all parameters of the underlying multivariate distribution are
known and externally assigned, the $T^2$ value for a
single multivariate observation of dimension $p$ is computed as

$T^2 = ({\mathbf Y}-{\mathbf \mu})'\Sigma^{-1}({\mathbf Y}-{\mathbf \mu})$

where $\mathbf \mu$ and $\mathbf \Sigma$ are the expected value and covariance
matrix, respectively.

The probability distribution of the $T^2$ statistic is a $\chi^2$
distribution with $p$ degrees of freedom. Accordingly, the
0.95 $UCL$ for $T^2$ is $UCL= \chi^2_{\nu,.95}$. When the data are
grouped in subgroups of size $m$, and both $\mathbf \mu$ and $\mathbf \Sigma$
are known, the $T^2$ value of the mean vector $\bar Y$ is
$T^2 = m(\bar Y-{\mathbf \mu})'{\mathbf \Sigma}^{-1}(\bar Y-{\mathbf \mu})$
with the same $UCL$ as above.

## Measurement Units Considered as Batches


In the semiconductor industry, production is typically organized in
batches or production lots. In such cases, the quality-control process
can be performed either at the completion of the batch or sequentially,
in a curtailed inspection, aiming at reaching a decision as soon as
possible. When the quality-control method used is reaching a decision
at the completion of the process, the possible outcomes are
(a) determine the production process to be in statistical control and
accept the batch or (b) stop the production flow because of a signal
that the process is out of control. On the other hand, in a curtailed
inspection, based on a statistical stopping rule, the results from the
first few items tested may suffice to stop the process prior to the
batch completion.

Consider a batch of size $n$, with the tested items
${\mathbf Y}_1,\dots,{\mathbf Y}_n$. The curtailed inspection
tests the items sequentially.
Assume that the targets are specified, either externally assigned,
or from a reference sample or batch. With respect to those targets,
let $V_i = 1$ if the $T^2$ of the ordered $i$-th observation exceeds the
critical value $\kappa$ and $V_i = 0$, otherwise. For the $i$-th
observation, the process is considered to be in control if for a prespecified
$P$, say $P = 0.95$, $Pr(V_i = 0)\geq P$. Obviously, the inspection
will be curtailed only at an observation $i$
for which $V_i = 1$ (not necessarily the first).