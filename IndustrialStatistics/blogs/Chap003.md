# Bayesian Detection

The Bayesian approach to the problem of detecting changes in
distributions, can be described in the following terms.  Suppose that we
decide to monitor the stability of a process with a statistic $T$,
having a distribution with p.d.f. $f_T(t;\theta)$, where $\theta$
designates the parameters on which the distribution depends (process
mean, variance, etc.).  The statistic $T$ could be the mean, $\bar X$,
of a random sample of size $n$; the sample standard-deviation, $S$, or
the proportion defectives in the sample.  A sample of size $n$ is drawn
from the process at predetermined epochs.  Let $T_i$ $(i = 1,2,\cdots)$
denote the monitoring statistic at the $i$-th epoch.  Suppose that $m$
such samples were drawn and that the statistics $T_1,T_2,\cdots,T_m$ are
independent.  Let $\tau = 0,1,2,\cdots$ denote the location of the point
of change in the process parameter $\theta_0$, to $\theta_1 =
\theta_0 + \Delta$.  $\tau$ is called the {\bf change-point} of
$\theta_0$.  The event $\{\tau = 0\}$ signifies that all the $n$
samples have been drawn after the change-point.  The event $\{\tau
=i\}$, for $i = 1,\cdots,m-1$, signifies that the change-point occurred
between the $i$-th and $(i+1)$-st sampling epoch.  Finally, the event
$\{\tau = m^+\}$ signifies that the change-point has not occurred before
the first $m$ sampling epochs.