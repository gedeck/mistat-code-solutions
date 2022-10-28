# Reliability Demonstration}

Reliability demonstration is a procedure for testing whether the reliability
of a given device (system) at a certain age is sufficiently high.  More
precisely, a time point $t_0$ and a desired reliability $R_0$ are
specified, and we wish to test whether the reliability of the device at age
$t_0$, $R(t_0)$, satisfies the requirement that $R(t_0) \geq R_0$.  If the
life distribution of the device is completely known, including all
parameters, there is no problem of reliability demonstration - one computes
$R(t_0)$ exactly and determines whether $R(t_0) \geq R_0$.  If, as is
generally the case, either the life distribution or its parameters are
unknown, then the problem of reliability demonstration is that of obtaining
suitable data and using them to test the statistical hypothesis that 
$R(t_0) \geq R_0$ versus the alternative that $R(t_0) < R_0$.  Thus, the theory of
testing statistical hypotheses provides the tools for reliability
demonstration.  In the present section we review some of the basic notions
of hypothesis testing as they pertain to reliability demonstration.

We develop several tests of interest in
reliability demonstration.  We remark here that procedures for obtaining
confidence intervals for $R(t_0)$, which were discussed in the previous
sections, can be used to test hypotheses.  Specifically, the procedure
involves computing the upper confidence limit of a $(1-2\alpha)$-level
confidence interval for $R(t_0)$ and comparing it with the value $R_0$.  If
the upper confidence limit exceeds $R_0$ then the null hypothesis 
$H_0 : R(t_0) > R_0$ is
accepted, otherwise it is rejected.  This test will have a significance
level of $\alpha$.

For example, if the specification of the reliability at age $t = t_0$ is
$R = .75$ and the confidence interval for $R(t_0)$, at level of confidence
$\gamma = .90$, is $(.80,.85)$, the hypothesis $H_0$ can be immediately
accepted at a level of significance of $\alpha = (1-\gamma)/2=.05$.  There is
a duality between procedures for testing hypotheses and for confidence
intervals.