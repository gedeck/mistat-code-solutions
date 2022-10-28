# Loss Functions and Bayes Estimators


In order to define Bayes estimators we must first specify a
**loss function**, $L(\hat{\mathbf \theta},\mathbf \theta)$, which represents the cost
involved in using the estimate $\hat{\mathbf \theta}$ when the true value is
$\mathbf \theta$.  Often this loss is taken to be a function of the distance
between the estimate and the true value, i.e., $|\hat{\mathbf \theta} -
\mathbf \theta|$.  In such cases, the loss function is written as

$L(\hat{\mathbf \theta},\mathbf \theta) = W(|\hat{\mathbf \theta} - \mathbf \theta|).$

Examples of such loss functions are

- Squared-error loss: 
    $W(|\hat{\mathbf \theta} - \mathbf \theta|) = (\hat{\mathbf \theta} - \mathbf \theta)^2$,
- Absolute-error loss:
    $W(|\hat{\mathbf \theta} - \mathbf \theta|) = |\hat{\mathbf \theta} - \mathbf \theta|$.

The loss function does not have to be symmetric.  For example, we may
consider the function

$L(\hat\theta,\theta) = \alpha(\theta-\hat\theta)$, if $\hat\theta \leq \theta$

$L(\hat\theta,\theta) = \beta(\hat\theta-\theta)$,
if $\hat\theta > \theta$

where $\alpha$ and $\beta$ are some positive constants.

The **Bayes estimator** of $\mathbf \theta$, with respect to a loss function
$L(\hat{\mathbf \theta},\mathbf \theta)$, is defined as the value of
$\hat{\mathbf \theta}$ which minimizes the **posterior risk**, given $x$,
where the posterior risk is the expected loss with respect to the posterior
distribution.  For example, suppose that the p.d.f. of $X$ depends on several
parameters $\theta_1,\dots,\theta_k$, but we wish to derive a Bayes
estimator of $\theta_1$ with respect to the squared-error loss function.  We
consider the marginal posterior p.d.f. of $\theta_1$, given $\mathbf x$,
$h(\theta_1\mid x)$.  The posterior risk is

$R(\hat\theta_1,\mathbf{x})    = \int(\hat\theta_1 - \theta_1)^2 h(\theta_1\mid\mathbf{x}) \,d\theta_1.$

It is easily shown that the value of $\hat\theta_1$ which minimizes the
posterior risk $R(\hat\theta_1,\mathbf x)$ is the **posterior expectation**
of $\theta_1$:

$E\{\theta_1\mid\mathbf{x}\}
    = \int \theta_1h(\theta_1\mid \mathbf{x}) \,d\theta_1.$

If the loss function is
$L(\hat\theta_1,\hat\theta) = |\hat\theta_1 - \theta_1|$,
the Bayes estimator of $\theta_1$ is the **median** of the
posterior distribution of $\theta_1$ given $\mathbf x$.
