# Cybermanufacturing Analytics

The third industrial revolution, introduced in Chapter 1,
involved embedded systems such as sensors and programmable logic controllers to
achieve automation in manufacturing. With the extensive use of embedded systems,
the third industrial revolution significantly improved throughput, efficiency,
and product quality in the manufacturing industry, while reducing reliance on
manual operations. This opened the era of ``smart manufacturing", 
which is utilizing sensor data to enable data-driven decision-making and equipment managed
by numerical controllers (Kenett et al., 2018b). Cybermanufacturing
is the next phase in industry, leveraging advances in manufacturing technologies,
sensors and analytics (Kenett and Redman, 2019; Kang et al., 2021b),
Suppose that $X$ is a set of process variables (scalar, vector, or matrix) and
$Y$ are performance variables (scalar, vector, or matrix). Modeling consists of
identifying the relationship $f$, such that $Y=f(X)$, or an approximation $f'$,
such that $Y=f'(X)+ \varepsilon$ where $\varepsilon$ is an error term. Modeling
and analysis provide the foundation for 
process monitoring, 
root-cause diagnosis, and control.

Modeling can be based on physical principles such as thermodynamics, fluid
mechanics and dynamical systems and involve deriving exact solutions for
ordinary or partial differential equations (ODE/PDEs)
or solving an approximation of ODE/PDEs via numerical methods such as finite element analysis. 
It aims at
deriving the exact relationship $f$, or its approximation (or discretization)
$\tilde{f}$ , between process variables $X$ and performance variables $Y$ (Chinesta, 2019; Dattner, 2021). When
there is a significant gap between the assumption of physical principles and
the actual manufacturing conditions, empirical models derived from statistically
designed of experiments (DOE).

Smart manufacturing exploits advances in sensing technologies, with in situ process
variables, having an impact on modeling and analysis. This enables online updates
and provides improvements in real time product quality and process efficiency. Another advance has been the development of
soft sensors which provide online surrogates to laboratory tests.

There are two important challenges in manufacturing analytics: 1) data integration, also called data fusion and
2) development and deployment of analytic methods. Data fusion refers to the
methods of integrating different models and data sources or different types
of datasets.
Machine learning
refers to the building a mathematical model
based on data, such that the model can make predictions without being explicitly
programmed to do so.
In particular, deep neural networks have shown superior performance in modeling
complex manufacturing processes. Data fusion
and data analytics play crucial roles in cybermanufacturing as they provide
accurate approximations of the relationship between process variable $X$ and
performance variables $Y$, denoted as $f'$, by utilizing experimental and/or
simulation data. In predictive analytics, models are validated by splitting
the data into training and validation sets. The structure of data needs to be 
accounted in the cross validation. An approach called
befitting cross validation (BCV)
is proposed in Kenett et al. 2022.

