# Computer Experiments

Computer experiments consist of a number of runs of a simulation code and
factors level combinations correspond to a subset of code inputs. By
considering computer runs as a realization of a stochastic process, a
statistical framework is available both to design the experimental
points and to analyze the responses. A major difference between computer
numerical experiments and physical experiments is the logical difficulty
in specifying a source of randomness for computer experiments.

The complexity of the mathematical models implemented in the computer
programs can, by themselves, build equivalent sources of random noise. In
complex code, a number of parameters and model choices gives the user
many degrees of freedom that provide potential 
variability to the outputs
of the simulation. Examples include different solution algorithms
(i.e. implicit or explicit methods for solving differential systems),
approach to discretization intervals and convergence thresholds for
iterative techniques. In this very sense, an experimental error can be
considered in the statistical analysis of computer experiments. The
nature of the experimental error in both physical and simulated
experiments, is our ignorance about the phenomena and the intrinsic
error of the measurements. Real world phenomena are too complex for
the experimenter to keep under control by specifying all the
factors affecting the response of the experiment. Even if it were
possible, the physical measuring instruments,
being not ideal, introduce problems of accuracy and precision. Perfect knowledge would be
achieved in physical experiments only if all experimental factors can be
controlled and measured without any error. Similar phenomena occur in
computer experiments. A complex code has several degrees of freedom in its
implementation that are not controllable.

A specific case where randomness is introduced to computer experiments
consists of the popular Finite Element Method (FEM) programs. 
These models
are applied in a variety of technical sectors such as electromagnetics,
fluid-dynamics, mechanical design, civil design. The FEM mathematical
models are based on a system of partial differential equations defined on
a time-space domain for handling linear or non-linear, steady-state or dynamic
problems. FEM software can deal with very complex shapes as well as
with a variety of material properties, boundary conditions and loads.
Applications of FEM simulations require subdivision of the space-domain
into a finite number of subdomains, named finite elements, and solving
the partial differential system within each subdomain, letting
the field-function to be continuous on its border.

Experienced FEM practitioners are aware that results of complex simulations
(complex shapes, non-linear constitutive equations, dynamic problems,
contacts among different bodies, etc.) can be sensitive to the choice of
manifold model parameters. Reliability of FEM results is a critical issue
for the single simulation and even more for a series of computer experiment.
The model parameters used in the discretization of 
the geometry are likely to
be the most critical. Discretization of the model geometry consists in a
set of points (nodes of the mesh) and a set of elements (two-dimensional
patches or three-dimensional volumes) defined through a connectivity matrix
whose rows list the nodes enclosing the elements. Many 
degrees of freedom
are available to the analyst when defining a mesh on a given model.
Changing the location and the number of nodes, the shape and the number
of elements an infinity of meshes are obtained. Any of them will produce
different results. How can we model the effects of different meshes on
the experimental response? In principle, the finer the discretization
the better the approximation of numerical solution, even if numerical
instabilities may occur using very refined meshes. Within a
reasonable approximation, a systematical effect can be assigned to
mesh density; it would be a fixed-effect factor if it is included in
the experiment. A number of topological features (node locations,
element shape), which the analyst has no meaningful effect to assign
to, are generators of random variability. One can assume that they are
randomized along the experiment or random-effect factors with nuisance
variance components if they are included as experimental factors.
