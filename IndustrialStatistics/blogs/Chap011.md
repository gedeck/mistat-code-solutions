# Double-Sampling Plans for Attributes

A double sampling plan for attributes is a two-stage procedure.  In
the first stage, a random sample of size $n_1$ is drawn, without
replacement, from the lot.  Let $X_1$ denote the number of defective
items in this first stage sample.  Then the rules for the second stage
are the following:  if $X_1\leq c_1$, sampling terminates and
the lot is accepted; if $X_1 \geq c_2$, sampling terminates and the lot
is rejected; if $X_1$ is between $c_1$ and $c_2$, a second stage random
sample, of size $n_2$, is drawn, without replacement, from the remaining
items in the lot.  Let $X_2$ be the number of defective items in this
second-stage sample.  Then, if $X_1 + X_2 \leq c_3$, the lot is accepted
and if $X_1 + X_2 > c_3$ the lot is rejected.


Generally, if there are
very few (or very many) defective items in the lot, the decision
to accept or reject the lot can be reached after the first stage of
sampling.  Since the first
stage samples are smaller than those needed in a single stage sampling
a considerable saving in inspection cost may be attained.

In this type of sampling plan, there are five parameters to
select, namely, $n_1$, $n_2$, $c_1$, $c_2$ and $c_3$.  Variations in the
values of these parameters affect the operating characteristics of the
procedure, as well as the expected number of observations required
(i.e. the total sample size).  Theoretically, we could determine the
optimal values of these five parameters by imposing five independent
requirements on the OC function and the function of expected total sample
size, called the **Average Sample Number** or
**ASN-function**, at various values of $p$.  However,
to simplify this procedure, it is common practice to set $n_2 = 2n_1$
and $c_2 = c_3 = 3c_1$.  This reduces the problem to that of selecting
just $n_1$ and $c_1$.  Every such selection will specify a particular
double-sampling plan.  For example, if the lot consists of $N=150$
items, and we choose a plan with $n_1 = 20$, $n_2 = 40$, $c_1 = 2$, 
$c_2 = c_3 = 6$, we will achieve certain properties.  On the other hand, if
we set $n_1 = 20$, $n_2 = 40$, $c_1 = 1$, $c_2 = c_3 = 3$, the plan will
have different properties.

The formula of the OC function associated with a double-sampling plan
$(n_1,n_2,c_1,c_2,c_3)$ is

$\text{OC}(p)      = H(c_1;N,M_p,n_1) +   \sum\limits_{j=c_1+1}^{c_2-1} h(j;N,M_p,n_1)H(c_3-j;N-n_1,M_p-j,n_2)$
      
where $M_p = [Np]$.  Obviously, we must have $c_2\geq c_1 + 2$, for
otherwise the plan is a single-stage plan.