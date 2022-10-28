# Blocking and Randomization

Blocking and randomization are used in planning of experiments,
in order to increase the precision of the outcome and ensure
the validity of the inference.  Blocking is used to reduce errors.  A
block is a portion of the experimental material that is expected to be
more homogeneous than the whole aggregate.  For example, if the
experiment is designed to test the effect of polyester coating of
electronic circuits on their current output, the variability between
circuits could be considerably bigger than the effect of the coating on
the current output.  In order to reduce this component of variance, one
can block by circuit.  Each circuit will be tested under two treatments:
no-coating and coating.  We first test the current output of a circuit
without coating.  Later we coat the circuit, and test again.  Such a
comparison of before and after a treatment, of the same units, is called **paired-comparison**.

Another example of blocking is the boy's shoes examples of Box et al. (2005). Two kinds of shoe soles' materials
are to be tested by fixing the soles on $n$ pairs of boys' shoes, and
measuring the amount of wear of the soles after a period of actively
wearing the shoes.  Since there is high variability between activity of
boys, if $m$ pairs will be with soles of one type and the rest of the
other, it will not be clear whether any difference that might be
observed in the degree of wearout is due to differences between the
characteristics of the sole material or to the differences between the
boys.  By blocking by pair of shoes, we can reduce much of the
variability.  Each pair of shoes is assigned the two types of soles.
The comparison within each block is free of the variability between
boys.  Furthermore, since boys use their right or left foot differently,
one should assign the type of soles to the left or right shoes at
random.  Thus, the treatments (two types of soles) are assigned within
each block at random.

Other examples of blocks could be machines, shifts of production, days
of the week, operators, etc.

Generally, if there are $t$ treatments to compare, and $b$ blocks, and
if all $t$ treatments can be performed within a single block, we assign
all the $t$ treatments to each block.  The order of applying the
treatments within each block should be **randomized**.  Such a design
is called a **randomized complete block design**.  We will see later
how a proper analysis of the yield can validly test for the effects of
the treatments.

If not all treatments can be applied within each block it is desirable
to assign treatments to blocks in some balanced fashion.  Such designs, to be
discussed later, are called **balanced incomplete block designs**
(BIBD).

Randomization within each block is important also to validate the
assumption that the error components in the statistical model are
independent.  This assumption may not be valid if treatments are not
assigned at random to the experimental units within each block.
