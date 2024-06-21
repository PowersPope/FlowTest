# FlowTest

## Description
A pytorch implementation of Continuous Flow Matching (CNF) for learning purposes.

A CNF can be thought of as some $p_0(x) = \mathcal{N}(0, I)$ with some probability density flow ($p \in [0, 1] \cross \mathbb{R}^{d} -> \mathbb{R}_{\ge 0}$)  this $p$ directs our gaussian prior
to $p_1(x_1) ~ q(x_1)$
