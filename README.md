# FlowTest

## Description
A pytorch implementation of Continuous Flow Matching (CNF) for learning purposes.

Inspiration taken heavily from Whalefishin.

Original Repo: https://github.com/Whalefishin/Latent_Flow_Matching_MNIST/tree/main

A CNF can be thought of as some $p_0(x) = \mathcal{N}(0, I)$ with some probability density flow ($p \in [0, 1] \times \mathbb{R}^{d} \rightarrow \mathbb{R}_{>0}$)  this $p$ directs our gaussian prior
to our target distribution $p_1(x_1) \sim q(x_1)$
