#!/usr/bin/env python

### Imports

# ML
import torch
import torch.nn as nn

# Math & Vectors
import numpy as np
import math
from zuko.utils import odeint



class FlowMatching(nn.Module):
    def __init__(
        self,
        v_t: nn.Module,
        flow: nn.Module,
        d: int,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
        ):
        super().__init__()
        """Overall Architecture for generating a Flow Matching flow $p$ for going
        from p_0 -> p_1. 

        PARAMS
        ------
        v_t: nn.Module
            Time-dependent Vector Field Model
        flow: nn.Module
            Optimal Transport Flow Matching Implementation Conditioanl approach.
        d: int
            Latent Space Representation 
        """
        # Store Passed arguments for later use
        self.v_t = v_t
        self.flow = flow
        self.d = d
        self.device = device

    def sample(self, b: int):
        """Randomly sample from a guassian and decode X_0 
        across our vector field to X_1

        PARAMS
        ------
        b: int
            Batch size for random sample creation
        """
        # Sample form the gaussian prior N(0, I) which is isotropic
        x_0 = torch.randn(b, self.d, device=self.device)

        # Using the time dependent vector field integrate over it to generate get
        # to our p_1 distribution
        x_1 = v_t.decode(x_0)

        return x_1

    def get_loss(self, x: torch.Tensor):
        """Calculate whatever the desired flow we are using's loss term

        PARAMS
        ------
        x: Tensor (B, D)
            Flow generated x

        RETURNS
        -------
        loss: Tensor (1)
            Loss term calculated using the desired flow
        """
        return self.flow.loss(self.v_t, x)

class CondVF(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        """Conditional Vector Field that is a neural network
        to establish the probability density path 

        PARAMS
        ------
        net: nn.Module
            Neural Network for some VF
        """
        # Set the params
        self.net = net

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        """Taking some t time step forward through the time 
        t \in [0,1]

        PARAMS
        ------
        t: Tensor (float)
            Time step
        x: Tensor (B, D)
            Sample taken from the p_t(x) distribution
        """
        # Return the current time step in the vector fiedl with the given
        # x
        return self.net(t, x)

    def wrapper(self, t: torch.Tensor, x: torch.Tensor):
        """Wrapper to update the time step portion 

        PARAMS
        ------
        t: Tensor (float)
            Time step
        x: Tensor (B, D)
            Sample taken from the p_t(x) distribution
        """
        # generate column vector of size (D)
        t = t * torch.ones(x.size(-1), device=x.device)
        return self(t, x)

    def decode_t0_t1(
        self,
        x_0: torch.Tensor,
        t0: float,
        t1: float,
        ):
        """Perform ODE steps from 0 -> 1 along x_0

        PARAMS
        ------
        x_0: Tensor (B, D)
            The initial gaussian prior distribution
        t0: float
            Starting time point
        t1: float
            End time point
        """
        return odeint(self.wrapper, x_0, t0, t1, self.parameters())

    def decode(self, x_0: torch.Tensor):
        """Generation Function. Ssample from x_0 ~ p_0,
        then we calc x_1 by integrating v_theta from t 0 -> 1
        with x_0

        PARAMS
        ------
        x_0: Tensor (B, D)
            The initial gaussian prior distribution
        """
        return odeint(self.wrapper, x_0, 0., 1.)





