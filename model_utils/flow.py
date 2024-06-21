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
