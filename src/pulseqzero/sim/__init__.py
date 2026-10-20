"""Simulators for ``seq.simulate(sim)``.

A simulator turns an MR-zero sequence (what ``seq.to_mr0()`` builds) into the
measured signal as a differentiable torch tensor of shape ``[ADC samples, 1]``,
so that the same pulseq-zero sequence function can be optimized with different
simulators:

- :class:`MR0` - MR-zero's PDG, locally
- :class:`Rapisim` - rapisim (segmented bifti phantoms), a local binary or an
  MCP gateway. Needs ``pip install pulseqzero[rapisim]``.

    sim = pp.sim.MR0(sim_data)                  # or: pp.sim.Rapisim("phantom.zip")
    signal = my_sequence(flips).simulate(sim)
    loss = my_loss(signal)
    loss.backward()
"""

from typing import Protocol

import torch
from MRzeroCore import Sequence as _Mr0Sequence

from .mr0 import MR0

__all__ = ["MR0", "Rapisim", "Simulator"]


class Simulator(Protocol):
    def simulate(self, seq: _Mr0Sequence) -> torch.Tensor:
        """Complex signal ``[ADC samples, 1]`` with gradients of ``seq``"""
        ...


def __getattr__(name):
    # fastmcp and friends are only needed (and imported) when rapisim is used
    if name == "Rapisim":
        from .rapisim import Rapisim

        return Rapisim
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
