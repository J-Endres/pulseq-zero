"""Pulseq-zero demo: optimize TSE refocusing flip angles to minimize SAR.

Builds a 2D TSE sequence (see ``write_tse.py``), simulates a target image
with a fully-180° refocusing train, then optimizes a per-echo flip-angle
vector so the reconstruction stays close to the target (RMS loss on the
magnitude image) while the SAR proxy ``sum(flips**2)`` goes down.

The TSE is built once; the refocusing pulses keep the flip-angle tensor, so
later iterations only refresh ``to_mr0``. Pass ``const_structure=True`` (or
``--fast``) to replay that conversion when the block layout is unchanged.
``main_fast.py`` is that same run. The phase-graph prepass is rebuilt every
fifth iteration and reused in between. The phantom is the default brain
scaled to the 64×64 reconstruction grid.
"""

from time import time

import matplotlib.pyplot as plt
import MRzeroCore as mr0
import numpy as np
import torch

# Force write_tse to use pulseq-zero:
import pulseqzero
import sys
sys.modules["pypulseq"] = pulseqzero
from write_tse import main as build_tse  # noqa: E402


N_ECHO = 16
N_ITER = 30
N_READ = 64
PREPASS_EVERY = 5
TARGET_AMPLITUDE = 0.8
INITIAL_FLIP_DEG = 160.0


def simulate(pp_seq, data, *, const_structure=False, graph=None):
    seq = pp_seq.to_mr0(
        speed_up_by_assuming_const_seq_structure=const_structure,
    )

    # graph[0] is the relaxed z0; the remaining entries match seq repetitions.
    if graph is None or len(graph) != len(seq) + 1:
        graph = mr0.compute_graph(seq, data)
        prepass = True
    else:
        prepass = False

    signal = mr0.execute_graph(graph, seq, data, print_progress=False)
    reco = mr0.reco_adjoint(
        signal, seq.get_kspace(), (N_READ, N_READ, 1), (0.256, 0.256, 1)
    )
    return reco, graph, prepass


def main(*, const_structure=False):
    data = mr0.util.load_phantom(size=(N_READ, N_READ)).build()
    print('TSE built once; later iterations only refresh to_mr0')
    if const_structure:
        print('to_mr0: reusing conversion when the block layout is unchanged')

    # Target image: its own sequence, so it does not alias the optimized flips.
    with torch.no_grad():
        target_seq = build_tse(refoc_flips=torch.full((N_ECHO,), np.pi))
        target = TARGET_AMPLITUDE * simulate(
            target_seq, data, const_structure=const_structure
        )[0]

    # The refocusing pulses store views of this tensor. Adam updates them in place.
    flips = torch.full((N_ECHO,), INITIAL_FLIP_DEG * np.pi / 180, requires_grad=True)
    optimizer = torch.optim.Adam([flips], lr=0.02)
    pp_seq = build_tse(refoc_flips=flips)

    start = simulate(pp_seq, data, const_structure=const_structure)[0].detach()
    RMS_WEIGHT = 1 / ((start.abs() - target.abs()) ** 2).mean().sqrt()
    SAR_WEIGHT = 1 / (flips.detach() ** 2).sum()

    data_hist = []
    sar_hist = []
    loss_hist = []
    flip_hist = []

    t0 = time()
    graph = None
    for i in range(N_ITER):
        t_iter = time()
        optimizer.zero_grad()
        if i % PREPASS_EVERY == 0:
            graph = None
        reco, graph, prepass = simulate(
            pp_seq, data, const_structure=const_structure, graph=graph
        )

        data_loss = ((reco.abs() - target.abs()) ** 2).mean().sqrt()
        sar_loss = (flips ** 2).sum()
        loss = RMS_WEIGHT * data_loss + SAR_WEIGHT * sar_loss

        loss.backward()
        optimizer.step()

        data_hist.append(data_loss.item())
        sar_hist.append(sar_loss.item())
        loss_hist.append(loss.item())
        flip_hist.append(flips.detach().clone().numpy())
        print(
            f'{i + 1}/{N_ITER}: data={data_loss.item():.4f}, '
            f'SAR={sar_loss.item():.2f}, total={loss.item():.4f}, '
            f'{"prepass" if prepass else "graph reused"}, '
            f'{time() - t_iter:.2f} s'
        )
    print(f'Optimization took {time() - t0:.1f} s')

    best_idx = int(np.argmin(loss_hist))
    with torch.no_grad():
        flips.copy_(torch.as_tensor(flip_hist[best_idx], dtype=flips.dtype))
    best = simulate(pp_seq, data, const_structure=const_structure)[0].detach()

    plt.figure(figsize=(10, 8), dpi=120)
    plt.subplot(2, 2, 1)
    plt.title('Target (all 180°)')
    mr0.util.imshow(target.abs(), vmin=0, cmap='gray')
    plt.colorbar()
    plt.subplot(2, 2, 2)
    plt.title(f'Best (iter {best_idx + 1})')
    mr0.util.imshow(best.abs(), vmin=0, cmap='gray')
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.title('Loss')
    plt.plot([RMS_WEIGHT * d for d in data_hist], label='RMS image loss')
    plt.plot([SAR_WEIGHT * s for s in sar_hist], label='SAR loss')
    plt.plot(loss_hist, label='total', linestyle='--')
    plt.xlabel('iteration')
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.title('Refocusing flip angles')
    cmap = plt.get_cmap('viridis')
    for i, flips in enumerate(flip_hist):
        plt.plot(
            [f * 180 / np.pi for f in flips],
            color=cmap(i / (N_ITER - 1)),
        )
    plt.xlabel('iteration')
    plt.ylabel('Flip [°]')
    plt.grid()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--fast',
        action='store_true',
        help='reuse to_mr0() while the TSE block layout stays fixed',
    )
    main(const_structure=parser.parse_args().fast)
