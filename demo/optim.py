"""Optimize TSE refocusing flip angles - with MR-zero or rapisim as simulator.

``write_tse.py`` is a plain PyPulseq script. Here it is run through pulseq-zero
and its refocusing flip angles are optimized: the image has to stay close to
the one of an all-180° train (RMS loss on the magnitude), while the SAR proxy
``sum(flips**2)`` goes down. The optimization is the same for both simulators,
``--sim`` only chooses which one computes the signal and its gradients:

    python optim.py --phantom subj04-3T-2mm-tra.zip --sim mr0
    python optim.py --phantom subj04-3T-2mm-tra.zip --sim rapisim --rapisim-bin ~/tool-rapisim/target/release/rapisim
    python optim.py --phantom subj04-3T-2mm-tra.zip --sim rapisim --url https://mrx-org--mcp-gateway-func.modal.run

The phantom is a bifti file (https://github.com/mrx-org/bifti-phantoms), which
both simulators read. rapisim needs ``--ignore-b1`` for phantoms with a B1+ map
(gradients with spatially varying B1+ are not implemented yet); it is applied to
MR-zero as well so that both see the same phantom.

rapisim needs ``pip install pulseqzero[rapisim]``.
"""

import argparse
import sys
import tempfile
import zipfile
from pathlib import Path
from time import time

import MRzeroCore as mr0
import numpy as np
import torch

# Force write_tse to use pulseq-zero:
import pulseqzero as pp

sys.modules["pypulseq"] = pp
from write_tse import main as build_tse  # noqa: E402

N_ECHO = 16
TARGET_AMPLITUDE = 0.8
INITIAL_FLIP_DEG = 160.0


def make_simulator(args):
    if args.sim == "rapisim":
        return pp.sim.Rapisim(
            args.phantom,
            url=args.url,
            command=[args.rapisim_bin] if args.rapisim_bin else None,
            ignore_b1=args.ignore_b1,
        )

    # MR-zero reads the bifti phantom from the unpacked archive
    unpacked = Path(tempfile.mkdtemp(prefix="phantom-"))
    with zipfile.ZipFile(args.phantom) as archive:
        archive.extractall(unpacked)
    data = mr0.phantom.tissue_dict.TissueDict.load(next(unpacked.glob("*.json"))).build()
    if args.ignore_b1:
        data.B1 = torch.ones_like(data.B1)
    # MR-zero's default thresholds drop weak states, which changes the signal
    # and - more so - its gradients for long echo trains. Tighter ones agree
    # with rapisim, which by default keeps everything above 1e-6.
    return pp.sim.MR0(
        data,
        graph_kwargs=dict(max_state_count=100_000, min_state_mag=1e-6),
        execute_kwargs=dict(min_emitted_signal=1e-6, min_latent_signal=1e-6),
    )


def reconstruct(flips, sim):
    seq = build_tse(refoc_flips=flips, check_timing=False)
    signal = seq.simulate(sim)
    kspace = seq.to_mr0().get_kspace()
    return mr0.reco_adjoint(signal, kspace, (64, 64, 1), (0.256, 0.256, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phantom", required=True, help="bifti phantom (.zip)")
    parser.add_argument("--sim", choices=["mr0", "rapisim"], default="mr0")
    parser.add_argument("--url", help="rapisim: MCP gateway instead of a local binary")
    parser.add_argument("--rapisim-bin", help="rapisim: local binary (default: $RAPISIM_BIN or 'rapisim')")
    parser.add_argument("--ignore-b1", action="store_true", help="ignore the B1+ map of the phantom")
    parser.add_argument("--n-iter", type=int, default=30)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    sim = make_simulator(args)
    build_tse(check_timing=True)  # same timing for every flip angle train

    # Target image: the full-180° refocusing train.
    with torch.no_grad():
        target = TARGET_AMPLITUDE * reconstruct(torch.full((N_ECHO,), np.pi), sim)

    flips = torch.full((N_ECHO,), INITIAL_FLIP_DEG * np.pi / 180, requires_grad=True)
    optimizer = torch.optim.Adam([flips], lr=0.02)

    start = reconstruct(flips, sim).detach()
    rms_weight = 1 / ((start.abs() - target.abs()) ** 2).mean().sqrt()
    sar_weight = 1 / (flips.detach() ** 2).sum()

    history = []
    t0 = time()
    for i in range(args.n_iter):
        t_iter = time()
        optimizer.zero_grad()
        reco = reconstruct(flips, sim)

        data_loss = ((reco.abs() - target.abs()) ** 2).mean().sqrt()
        sar_loss = (flips**2).sum()
        loss = rms_weight * data_loss + sar_weight * sar_loss
        loss.backward()
        optimizer.step()

        history.append((data_loss.item(), sar_loss.item(), loss.item(), flips.detach().clone().numpy()))
        print(
            f"{i + 1}/{args.n_iter} [{args.sim}]: data={data_loss.item():.4f}, "
            f"SAR={sar_loss.item():.2f}, total={loss.item():.4f}, {time() - t_iter:.2f} s"
        )
    print(f"Optimization took {time() - t0:.1f} s")

    if args.plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4), dpi=120)
        plt.subplot(1, 2, 1)
        plt.title("Loss")
        plt.plot([rms_weight * h[0] for h in history], label="RMS image loss")
        plt.plot([sar_weight * h[1] for h in history], label="SAR loss")
        plt.plot([h[2] for h in history], label="total", linestyle="--")
        plt.xlabel("iteration")
        plt.legend()
        plt.grid()
        plt.subplot(1, 2, 2)
        plt.title("Refocusing flip angles")
        cmap = plt.get_cmap("viridis")
        for i, h in enumerate(history):
            plt.plot(np.degrees(h[3]), color=cmap(i / max(len(history) - 1, 1)))
        plt.xlabel("echo")
        plt.ylabel("Flip [°]")
        plt.grid()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
