"""Benchmark of ``seq.simulate(sim)``: variable flip angle TSE optimization.

Runs ``--n-iter`` Adam iterations on the refocusing flip angles of
``write_tse.py`` (as in ``optim.py``) with one simulator and prints the mean time
per iteration of every step. With rapisim the time spent in the MCP client
(upload, call, download) is listed too; compare it with the compute time the
rapisim server logs ("stage done" lines on stderr) to get the transport overhead.

    python benchmark_simulate.py --sim rapisim --phantom subj04-3T-2mm-tra.zip --rapisim-bin target/release/rapisim 2> server.log
    python benchmark_simulate.py --sim mr0 --phantom subj04-3T-2mm-tra.zip

``--sim mr0`` uses tight state thresholds (see ``optim.py``), ``--sim mr0-default``
MR-zero's defaults.
"""

import argparse
import json
import sys
import tempfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import MRzeroCore as mr0
import numpy as np
import torch

import pulseqzero as pp

sys.modules["pypulseq"] = pp
from write_tse import main as build_tse  # noqa: E402

N_ECHO = 16


def make_simulator(args, timings, phase):
    if args.sim == "rapisim":
        from pulseqzero.sim import rapisim

        sim = pp.sim.Rapisim(
            args.phantom,
            url=args.url,
            command=[args.rapisim_bin] if args.rapisim_bin else None,
            ignore_b1=True,
        )
        # time the MCP client; `call` includes the `get` of its result
        for name in ("put", "get", "call"):
            original = getattr(rapisim._Connection, name)

            def timed(self, *a, _original=original, _name=name, **k):
                start = time.perf_counter()
                try:
                    return _original(self, *a, **k)
                finally:
                    timings[f"{phase[0]}_{_name}"] += time.perf_counter() - start

            setattr(rapisim._Connection, name, timed)
        return sim

    unpacked = Path(tempfile.mkdtemp(prefix="phantom-"))
    with zipfile.ZipFile(args.phantom) as archive:
        archive.extractall(unpacked)
    data = mr0.phantom.tissue_dict.TissueDict.load(next(unpacked.glob("*.json"))).build()
    data.B1 = torch.ones_like(data.B1)  # as ignore_b1 for rapisim
    if args.sim == "mr0-default":
        return pp.sim.MR0(data)
    return pp.sim.MR0(
        data,
        graph_kwargs=dict(max_state_count=100_000, min_state_mag=1e-6),
        execute_kwargs=dict(min_emitted_signal=1e-6, min_latent_signal=1e-6),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim", choices=["rapisim", "mr0", "mr0-default"], required=True)
    parser.add_argument("--phantom", required=True, help="bifti phantom (.zip)")
    parser.add_argument("--n-iter", type=int, default=100)
    parser.add_argument("--url", help="rapisim: MCP gateway instead of a local binary")
    parser.add_argument("--rapisim-bin", help="rapisim: local binary (default: $RAPISIM_BIN or 'rapisim')")
    parser.add_argument("--out", help="write the results as json")
    args = parser.parse_args()

    timings = defaultdict(float)
    phase = ["fwd"]
    sim = make_simulator(args, timings, phase)

    def step(flips):
        t0 = time.perf_counter()
        seq = build_tse(refoc_flips=flips, check_timing=False)
        t1 = time.perf_counter()
        mseq = seq.to_mr0()
        t2 = time.perf_counter()
        signal = sim.simulate(mseq)
        t3 = time.perf_counter()
        reco = mr0.reco_adjoint(signal, mseq.get_kspace(), (64, 64, 1), (0.256, 0.256, 1))
        t4 = time.perf_counter()
        for name, seconds in zip(("build", "to_mr0", "simulate", "reco"), (t1 - t0, t2 - t1, t3 - t2, t4 - t3)):
            timings[name] += seconds
        return reco

    def reset():
        timings.clear()
        phase[0] = "fwd"

    with torch.no_grad():
        target = 0.8 * step(torch.full((N_ECHO,), np.pi))  # includes the one-off phantom decoding
    flips = torch.full((N_ECHO,), 160 * np.pi / 180, requires_grad=True)
    optimizer = torch.optim.Adam([flips], lr=0.02)
    start = step(flips).detach()
    rms_weight = 1 / ((start.abs() - target.abs()) ** 2).mean().sqrt()
    sar_weight = 1 / (flips.detach() ** 2).sum()
    reset()

    history = []
    t_start = time.perf_counter()
    for _ in range(args.n_iter):
        optimizer.zero_grad()
        phase[0] = "fwd"
        reco = step(flips)
        loss = rms_weight * ((reco.abs() - target.abs()) ** 2).mean().sqrt() + sar_weight * (flips**2).sum()
        phase[0] = "bwd"
        t = time.perf_counter()
        loss.backward()
        timings["backward"] += time.perf_counter() - t
        optimizer.step()
        history.append(loss.item())
    total = time.perf_counter() - t_start

    per_iter = {k: 1e3 * v / args.n_iter for k, v in timings.items()}
    print(f"{args.sim}: {args.n_iter} iterations in {total:.1f} s ({1e3 * total / args.n_iter:.0f} ms each)")
    for name, ms in per_iter.items():
        print(f"  {name:10s} {ms:8.1f} ms")
    print(f"  loss {history[0]:.4f} -> {history[-1]:.4f} (min {min(history):.4f})")
    print("  flips [deg]", np.degrees(flips.detach().numpy()).round(1).tolist())

    if args.out:
        Path(args.out).write_text(
            json.dumps(dict(sim=args.sim, n_iter=args.n_iter, total_s=total, per_iter_ms=per_iter, loss=history), indent=1)
        )
    if args.sim == "rapisim":
        sim.close()


if __name__ == "__main__":
    main()
