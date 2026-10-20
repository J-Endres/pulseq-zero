"""``seq.simulate(sim)`` with the two simulators: MR-zero and rapisim.

Both simulate the same sequence on the same segmented phantom (a bifti file,
which MR-zero reads as well). They are different implementations of PDG, so
they have to agree on the signal, and their gradients by the flip angle must
agree with each other and with finite differences.

Needs the rapisim binary (``cargo build --release`` in tool-rapisim, then
``RAPISIM_BIN=.../target/release/rapisim``), the tests are skipped without.
"""

import json
import os
import shutil
import zipfile

import numpy as np
import pytest
import torch

import pulseqzero as pp
from sequences import build_se

nib = pytest.importorskip("nibabel")
pytest.importorskip("fastmcp")
mr0 = pytest.importorskip("MRzeroCore")

RAPISIM = os.environ.get("RAPISIM_BIN") or shutil.which("rapisim")
pytestmark = pytest.mark.skipif(RAPISIM is None, reason="set RAPISIM_BIN to the rapisim binary")


@pytest.fixture(scope="module")
def phantom(tmp_path_factory):
    """Two tissues on a 24 x 24 x 1 grid: a disk and a ring around it"""
    root = tmp_path_factory.mktemp("phantom")
    n = 24
    y, x = np.mgrid[:n, :n] - (n - 1) / 2
    r = np.hypot(x, y)
    density = np.stack([(r < 5).astype(np.float32), ((r >= 6) & (r < 10)).astype(np.float32)], axis=-1)
    affine = np.diag([3.0, 3.0, 8.0, 1.0])
    nib.save(nib.Nifti1Image(density[:, :, None, :], affine), root / "pd.nii.gz")

    def tissue(index, t1, t2, t2dash, adc):
        return {
            "density": f"pd.nii.gz[{index}]",
            "T1": t1, "T2": t2, "T2'": t2dash, "ADC": adc,
            "dB0": 0.0, "B1+": [1.0], "B1-": [1.0],
        }

    spec = {
        "$schema": "https://raw.githubusercontent.com/mrx-org/nifti-phantoms/refs/heads/main/nifti-phantom-v1.schema.json",
        "units": {"gyro": "MHz/T", "B0": "T", "T1": "s", "T2": "s", "T2'": "s", "ADC": "10^-3 mm^2/s",
                  "dB0": "Hz", "B1+": "rel", "B1-": "rel"},
        "system": {"gyro": 42.5764, "B0": 3},
        "tissues": {"disk": tissue(0, 1.2, 0.09, 0.05, 0.8), "ring": tissue(1, 0.5, 0.06, 0.03, 0.9)},
    }
    (root / "phantom.json").write_text(json.dumps(spec))
    with zipfile.ZipFile(root / "phantom.zip", "w") as z:
        for name in ("phantom.json", "pd.nii.gz"):
            z.write(root / name, name)
    return root


@pytest.fixture(scope="module")
def mr0_sim(phantom):
    data = mr0.phantom.tissue_dict.TissueDict.load(phantom / "phantom.json").build()
    return pp.sim.MR0(data)


@pytest.fixture(scope="module")
def rapisim_sim(phantom):
    sim = pp.sim.Rapisim(phantom / "phantom.zip", command=[RAPISIM])
    yield sim
    sim.close()


def spin_echo(refoc_deg):
    return build_se(pp, refoc_deg=refoc_deg, n=16)


def test_signals_agree(mr0_sim, rapisim_sim):
    seq = spin_echo(150.0)
    a = seq.simulate(mr0_sim).detach().flatten().numpy()
    b = seq.simulate(rapisim_sim).detach().flatten().numpy()
    assert a.shape == b.shape == (16 * 16,)

    peak = np.abs(a).max()
    assert peak > 0.01, "vacuous"
    assert abs(np.abs(b).max() / peak - 1) < 0.03
    corr = np.abs(np.vdot(a, b)) / np.linalg.norm(a) / np.linalg.norm(b)
    print(f"\nsignals: peak ratio {np.abs(b).max() / peak:.4f}, correlation {corr:.5f}")
    assert corr > 0.98, corr


def loss(signal):
    return signal.abs().pow(2).sum().sqrt()


def gradient(sim, refoc_deg):
    flip = torch.tensor(refoc_deg, requires_grad=True)
    value = loss(spin_echo(flip).simulate(sim))
    value.backward()
    return value.item(), flip.grad.item()


def test_gradients_agree(mr0_sim, rapisim_sim):
    value_r, grad_r = gradient(rapisim_sim, 150.0)
    value_m, grad_m = gradient(mr0_sim, 150.0)
    assert abs(grad_m) > 1e-3, "vacuous"
    assert abs(value_r / value_m - 1) < 0.05
    print(f"\ngradient: rapisim {grad_r:.5f}, mr0 {grad_m:.5f}, value ratio {value_r / value_m:.4f}")
    assert abs(grad_r / grad_m - 1) < 0.1, (grad_r, grad_m)


def test_gradients_match_finite_differences(rapisim_sim):
    eps = 0.5  # degrees
    _, grad = gradient(rapisim_sim, 150.0)
    with torch.no_grad():
        up = loss(spin_echo(torch.tensor(150.0 + eps)).simulate(rapisim_sim)).item()
        down = loss(spin_echo(torch.tensor(150.0 - eps)).simulate(rapisim_sim)).item()
    fd = (up - down) / (2 * eps)
    assert abs(fd) > 1e-3, "vacuous"
    print(f"\nrapisim gradient {grad:.5f}, finite difference {fd:.5f}")
    assert abs(grad / fd - 1) < 1e-2, (grad, fd)
