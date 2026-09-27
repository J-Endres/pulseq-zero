"""Repeated to_mr0() with a fixed block layout refreshes tensor values only."""

import torch

import pulseqzero as pp
from pulseqzero.convert_cache import clear_structure_cache
from pulseqzero.seq_convert import convert


def setup_function():
    clear_structure_cache()


def _to_mr0(seq):
    return seq.to_mr0(speed_up_by_assuming_const_seq_structure=True)


def _gre(flip, tr, samples=8):
    seq = pp.Sequence()
    rf = pp.make_block_pulse(flip_angle=flip, duration=1e-3)
    seq.add_block(rf)
    seq.add_block(pp.make_delay(tr))
    gx = pp.make_trapezoid(channel="x", flat_area=200.0, flat_time=1e-3)
    adc = pp.make_adc(num_samples=samples, duration=1e-3, delay=gx.rise_time)
    seq.add_block(adc, gx)
    return seq


def _angles(seq):
    return torch.stack([rep.pulse.angle.reshape(-1)[0] for rep in seq])


def _times(seq):
    return torch.stack([rep.event_time.sum() for rep in seq])


def test_replay_matches_full_conversion_and_keeps_gradients():
    fa = torch.tensor(0.4, requires_grad=True)
    tr = torch.tensor(0.03, requires_grad=True)
    _to_mr0(_gre(fa, tr))

    fa2 = torch.tensor(0.7, requires_grad=True)
    tr2 = torch.tensor(0.05, requires_grad=True)
    fast = _to_mr0(_gre(fa2, tr2))
    full = convert(_gre(fa2.detach(), tr2.detach()), 1, 1, 1)
    assert torch.allclose(_angles(fast), _angles(full))
    assert torch.allclose(_times(fast), _times(full))

    (_angles(fast).sum() + _times(fast).sum()).backward()
    assert fa2.grad is not None and float(fa2.grad) != 0.0
    assert tr2.grad is not None and float(tr2.grad) != 0.0


def test_previous_conversion_keeps_its_values():
    fa = torch.tensor(0.2, requires_grad=True)
    first = _to_mr0(_gre(fa, torch.tensor(0.02)))
    kept = first[0].pulse.angle.detach().clone()

    _to_mr0(_gre(torch.tensor(0.9, requires_grad=True), torch.tensor(0.02)))
    assert torch.allclose(first[0].pulse.angle, kept)


def test_layout_change_is_not_served_from_the_old_cache():
    fa = torch.tensor(0.3, requires_grad=True)
    _to_mr0(_gre(fa, torch.tensor(0.02), samples=8))
    fa2 = torch.tensor(0.6, requires_grad=True)
    fast = _to_mr0(_gre(fa2, torch.tensor(0.04), samples=32))
    full = convert(_gre(fa2.detach(), torch.tensor(0.04), samples=32), 1, 1, 1)
    assert len(fast) == len(full)
    assert torch.allclose(_angles(fast), _angles(full))
    assert torch.allclose(_times(fast), _times(full))


def test_flag_off_still_converts():
    fa = torch.tensor(0.25, requires_grad=True)
    seq = pp.Sequence()
    seq.add_block(pp.make_block_pulse(flip_angle=fa, duration=1e-3))
    seq.add_block(pp.make_delay(0.01))
    out = seq.to_mr0()
    (_angles(out).sum()).backward()
    assert fa.grad is not None
