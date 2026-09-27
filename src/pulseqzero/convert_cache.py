"""Reuse a previous ``to_mr0()`` when the block layout is unchanged.

The first conversion builds the MRzero sequence and records which outputs are
live tensors (flip angle, phase, delay, ADC phase). A later sequence with the
same layout clones that result and writes the new tensor values in. Anything
else — a different block list, a tensor gradient waveform, a soft delay —
falls back to a full conversion.
"""

import math
from collections import OrderedDict

import numpy as np
import torch
import MRzeroCore as mr0

from .events import Adc, ArbitraryGrad, Delay, ExtTrapGrad, RfPulse, SoftDelay, TrapGrad
from .seq_convert import convert, convert_tensors_to_float32, parse_spoiler

_CACHE: OrderedDict = OrderedDict()
_MAX = 4


def clear_structure_cache() -> None:
    _CACHE.clear()


def convert_cached(pp0, samples_offres: int, samples_slicesel: int, samples_onres: int):
    for block in pp0.blocks:
        for ev in block:
            convert_tensors_to_float32(ev)
    key = _structure_key(pp0, samples_offres, samples_slicesel, samples_onres)
    hit = _CACHE.get(key)
    if hit is not None:
        try:
            _CACHE.move_to_end(key)
            return _replay(hit, pp0)
        except Exception:
            _CACHE.pop(key, None)
    box = {"ok": True, "ops": []}
    seq = convert(pp0, samples_offres, samples_slicesel, samples_onres, _patches=box)
    if box["ok"]:
        _remember(key, seq, box["ops"])
    return seq


def _remember(key, seq, ops) -> None:
    _CACHE[key] = (_detach_sequence(seq), tuple(ops))
    _CACHE.move_to_end(key)
    while len(_CACHE) > _MAX:
        _CACHE.popitem(last=False)


def _replay(hit, pp0):
    template, ops = hit
    out = template.clone()
    for op in ops:
        kind = op[0]
        if kind == "spoiler":
            _, rep, index, block = op
            delay, _, _, gx, gy, gz = _parts(pp0.blocks[block])
            (ev,) = parse_spoiler(delay, gx, gy, gz)
            out[rep].event_time[index] = ev.duration
            out[rep].gradm[index, :] = ev.gradm
        elif kind == "angle":
            _, rep, block, frac, usage_open = op
            rf = _parts(pp0.blocks[block])[2]
            pulse = out[rep].pulse
            angle = (torch.as_tensor(rf.flip_angle, dtype=torch.float32) * frac).reshape(pulse.angle.shape)
            pulse.angle = angle
            pulse.pulse_freq = angle / pulse.duration
            if usage_open:
                turn = float(angle.detach().reshape(-1)[0])
                pulse.usage = (
                    mr0.PulseUsage.REFOC if turn > 100 * math.pi / 180 else mr0.PulseUsage.EXCIT
                )
        elif kind == "phase":
            _, rep, block, coeff, freq_live = op
            rf = _parts(pp0.blocks[block])[2]
            pulse = out[rep].pulse
            phase = torch.as_tensor(rf.phase_offset, dtype=torch.float32) + coeff * torch.as_tensor(
                rf.freq_offset, dtype=torch.float32
            )
            pulse.phase = phase.reshape(pulse.phase.shape)
            if freq_live:
                pulse.freq_offset = torch.as_tensor(rf.freq_offset, dtype=torch.float32).reshape(
                    pulse.freq_offset.shape
                )
                pulse.off_res = bool(
                    (pulse.freq_offset != 0).any().item() or (pulse.grad != 0).any().item()
                )
        elif kind == "adc":
            _, rep, start, num, block, t_samples = op
            adc = _parts(pp0.blocks[block])[1]
            t = torch.tensor(t_samples, dtype=torch.float32)
            phase = torch.as_tensor(adc.phase_offset, dtype=torch.float32) + (
                2.0 * math.pi * torch.as_tensor(adc.freq_offset, dtype=torch.float32) * t
            )
            out[rep].adc_phase[start : start + num] = torch.pi / 2 - phase
        else:
            raise RuntimeError(f"unknown replay op {kind}")
    return out


def _detach_sequence(seq):
    out = seq.clone()
    for rep in out:
        pulse = rep.pulse
        for name in ("angle", "phase", "pulse_freq", "freq_offset", "duration", "grad", "shim_array"):
            value = getattr(pulse, name)
            if torch.is_tensor(value):
                setattr(pulse, name, value.detach())
        rep.event_time = rep.event_time.detach()
        rep.gradm = rep.gradm.detach()
        rep.adc_phase = rep.adc_phase.detach()
        rep.adc_usage = rep.adc_usage.detach()
    return out


def _structure_key(pp0, samples_offres, samples_slicesel, samples_onres):
    blocks = tuple(tuple(_event_key(ev) for ev in block) for block in pp0.blocks)
    return (samples_offres, samples_slicesel, samples_onres, blocks)


def _event_key(ev):
    if isinstance(ev, Delay) and not isinstance(ev, SoftDelay):
        return ("delay", _freeze(ev.delay))
    if isinstance(ev, SoftDelay):
        return (
            "soft",
            ev.hint,
            ev.numID,
            _freeze(ev.offset),
            _freeze(ev.factor),
            _freeze(ev.default_duration),
        )
    if isinstance(ev, Adc):
        return (
            "adc",
            ev.num_samples,
            _freeze(ev.dwell),
            _freeze(ev.delay),
            _freeze(ev.freq_offset),
            _freeze(ev.phase_offset),
            _freeze(ev.freq_ppm),
            _freeze(ev.phase_ppm),
        )
    if isinstance(ev, RfPulse):
        return (
            "rf",
            _freeze(ev.flip_angle),
            _freeze(ev.freq_offset),
            _freeze(ev.phase_offset),
            _freeze(ev.delay),
            _freeze(ev.shape_dur),
            _freeze(ev.center),
            _freeze(ev.ringdown_time),
            ev.use,
            _freeze(ev.shim_array),
        )
    if isinstance(ev, TrapGrad):
        return (
            "trap",
            ev.channel,
            _freeze(ev.amplitude),
            _freeze(ev.rise_time),
            _freeze(ev.flat_time),
            _freeze(ev.fall_time),
            _freeze(ev.delay),
        )
    if isinstance(ev, ArbitraryGrad):
        return (
            "arb",
            ev.channel,
            _freeze(ev.waveform),
            _freeze(ev.delay),
            _freeze(ev.first),
            _freeze(ev.last),
            bool(ev.oversampling),
            _freeze(ev._grad_raster),
        )
    if isinstance(ev, ExtTrapGrad):
        return ("ext", ev.channel, _freeze(ev.waveform), _freeze(ev._times))
    return ("other", type(ev).__name__)


def _freeze(value):
    if isinstance(value, torch.Tensor):
        return ("T", tuple(value.shape), str(value.dtype))
    if isinstance(value, np.ndarray):
        return ("A", tuple(value.shape), str(value.dtype), value.tobytes())
    if isinstance(value, np.generic):
        return ("S", value.item())
    if isinstance(value, float):
        return ("F", float(value))
    if value is None or isinstance(value, (int, bool, str)):
        return ("S", value)
    return ("U", type(value).__name__)


def _parts(block):
    delay = adc = rf = gx = gy = gz = None
    for ev in block:
        if isinstance(ev, (Delay, SoftDelay)):
            delay = ev
        elif isinstance(ev, Adc):
            adc = ev
        elif isinstance(ev, RfPulse):
            rf = ev
        elif isinstance(ev, (TrapGrad, ExtTrapGrad, ArbitraryGrad)):
            if ev.channel == "x":
                gx = ev
            elif ev.channel == "y":
                gy = ev
            else:
                gz = ev
    return delay, adc, rf, gx, gy, gz
