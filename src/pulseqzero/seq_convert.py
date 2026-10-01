"""Convert a pulseq-zero sequence into an MR-zero sequence.

Every block is converted on its own into temporary events: spoilers (time and
gradient moment), pulses and ADC samples. Which events a block produces, and
therefore where they end up in the MR-zero sequence, only depends on the kind
of block - an RF block with n sub-pulses, an ADC block with n samples or a
spoiler. This is the layout of the sequence.

`convert` builds the MR-zero sequence from scratch. `convert_cached` keeps the
sequence of the last call with the same layout and only writes the blocks that
changed, plus all blocks holding torch tensors, so that their gradients reach
the new sequence. Optimization loops rebuild the pulseq-zero sequence every
iteration and usually only change a few tensors, this skips the rest.
"""

from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import torch
import MRzeroCore as mr0
from pypulseq import Opts

from . import calc_duration
from .events import Adc, Delay, SoftDelay, RfPulse, TrapGrad, ExtTrapGrad, ArbitraryGrad
from .rf_shapes import RfShape


# =============================================================================
# Blocks
# =============================================================================


@dataclass
class Block:
    delay: Delay | SoftDelay | None
    rf: RfPulse | None
    adc: Adc | None
    grad_x: TrapGrad | ExtTrapGrad | ArbitraryGrad | None
    grad_y: TrapGrad | ExtTrapGrad | ArbitraryGrad | None
    grad_z: TrapGrad | ExtTrapGrad | ArbitraryGrad | None
    samples: int  # number of sub-pulses of an RF block
    content: tuple  # everything but tensor values, to detect changes
    live_fields: set[str]  # fields holding tensors, like "rf.flip_angle"

    @property
    def grads(self):
        return self.grad_x, self.grad_y, self.grad_z

    @property
    def duration(self):
        return calc_duration(self.delay, self.rf, self.adc, *self.grads)

    @property
    def layout(self) -> tuple:
        if self.rf:
            return ("rf", self.samples)
        if self.adc:
            return ("adc", self.adc.num_samples)
        return ("spoiler",)


def read_block(events, sample_counts: tuple[int, int, int]) -> Block:
    parts = {}
    content = []
    live_fields = set()
    for ev in events:
        if isinstance(ev, (TrapGrad, ExtTrapGrad, ArbitraryGrad)):
            assert ev.channel in ["x", "y", "z"]
            name = "grad_" + ev.channel
        else:
            name = _PART_NAMES.get(type(ev))
            if name is None:
                continue  # labels etc. are not simulated
        assert name not in parts
        parts[name] = ev

        content.append(type(ev))
        for field, value in vars(ev).items():
            if isinstance(value, torch.Tensor):
                if value.dtype == torch.float64:
                    value = value.to(dtype=torch.float32)
                    setattr(ev, field, value)
                live_fields.add(f"{name}.{field}")
                content.append(value.shape)  # the value itself is not cached
            else:
                content.append(_freeze(value))

    block = Block(
        delay=parts.get("delay"),
        rf=parts.get("rf"),
        adc=parts.get("adc"),
        grad_x=parts.get("grad_x"),
        grad_y=parts.get("grad_y"),
        grad_z=parts.get("grad_z"),
        samples=0,
        content=tuple(content),
        live_fields=live_fields,
    )
    assert not (block.rf and block.adc)
    if block.rf:
        # Use pulse sub-samples according to the type of pulse
        samples_offres, samples_slicesel, samples_onres = sample_counts
        if block.rf.freq_offset != 0:
            block.samples = samples_offres
        elif any(block.grads):
            block.samples = samples_slicesel
        else:
            block.samples = samples_onres
    return block


_PART_NAMES = {Delay: "delay", SoftDelay: "delay", RfPulse: "rf", Adc: "adc"}


def _freeze(value):
    """Turn a non-tensor field value into something that can be compared with ==."""
    if isinstance(value, np.ndarray):
        return (value.shape, value.tobytes())
    if isinstance(value, SimpleNamespace):
        return tuple((k, _freeze(v)) for k, v in vars(value).items())
    return value


# =============================================================================
# Layout: where the events of every block go
# =============================================================================


class Slot(NamedTuple):
    rep: int  # repetition the event belongs to
    start: int  # first event index in that repetition (unused for pulses)
    count: int  # number of events: 0 for a pulse, 1 for a spoiler, n for an ADC


def place_events(blocks: list[Block]) -> tuple[list[list[Slot | None]], list[int]]:
    """Return the slots of every block's events and the size of every repetition.

    Every pulse starts a new repetition. Events before the first pulse are not
    part of any repetition and get no slot.
    """
    block_slots = []
    rep_sizes = []
    for block in blocks:
        slots = []
        for count in _event_counts(block):
            if count == 0:
                rep_sizes.append(0)
                slots.append(Slot(len(rep_sizes) - 1, 0, 0))
            elif rep_sizes:
                slots.append(Slot(len(rep_sizes) - 1, rep_sizes[-1], count))
                rep_sizes[-1] += count
            else:
                slots.append(None)
        block_slots.append(slots)
    return block_slots, rep_sizes


def _event_counts(block: Block) -> list[int]:
    """Event counts of what parse_block returns, see Slot.count."""
    if block.rf:
        return [1] + [0, 1] * block.samples
    if block.adc:
        return [block.adc.num_samples, 1]
    return [1]


def new_sequence(rep_sizes: list[int]) -> mr0.Sequence:
    seq = mr0.Sequence(normalized_grads=False)
    for size in rep_sizes:
        seq.new_rep(size)
    return seq


# =============================================================================
# Conversion
# =============================================================================


def read_blocks(pp0, samples_offres, samples_slicesel, samples_onres) -> list[Block]:
    sample_counts = (samples_offres, samples_slicesel, samples_onres)
    return [read_block(events, sample_counts) for events in pp0.blocks]


def convert(
    pp0, samples_offres: int, samples_slicesel: int, samples_onres: int
) -> mr0.Sequence:
    blocks = read_blocks(pp0, samples_offres, samples_slicesel, samples_onres)
    block_slots, rep_sizes = place_events(blocks)
    seq = new_sequence(rep_sizes)
    for block, slots in zip(blocks, block_slots):
        write_block(seq, slots, block)
    return seq


@dataclass
class CachedSequence:
    seq: mr0.Sequence  # without gradients
    block_slots: list[list[Slot | None]]
    block_contents: list[tuple | None]  # what every block was written from


_cache: dict[tuple, CachedSequence] = {}
_CACHE_SIZE = 4


def clear_cache():
    _cache.clear()


def convert_cached(
    pp0, samples_offres: int, samples_slicesel: int, samples_onres: int
) -> mr0.Sequence:
    """Same result as `convert`, reusing the last sequence with the same layout."""
    blocks = read_blocks(pp0, samples_offres, samples_slicesel, samples_onres)
    layout = tuple(block.layout for block in blocks)

    cached = _cache.get(layout)
    if cached is None:
        if len(_cache) >= _CACHE_SIZE:
            del _cache[next(iter(_cache))]
        block_slots, rep_sizes = place_events(blocks)
        cached = CachedSequence(new_sequence(rep_sizes), block_slots, [None] * len(blocks))
        _cache[layout] = cached

    # Bring the cached sequence up to date, without tracking gradients
    with torch.no_grad():
        for i, block in enumerate(blocks):
            if block.content != cached.block_contents[i]:
                write_block(cached.seq, cached.block_slots[i], block)
                cached.block_contents[i] = block.content

    # Write everything that depends on tensors again, this time with gradients
    seq = cached.seq.clone()
    for block, slots in zip(blocks, cached.block_slots):
        if not block.live_fields:
            continue
        if block.live_fields <= ROTATION_FIELDS:
            write_rotations(seq, slots, block)
        else:
            write_block(seq, slots, block)
    return seq


# Tensors in these fields only change pulse rotations and ADC phases
ROTATION_FIELDS = {
    "rf.flip_angle",
    "rf.phase_offset",
    "rf.freq_offset",
    "adc.phase_offset",
    "adc.freq_offset",
}


# =============================================================================
# Writing into the MR-zero sequence
# =============================================================================


def write_block(seq: mr0.Sequence, slots: list[Slot | None], block: Block):
    for slot, ev in zip(slots, parse_block(block)):
        if slot is None:
            continue
        rep = seq[slot.rep]
        if isinstance(ev, TmpPulse):
            write_pulse(rep.pulse, ev)
        elif isinstance(ev, TmpSpoiler):
            rep.event_time[slot.start] = ev.duration
            rep.gradm[slot.start, :] = ev.gradm
        else:
            assert isinstance(ev, TmpAdc)
            samples = slice(slot.start, slot.start + slot.count)
            rep.event_time[samples] = torch.as_tensor(ev.event_time)
            rep.gradm[samples, :] = torch.as_tensor(ev.gradm)
            rep.adc_phase[samples] = torch.pi / 2 - ev.phase
            rep.adc_usage[samples] = 1


def write_rotations(seq: mr0.Sequence, slots: list[Slot | None], block: Block):
    """Write only what depends on flip angle, phase and frequency offsets."""
    if block.rf:
        pulse_slots = slots[1::2]  # see _event_counts
        for slot, (angle, phase) in zip(pulse_slots, pulse_rotations(block)):
            write_rotation(seq[slot.rep].pulse, angle, phase, block.rf.freq_offset, _usage(block.rf))
    elif block.adc and slots[0] is not None:
        slot = slots[0]
        samples = slice(slot.start, slot.start + slot.count)
        seq[slot.rep].adc_phase[samples] = torch.pi / 2 - adc_phase(block.adc)


def write_pulse(pulse: mr0.Pulse, ev: "TmpPulse"):
    pulse.duration = torch.as_tensor(ev.duration, dtype=torch.float32)
    pulse.grad = torch.stack([
        torch.as_tensor(ev.grad_x, dtype=torch.float32),
        torch.as_tensor(ev.grad_y, dtype=torch.float32),
        torch.as_tensor(ev.grad_z, dtype=torch.float32),
    ])
    # pulse.selective: True when a z-gradient is active during the pulse
    pulse.selective = bool(torch.as_tensor(ev.grad_z) != 0)
    if ev.shim_array is not None:
        pulse.shim_array = ev.shim_array
    write_rotation(pulse, ev.angle, ev.phase, ev.freq_offset, ev.use)


def write_rotation(pulse: mr0.Pulse, angle, phase, freq_offset, use: mr0.PulseUsage):
    # dtype is pinned: a flip angle given as a numpy scalar (np.deg2rad(90),
    # ubiquitous in pypulseq scripts) would otherwise make pulse.angle
    # float64 and execute_graph fail against a float32 phantom.
    pulse.angle = torch.as_tensor(angle, dtype=torch.float32)
    pulse.phase = torch.as_tensor(phase, dtype=torch.float32)
    pulse.freq_offset = torch.as_tensor(freq_offset, dtype=torch.float32)
    pulse.off_res = bool(
        (pulse.freq_offset != 0).any().item() or (pulse.grad != 0).any().item()
    )
    # pulse_freq = ω₁ = angle/duration (legacy field, to be removed upstream)
    pulse.pulse_freq = pulse.angle / pulse.duration

    # pulse.usage: honour the rf.use tag when it's explicit; only fall back
    # to the flip-angle heuristic when the tag was not set ('undefined'/UNDEF).
    if use != mr0.PulseUsage.UNDEF:
        pulse.usage = use
    elif pulse.angle > 100 * torch.pi / 180:
        pulse.usage = mr0.PulseUsage.REFOC
    else:
        pulse.usage = mr0.PulseUsage.EXCIT


# =============================================================================
# Parsing blocks into temporary events
# =============================================================================


class TmpPulse:
    def __init__(
        self,
        angle,
        phase,
        duration,
        freq_offset,
        grad_x,
        grad_y,
        grad_z,
        shim_array,
        use: mr0.PulseUsage,
    ) -> None:
        self.angle = angle
        self.phase = phase
        self.freq_offset = freq_offset
        self.grad_x = grad_x
        self.grad_y = grad_y
        self.grad_z = grad_z
        self.duration = duration
        self.shim_array = shim_array
        self.use = use

    def __repr__(self) -> str:
        from math import pi

        return f"Pulse(angle={self.angle * 180 / pi}°, phase={self.phase * 180 / pi}°, shim_array={self.shim_array}, use={self.use})"


class TmpSpoiler:
    def __init__(self, duration, gx, gy, gz) -> None:
        self.duration = torch.as_tensor(duration)
        self.gradm = torch.cat(
            [
                torch.as_tensor(gx).view(1),
                torch.as_tensor(gy).view(1),
                torch.as_tensor(gz).view(1),
            ]
        )

    def __repr__(self) -> str:
        return f"Spoiler(gradm={self.gradm}, duration={self.duration})"


class TmpAdc:
    def __init__(self, event_time, gradm, phase) -> None:
        self.event_time = event_time
        self.gradm = gradm
        self.phase = phase

    def __repr__(self) -> str:
        from math import pi

        return f"Adc(phase={self.phase * 180 / pi}°, total_gradm={self.gradm.sum(0)}, total_time={self.event_time.sum(0)})"


def parse_block(block: Block) -> list[TmpPulse | TmpSpoiler | TmpAdc]:
    if block.rf:
        return parse_pulse(block)
    elif block.adc:
        return parse_adc(block)
    else:
        return parse_spoiler(block)


def _usage(rf: RfPulse) -> mr0.PulseUsage:
    if rf.use == "excitation":
        return mr0.PulseUsage.EXCIT
    elif rf.use == "refocusing":
        return mr0.PulseUsage.REFOC
    else:
        return mr0.PulseUsage.UNDEF


def pulse_rotations(block: Block) -> list[tuple]:
    """Flip angle and phase of every sub-pulse of an RF block."""
    rf = block.rf
    step = rf.shape_dur / block.samples

    # Adjusted to cover whole block
    # Integration windows used to divide the RF waveform.
    # The first and last windows include the regions before and after the
    # RF waveform. integrate_pulse() evaluates the RF as zero there.
    t_rf = [0] + [rf.delay + step * i for i in range(1, block.samples)] + [block.duration]

    rotations = []
    for i in range(block.samples):
        angle, phase = integrate_pulse(rf, t_rf[i], t_rf[i + 1])
        # phase profile due to off-resonance
        phase = phase + 2 * torch.pi * rf.freq_offset * step * i
        rotations.append((angle, phase))
    return rotations


def parse_pulse(block: Block) -> list[TmpPulse | TmpSpoiler]:
    rf = block.rf
    step = rf.shape_dur / block.samples

    # Effective time of each instantaneous MRzero sub-pulse.
    # These must be centred within the actual RF shape, without including
    # RF dead time, RF delay, or ringdown in the centring operation.
    pulse_times = [rf.delay + (i + 0.5) * step for i in range(block.samples)]
    # grads are integrated from one pulse center to the next
    t_grad = torch.stack(
        [torch.as_tensor(t, dtype=torch.float64) for t in [0, *pulse_times, block.duration]]
    )
    gradm = torch.stack([
        integrate(grad, t_grad) if grad else torch.zeros_like(t_grad)
        for grad in block.grads
    ], dim=1)
    gradm = torch.diff(gradm, dim=0)
    event_time = torch.diff(t_grad)

    # handle gradients in parallel to pulse
    rf_dur = rf.delay + rf.shape_dur  # rf duration without ringdown
    use = _usage(rf)

    # Alternate spoiler from one pulse center to next with pulse itself
    events: list[TmpPulse | TmpSpoiler] = []
    for i, (angle, phase) in enumerate(pulse_rotations(block)):
        events.append(TmpSpoiler(event_time[i], *gradm[i]))
        grad_ampl = [_amplitude_at(grad, rf_dur, t_grad[i + 1]) for grad in block.grads]
        events.append(
            TmpPulse(angle, phase, step, rf.freq_offset, *grad_ampl, rf.shim_array, use)
        )
    events.append(TmpSpoiler(event_time[-1], *gradm[-1]))

    return events


def _amplitude_at(grad, rf_dur, t):
    if grad is None or rf_dur <= grad.delay:  # gradient in block starts after pulse has ended
        return 0
    # distinguish between TrapGrad and FreeGrad
    if isinstance(grad, ExtTrapGrad | ArbitraryGrad):
        # find closest waveform point to gradient timepoint
        return grad.waveform[torch.argmin(torch.abs(torch.as_tensor(grad.tt) - t))]
    return grad.amplitude


def parse_spoiler(block: Block) -> tuple[TmpSpoiler]:
    gx, gy, gz = (grad.area if grad is not None else 0.0 for grad in block.grads)
    return (TmpSpoiler(block.duration, gx, gy, gz),)


def adc_phase(adc: Adc) -> torch.Tensor:
    # Per-sample ADC phase: constant phase_offset plus time-varying freq_offset term.
    # t_samples are the centre times of each ADC dwell interval.
    t_samples = (
        adc.delay
        + (torch.arange(adc.num_samples, dtype=torch.float32) + 0.5) * adc.dwell
    )
    return adc.phase_offset + 2 * torch.pi * adc.freq_offset * t_samples


def parse_adc(block: Block) -> tuple[TmpAdc, TmpSpoiler]:
    adc = block.adc
    time = torch.cat(
        [
            torch.as_tensor(0.0).view((1,)),
            adc.delay + (torch.arange(adc.num_samples) + 0.5) * adc.dwell,
            torch.as_tensor(block.duration).view((1,)),
        ]
    )

    gradm = torch.zeros((adc.num_samples + 2, 3))
    for channel, grad in enumerate(block.grads):
        if grad:
            gradm[:, channel] = integrate(grad, time)

    event_time = torch.diff(time)
    gradm = torch.diff(gradm, dim=0)
    return (
        TmpAdc(event_time[:-1], gradm[:-1, :], adc_phase(adc)),
        TmpSpoiler(event_time[-1], gradm[-1, 0], gradm[-1, 1], gradm[-1, 2]),
    )


# =============================================================================
# Integration of gradients and pulses
# =============================================================================


def split_gradm(grad, t):
    before = integrate(grad, t)
    total = integrate(grad, 1e9)  # Infinity produces 0*inf = NaNs internally
    return (before, total - before)


def integrate(grad, t):
    """Gradient moment from the start of the block up to t (scalar or 1D tensor)."""
    if isinstance(grad, TrapGrad):
        # The Heaviside terms only select which piece of the piecewise integral
        # is active; they are not part of the integrand. torch.heaviside has no
        # backward implementation, so the step is evaluated detached. Without
        # this, optimizing anything that ends up in a time shape raises
        # "derivative for aten::heaviside is not implemented".
        def h(x):
            x = torch.as_tensor(x)
            return torch.heaviside(x.detach(), torch.tensor(0.5, dtype=x.dtype))

        # https://www.desmos.com/calculator/0q5co02ecm

        d = grad.delay
        t1 = grad.rise_time
        t2 = grad.flat_time
        t3 = grad.fall_time
        T1 = d + t1
        T12 = d + t1 + t2
        T123 = d + t1 + t2 + t3

        # Trapezoid, could be provided as derivative:
        # f1 = h(t - d) * h(T1 - t) * (t - d) / t1
        # f2 = h(t - T1) * h(T12 - t)
        # f3 = h(t - T12) * h(T123 - t) * (T123 - t) / t3
        # f = grad.amplitude * (f1 + f2 + f3)

        F_inf = t1 / 2 + t2 + t3 / 2
        F1 = h(t - d) * h(T1 - t) * 0.5 * (t - d) ** 2 / t1
        F2 = h(t - T1) * h(T12 - t) * (t1 / 2 + t - T1)
        F3 = h(t - T12) * h(T123 - t) * (F_inf - 0.5 * (T123 - t) ** 2 / t3)
        F = grad.amplitude * (F1 + F2 + F3 + h(t - T123) * F_inf)

        return F
    elif isinstance(grad, ExtTrapGrad | ArbitraryGrad):
        # To stay differentiable, we don't want dynamic indexing, but instead
        # calculate, how much of every segment of the gradient contributes
        # https://www.desmos.com/calculator/j2vopzhb2z

        d = grad.delay

        tt = torch.as_tensor(grad.tt)
        waveform = torch.as_tensor(grad.waveform).reshape(-1)

        if isinstance(grad, ArbitraryGrad):
            tt = torch.cat((
                torch.zeros(1, dtype=tt.dtype),
                tt,
                torch.as_tensor(grad.shape_dur, dtype=tt.dtype).reshape(1),
            ))

            waveform = torch.cat((
                torch.as_tensor(grad.first, dtype=waveform.dtype).reshape(1),
                waveform,
                torch.as_tensor(grad.last, dtype=waveform.dtype).reshape(1),
            ))

        # Start and end time point and amplitude of all line segments
        t1 = d + tt[:-1]
        t2 = d + tt[1:]
        c1 = waveform[:-1]
        c2 = waveform[1:]

        # One row of segments per time point
        t = torch.as_tensor(t)[..., None]
        # This is how much of every segment contributes, clamped to [0, width]
        t_rel = torch.clamp(t - t1, 0 * t1, t2 - t1)
        # The amplitude of the segment at t, will be clamped to the amplitude
        # of the right point for segments before t and the left point for
        # segments after; only one segment where t lies in will be interpolated
        c_end = c1 + t_rel / (t2 - t1) * (c2 - c1)
        # For integration, we calculate the area of the rectangle with the
        # average height of the left and right side of the actual shape
        c_avg = 0.5 * (c1 + c_end)
        return (t_rel * c_avg).sum(-1)
    else:
        raise NotImplementedError


def integrate_pulse(rf: RfPulse, t_start, t_end):
    # The fraction of the pulse area inside [t_start, t_end] only depends on
    # the (detached) shape and timing. Multiplying by the live rf.flip_angle
    # tensor keeps the gradient of the user's flip-angle parameter.
    fraction = pulse_fraction(
        rf.waveform, float(rf.shape_dur), float(rf.delay), float(t_start), float(t_end)
    )
    flip = torch.as_tensor(rf.flip_angle) * fraction
    phase = (
        rf.phase_offset + 0.0
    )  # not returned by the _generate_shape() function - extend!

    return flip, phase


@lru_cache(maxsize=256)
def pulse_shape(shape: RfShape, shape_dur: float) -> tuple[np.ndarray, np.ndarray]:
    """Time points (from the shape start) and amplitudes of the pulse waveform."""
    unit_pulse = RfPulse(
        flip_angle=1.0,
        freq_offset=0.0,
        phase_offset=0.0,
        delay=0.0,
        shape_dur=shape_dur,
        center=0.0,
        ringdown_time=0.0,
        use="undefined",
        shim_array=None,
        freq_ppm=0.0,
        phase_ppm=0.0,
        waveform=shape,
    )
    pp_rf = unit_pulse.to_pulseq(Opts.default)
    return np.asarray(pp_rf.t), np.asarray(pp_rf.signal)


@lru_cache(maxsize=4096)
def pulse_fraction(
    shape: RfShape, shape_dur: float, delay: float, t_start: float, t_end: float
) -> float:
    """Fraction of the pulse area between t_start and t_end (block time)."""
    t_rel, amp_shape = pulse_shape(shape, shape_dur)
    time_shape = t_rel + delay

    # Clamp the window to the support of the RF shape. Outside of it the RF is
    # zero, but a trapezoidal segment joining a zero pad point to a non-zero
    # boundary sample would contribute area that the pulse does not have
    # (block pulses and most arbitrary RF are non-zero at their boundaries).
    lo = max(t_start, float(time_shape[0]))
    hi = min(t_end, float(time_shape[-1]))
    if hi <= lo:
        return 0.0
    # Find where lo and hi are placed in time_shape
    i_start = np.searchsorted(time_shape, lo, side="left")
    i_end = np.searchsorted(time_shape, hi, side="right")
    # Find the interpolated shape values at lo and hi
    v_start = np.interp(lo, time_shape, amp_shape)
    v_end = np.interp(hi, time_shape, amp_shape)
    # Construct the shape of the integrated part of the pulse
    time = [lo] + time_shape[i_start:i_end].tolist() + [hi]
    amp = [v_start] + amp_shape[i_start:i_end].tolist() + [v_end]

    window_area = np.trapezoid(amp, time)
    full_area = np.trapezoid(amp_shape, time_shape)
    return float(window_area / full_area) if full_area != 0 else 0.0
