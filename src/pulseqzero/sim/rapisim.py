"""Differentiable simulation with rapisim (https://github.com/mrx-org/tool-rapisim).

rapisim simulates sequences on segmented bifti phantoms with PDG. It is an MCP
tool: ``Rapisim`` runs it as a local binary (stdio) or talks to an MCP gateway.
Forward is the tool ``simulate_diff``, backward is ``simulate_vjp``: the sequence
is sent as the flattened tensors of the MR-zero sequence, because only those
carry gradients.

Differences to :class:`~pulseqzero.sim.MR0`: RF pulses are instantaneous and
have no slice profile, frequency offset or pTx shims, and gradients are limited
to phantoms without a spatially varying B1+ (use ``ignore_b1=True``) and to the
``direct`` sampler.
"""

import asyncio
import hashlib
import io
import json
import os
import re
import tempfile
import threading
import warnings
from pathlib import Path

import numpy as np
import torch

GATEWAY_URL = "https://mrx-org--mcp-gateway-func.modal.run"

# Differentiable arrays of the sequence, in the order of the autograd function
_ARRAYS = ("angle", "phase", "event_time", "gradm", "adc_phase")


class Rapisim:
    """rapisim as a simulator for ``seq.simulate(sim)``.

    ``phantom`` is a bifti phantom (``.zip``) or the CAS hash of one that was
    uploaded already. It is uploaded once, on first use.

    Without ``url`` rapisim is started as a local process (``command``, default
    ``$RAPISIM_BIN`` or ``rapisim``) and files are exchanged through a
    temporary directory. With ``url`` (e.g. :data:`GATEWAY_URL`) the MCP gateway
    at that address is used, which asks for a GitHub login in the browser on
    first use.

    ``sampler``, ``kernel_size`` and ``min_mag`` are rapisim's settings. With
    ``ignore_b1`` the phantom's B1+ is ignored, which gradients need if it varies
    spatially.
    """

    def __init__(
        self,
        phantom: str | os.PathLike,
        *,
        url: str | None = None,
        command: list[str] | None = None,
        sampler: str = "direct",
        kernel_size: int = 4,
        min_mag: float = 1e-6,
        ignore_b1: bool = False,
    ):
        self.phantom = phantom
        self.url = url.rstrip("/") if url else None
        self.command = command or [os.environ.get("RAPISIM_BIN", "rapisim")]
        self.settings = dict(
            sampler=sampler, kernel_size=kernel_size, min_mag=min_mag, ignore_b1=ignore_b1
        )
        self._conn: _Connection | None = None
        self._phantom_hash: str | None = None

    # ---------------------------------------------------------------------

    def simulate(self, seq) -> torch.Tensor:
        """Signal ``[ADC samples, 1]`` of an MR-zero sequence, complex64"""
        _warn_unsupported(seq)
        reps = list(seq)
        offsets = np.cumsum([0] + [len(r.event_time) for r in reps]).astype(np.int64)
        arrays = [
            torch.stack([r.pulse.angle for r in reps]),
            torch.stack([r.pulse.phase for r in reps]),
            torch.cat([r.event_time for r in reps]),
            torch.cat([r.gradm for r in reps]),
            torch.cat([r.adc_phase for r in reps]),
        ]
        static = dict(
            adc_usage=torch.cat([r.adc_usage for r in reps]).double().numpy(),
            rep_offsets=offsets,
        )
        return _SimulateFn.apply(self, static, *arrays)

    def close(self):
        """Stops the local process / closes the connection"""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ---------------------------------------------------------------------

    def _connect(self) -> "_Connection":
        if self._conn is None:
            self._conn = _Connection(self.url, self.command)
            phantom = str(self.phantom)
            if re.fullmatch(r"[0-9a-fA-F]{64}", phantom):
                self._phantom_hash = phantom.lower()
            else:
                data = Path(self.phantom).read_bytes()
                self._phantom_hash = self._conn.put(data, Path(self.phantom).name, "application/zip")
        return self._conn

    def _call(self, tool: str, **arguments) -> bytes:
        conn = self._connect()
        return conn.call(tool, phantom=self._phantom_hash, **self.settings, **arguments)


class _SimulateFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sim: Rapisim, static: dict, *arrays: torch.Tensor):
        conn = sim._connect()
        buf = io.BytesIO()
        np.savez(buf, **static, **{n: a.detach().double().cpu().numpy() for n, a in zip(_ARRAYS, arrays)})
        ctx.sim = sim
        ctx.sequence = conn.put(buf.getvalue(), "sequence.npz", "application/x-npz")
        ctx.meta = [(a.shape, a.dtype) for a in arrays]

        signal = np.load(io.BytesIO(sim._call("simulate_diff", sequence=ctx.sequence)))
        return torch.from_numpy(signal).to(torch.complex64).reshape(-1, 1)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        sim = ctx.sim
        g = grad.detach().reshape(-1).cpu().to(torch.complex128).numpy()
        buf = io.BytesIO()
        np.save(buf, g)
        grad_hash = sim._connect().put(buf.getvalue(), "grad_signal.npy", "application/x-npy")

        grads = np.load(io.BytesIO(sim._call("simulate_vjp", sequence=ctx.sequence, grad_signal=grad_hash)))
        return (None, None) + tuple(
            torch.from_numpy(grads[n]).to(dtype).reshape(shape)
            for n, (shape, dtype) in zip(_ARRAYS, ctx.meta)
        )


def _warn_unsupported(seq):
    pulses = [rep.pulse for rep in seq]
    if any(float(p.freq_offset) != 0 for p in pulses):
        warnings.warn("rapisim ignores the frequency offset of RF pulses.", stacklevel=3)
    if any(
        getattr(p, "shim_array", None) is not None
        and not torch.equal(p.shim_array.detach().cpu(), torch.tensor([[1.0, 0.0]]))
        for p in pulses
    ):
        warnings.warn("rapisim does not support pTx shims (shim_array) yet, they are ignored.", stacklevel=3)


# =============================================================================
# MCP connection and file exchange
# =============================================================================


class _Connection:
    """One persistent MCP client on a background event loop, so that the
    synchronous torch code neither blocks it nor restarts the process."""

    def __init__(self, url: str | None, command: list[str]):
        from fastmcp import Client

        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True, name="rapisim-mcp").start()
        self._tmp = None
        self.http = None

        if url is None:
            from fastmcp.client.transports import StdioTransport

            self._tmp = tempfile.TemporaryDirectory(prefix="rapisim-cas-")
            self.cache = Path(self._tmp.name)
            env = {**os.environ, "MRX_CACHE_DIR": self._tmp.name, "MRX_BROKER_PUBLIC_URL": "http://localhost"}
            transport = StdioTransport(command=command[0], args=command[1:], env=env)
            self.client = Client(transport)
            self.prefix = ""
        else:
            import httpx2 as httpx

            auth = _oauth(url)
            self.client = Client(f"{url}/mcp", auth=auth)
            self.http = httpx.AsyncClient(auth=auth, timeout=120.0)
            self.url = url
            self.prefix = "rapisim_"
        self.run(self.client.__aenter__())

    def run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def close(self):
        try:
            self.run(self.client.__aexit__(None, None, None))
            if self.http is not None:
                self.run(self.http.aclose())
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            if self._tmp is not None:
                self._tmp.cleanup()

    # --- the content addressed store --------------------------------------

    def put(self, data: bytes, name: str, mime: str) -> str:
        if self.http is not None:
            async def upload():
                response = await self.http.post(
                    f"{self.url}/files", params={"file_name": name}, content=data, headers={"content-type": mime}
                )
                response.raise_for_status()
                return response.text

            return self.run(upload())

        sha256 = hashlib.sha256(data).hexdigest()
        (self.cache / sha256).write_bytes(data)
        (self.cache / f"{sha256}.meta").write_text(json.dumps({"file_name": name, "mime": mime, "source": "upload"}))
        return sha256

    def get(self, sha256: str) -> bytes:
        if self.http is not None:
            async def download():
                response = await self.http.get(f"{self.url}/files/{sha256}")
                response.raise_for_status()
                return response.content

            return self.run(download())
        return (self.cache / sha256).read_bytes()

    def call(self, tool: str, **arguments) -> bytes:
        """Calls a tool and returns the file its (first) resource link points to"""
        result = self.run(self.client.call_tool(self.prefix + tool, arguments))
        link = next(b for b in result.content if b.type == "resource_link")
        return self.get(str(link.uri).rsplit("/", 1)[-1])


def _oauth(url: str):
    """GitHub login of the gateway, the tokens are kept in the OS keychain"""
    from fastmcp.client.auth import OAuth
    from key_value.aio.stores.keyring import KeyringStore
    from key_value.aio.stores.keyring.store import (
        KeyringV1CollectionSanitizationStrategy,
        KeyringV1KeySanitizationStrategy,
    )

    return OAuth(
        mcp_url=f"{url}/mcp",
        scopes="read:user",
        client_name="pulseq-zero rapisim",
        token_storage=KeyringStore(
            service_name="pulseqzero-rapisim-oauth",
            key_sanitization_strategy=KeyringV1KeySanitizationStrategy(),
            collection_sanitization_strategy=KeyringV1CollectionSanitizationStrategy(),
        ),
    )
