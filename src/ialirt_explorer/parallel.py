"""Parallel orchestration for multi-instrument I-ALiRT analysis.

Two compute backends are available, both as first-class, *installed* code
paths:

  * ``"threads"``: ``concurrent.futures.ThreadPoolExecutor``. The right pick
    on a laptop, a Vercel preview, or the live Render container, because
    ``_analyze_one`` is dominated by an HTTP fetch against
    ``ialirt.imap-mission.com`` (I/O-bound); threads release the GIL during
    ``socket.recv`` so we get real concurrency with no extra runtime cost.

  * ``"dask"``: ``dask.distributed.Client``. The HPC scale-out path,
    intended for running this analysis under a batch scheduler like Slurm
    on Princeton's research HPC. The :mod:`dask` and :mod:`distributed`
    packages are hard dependencies in ``pyproject.toml``; the test suite
    exercises the backend end-to-end (see ``tests/test_parallel.py``).

Auto-detection
--------------
:func:`parallel_analyze` defaults to ``backend=None``, which means
*pick the right one for this machine*. The selection is made by
:func:`_detect_backend`:

  1. If the env var ``IALIRT_PARALLEL_BACKEND`` is set, honor it verbatim
     (``"threads"`` or ``"dask"``). This is the manual override knob.
  2. If a known platform-as-a-service env var is present
     (``RENDER``, ``DYNO`` for Heroku, ``FLY_APP_NAME``, ``VERCEL``), force
     ``"threads"``. Those hosts are single small containers and have no
     business spinning up a multi-worker Dask cluster — this guard exists
     specifically so the live Render free-tier deploy *never* triggers Dask
     even if some other env var sneaks in.
  3. If a known HPC batch-scheduler env var is present
     (``SLURM_JOB_ID`` / ``PBS_JOBID`` / ``LSB_JOBID`` / ``SGE_TASK_ID``),
     the process is running inside an allocated job on a cluster — switch
     to ``"dask"``.
  4. Otherwise fall back to ``"threads"``.

So in practice:

  * Local development, CI, Vercel previews, Render production
        → threads (no env vars match).
  * ``sbatch run_analysis.slurm`` on Princeton HPC, where the script calls
    :func:`parallel_analyze` from inside the Slurm allocation
        → dask (``SLURM_JOB_ID`` is set by Slurm).
  * Manual override on any host:
        ``IALIRT_PARALLEL_BACKEND=dask python -m ...``

HPC usage with an externally-built cluster (the recommended pattern, since
the auto-detection's default local cluster is single-node)::

    from dask_jobqueue import SLURMCluster
    from dask.distributed import Client
    cluster = SLURMCluster(cores=4, memory="16GB", queue="cpu", walltime="1:00:00")
    cluster.scale(jobs=8)
    client = Client(cluster)
    results = parallel_analyze(
        ["mag", "swapi", "swe", "hit", "codice_lo", "codice_hi"],
        days=30,
        dask_client=client,  # auto-detect picks "dask"; client is honored
    )
    client.close()
    cluster.close()
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any, Protocol

from dask.distributed import Client as DaskClient
from dask.distributed import as_completed as dask_as_completed

from ialirt_explorer.analytics import analyze, calibrate_mag, detect_anomalies
from ialirt_explorer.ingestion import IALIRT_INSTRUMENTS, fetch_latest

log = logging.getLogger(__name__)

_BACKEND_OVERRIDE_ENV = "IALIRT_PARALLEL_BACKEND"

# Hosts where a multi-worker cluster would be wrong on its face: small
# single-container PaaS deploys. We *always* fall back to threads here even
# if a stale HPC env var leaks into the process. This is the explicit guard
# that keeps the live Render free-tier service from ever instantiating Dask.
_PAAS_HOST_ENV_VARS: tuple[str, ...] = (
    "RENDER",
    "RENDER_SERVICE_NAME",
    "DYNO",  # Heroku
    "FLY_APP_NAME",  # Fly.io
    "VERCEL",
)

# HPC batch schedulers set these inside an allocated job. Presence of any
# one means we are running on real cluster nodes and should fan out.
_HPC_SCHEDULER_ENV_VARS: tuple[str, ...] = (
    "SLURM_JOB_ID",  # Slurm (Princeton HPC default)
    "PBS_JOBID",  # PBS / Torque
    "LSB_JOBID",  # IBM LSF
    "SGE_TASK_ID",  # Sun/Univa/Altair Grid Engine
)


def _detect_backend(env: dict[str, str] | None = None) -> str:
    """Pick the compute backend appropriate for the current machine.

    See module docstring for the full decision table. The ``env`` parameter
    exists so tests can pass a synthetic environment dict.
    """

    environ = env if env is not None else os.environ

    override = environ.get(_BACKEND_OVERRIDE_ENV)
    if override:
        if override not in {"threads", "dask"}:
            log.warning(
                "%s=%r is not a recognized backend; falling back to autodetection.",
                _BACKEND_OVERRIDE_ENV,
                override,
            )
        else:
            return override

    if any(environ.get(var) for var in _PAAS_HOST_ENV_VARS):
        return "threads"

    if any(environ.get(var) for var in _HPC_SCHEDULER_ENV_VARS):
        return "dask"

    return "threads"


def _analyze_one(instrument: str, days: int) -> dict[str, Any]:
    """Per-instrument unit of work; executed on whatever backend is active."""

    frame = fetch_latest(instrument, days=days)
    analysis_frame = calibrate_mag(frame) if instrument == "mag" else frame
    return {
        "data": analysis_frame,
        "stats": analyze(analysis_frame),
        "flagged": detect_anomalies(analysis_frame, instrument),
    }


class _ComputeBackend(Protocol):
    """Minimal subset of executor behavior shared by threads and Dask.

    Both :class:`concurrent.futures.ThreadPoolExecutor` and
    :class:`dask.distributed.Client` support this exact surface, which is
    why :func:`parallel_analyze` does not need to special-case them at the
    orchestration site.
    """

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any: ...


@contextmanager
def _compute_backend(
    backend: str,
    *,
    max_workers: int,
    dask_client: DaskClient | None = None,
) -> Iterator[tuple[_ComputeBackend, Callable[[Iterable[Any]], Iterator[Any]]]]:
    """Yield ``(executor, as_completed_fn)`` for the requested backend.

    The returned ``as_completed_fn`` accepts an iterable of futures and
    yields them in completion order, matching the semantics of
    :func:`concurrent.futures.as_completed`. We tunnel it through this
    context because Dask exposes its own ``as_completed`` callable.
    """

    if backend == "threads":
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            yield executor, as_completed
        return

    if backend == "dask":
        # Honor a user-supplied client (e.g. one already bound to a
        # SLURMCluster on Princeton's HPC). Spin up a default local cluster
        # only when none was passed in, and tear it down when we own it.
        owned_client = dask_client is None
        client = dask_client or DaskClient(
            n_workers=max_workers,
            threads_per_worker=1,
            processes=False,  # in-process workers: light enough for CI
            dashboard_address=None,  # do not attempt to bind the diagnostics port
        )
        try:
            yield client, dask_as_completed
        finally:
            if owned_client:
                client.close()
        return

    raise ValueError(
        f"Unknown backend {backend!r}; expected 'threads' (default) or 'dask'."
    )


def parallel_analyze(
    instruments: Iterable[str] | None = None,
    *,
    days: int = 3,
    max_workers: int | None = None,
    backend: str | None = None,
    dask_client: DaskClient | None = None,
) -> dict[str, dict[str, Any]]:
    """Fetch, summarize, and flag multiple instruments concurrently.

    Parameters
    ----------
    instruments:
        Iterable of instrument names. Defaults to every supported instrument.
    days:
        Lookback window passed to :func:`fetch_latest`.
    max_workers:
        Concurrency cap. For threads it sizes the ``ThreadPoolExecutor``;
        for Dask without an explicit ``dask_client`` it sizes the default
        local cluster.
    backend:
        ``None`` (default) calls :func:`_detect_backend` to auto-pick the
        right compute backend for this machine — threads on a laptop or PaaS
        host, Dask inside an HPC Slurm/PBS/LSF allocation. Pass
        ``"threads"`` or ``"dask"`` to override.
    dask_client:
        Optional pre-built ``dask.distributed.Client``, typically one bound
        to a ``SLURMCluster`` from ``dask-jobqueue``. Implies ``backend="dask"``
        when supplied; honored regardless of auto-detection.
    """

    selected = [item.lower() for item in (instruments or IALIRT_INSTRUMENTS.keys())]
    workers = max_workers or min(4, len(selected))

    if dask_client is not None and backend is None:
        # Passing a Dask client is itself a signal that the caller wants
        # the Dask path; auto-detect would otherwise need to guess.
        backend = "dask"
    elif backend is None:
        backend = _detect_backend()
        log.info("parallel_analyze auto-selected backend=%s", backend)

    results: dict[str, dict[str, Any]] = {}

    with _compute_backend(
        backend, max_workers=workers, dask_client=dask_client
    ) as (executor, wait_for):
        futures = {
            executor.submit(_analyze_one, instrument, days): instrument
            for instrument in selected
        }
        for future in wait_for(futures):
            instrument = futures[future]
            results[instrument] = future.result()

    return results
