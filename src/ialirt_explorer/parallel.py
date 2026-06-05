"""Parallel orchestration for multi-instrument I-ALiRT analysis.

Two compute backends are available, both as first-class, *installed* code
paths:

  * ``"threads"``: ``concurrent.futures.ThreadPoolExecutor``. The right
    pick on any single-machine host, because ``_analyze_one`` is dominated
    by an HTTP fetch against ``ialirt.imap-mission.com`` (I/O-bound);
    threads release the GIL during ``socket.recv`` so we get real
    concurrency with no extra runtime cost.

  * ``"dask"``: ``dask.distributed.Client``. The scale-out path for any
    environment where someone has already provisioned a Dask cluster, e.g.
    a ``dask-jobqueue.SLURMCluster`` on Princeton's research HPC, a
    ``dask-kubernetes`` deploy, or a manually started ``dask-scheduler``.

Auto-detection (capability-based, no hardcoded vendor names)
------------------------------------------------------------
:func:`parallel_analyze` defaults to ``backend=None``, which calls
:func:`_detect_backend`. The selection is intentionally based on a single
*capability* signal — **is a Dask scheduler advertised to this process?**
— rather than on enumerating "if I'm on Render do X, if I'm on Slurm do Y".
That kind of enumeration breaks the first time the code runs somewhere new
(Docker on a private VM, Kubernetes, a bare-metal workstation), so we don't
do it. The full decision logic is:

  1. ``IALIRT_PARALLEL_BACKEND`` env var → honored verbatim
     (``"threads"`` or ``"dask"``). Manual override.

  2. A Dask scheduler address is resolvable through Dask's *own*
     configuration chain — ``DASK_SCHEDULER_ADDRESS`` env var, YAML files
     in ``~/.config/dask`` or ``/etc/dask``, or programmatic
     ``dask.config.set``. This is the canonical "I have a cluster"
     capability flag that every dask-jobqueue / dask-kubernetes /
     dask-gateway / dask-yarn deployment uses, regardless of host.
     → ``"dask"``.

  3. A ``dask.distributed.Client`` has already been constructed in this
     process (the standard pattern on HPC: ``Client(SLURMCluster(...))``
     before calling :func:`parallel_analyze`).
     → ``"dask"``.

  4. Otherwise → ``"threads"``.

This gives correct behavior everywhere with no host-name list to maintain:

  * Laptop, CI, Docker container without cluster config, the live Render
    deployment, a Vercel preview: nothing advertises a scheduler →
    threads.
  * Inside a Slurm batch job on Princeton HPC, after the user has spun up
    a ``SLURMCluster`` and exported ``DASK_SCHEDULER_ADDRESS`` (the
    standard dask-jobqueue pattern): scheduler is advertised → dask.
  * Any future deploy target: same code, no edit required.

HPC usage with an externally-built cluster (recommended pattern)::

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

import dask
from dask.distributed import Client as DaskClient
from dask.distributed import as_completed as dask_as_completed

from ialirt_explorer.analytics import analyze, calibrate_mag, detect_anomalies
from ialirt_explorer.ingestion import IALIRT_INSTRUMENTS, fetch_latest

log = logging.getLogger(__name__)

_BACKEND_OVERRIDE_ENV = "IALIRT_PARALLEL_BACKEND"


def _dask_scheduler_is_active() -> bool:
    """Return True iff a Dask scheduler is reachable by this process.

    Two sources, in order:

    1. ``dask.config.get("scheduler-address")``. Dask resolves this from
       the ``DASK_SCHEDULER_ADDRESS`` env var, YAML files in
       ``~/.config/dask`` or ``/etc/dask``, and programmatic
       ``dask.config.set(...)`` calls. Every cluster-spawning tool in the
       Dask ecosystem (``dask-jobqueue``, ``dask-kubernetes``,
       ``dask-gateway``, ``dask-yarn``, manual ``dask-scheduler``)
       advertises through one of those channels.
    2. ``dask.distributed.default_client()``. If user code already
       constructed a ``Client(cluster)`` in this process — the canonical
       HPC pattern — that client is registered as the default for the
       process and we can detect it without touching any env vars.

    Kept as a separate function so tests can stub it independently of the
    pure env-var detection path.
    """

    try:
        if dask.config.get("scheduler-address", default=None):
            return True
    except Exception:  # noqa: BLE001 - dask.config may raise on malformed YAML
        log.debug("dask.config.get('scheduler-address') raised", exc_info=True)

    try:
        from dask.distributed import default_client

        default_client()
        return True
    except ValueError:
        return False
    except Exception:  # noqa: BLE001
        log.debug("default_client() raised unexpectedly", exc_info=True)
        return False


def _detect_backend(env: dict[str, str] | None = None) -> str:
    """Pick the compute backend by capability, not by vendor name.

    Decision order (see module docstring for the full rationale):
      1. ``IALIRT_PARALLEL_BACKEND`` override (``"threads"`` or ``"dask"``).
      2. ``DASK_SCHEDULER_ADDRESS`` advertised in the environment.
      3. A scheduler reachable through Dask's own config / default client.
      4. Default: ``"threads"``.

    The ``env`` parameter exists so tests can pass a synthetic environment
    dict. When ``env`` is supplied, step (3) is skipped — we don't want a
    real ambient ``DASK_SCHEDULER_ADDRESS`` (or a Dask YAML file on the
    developer's laptop) to bleed into a deterministic unit test.
    """

    environ = env if env is not None else os.environ

    override = environ.get(_BACKEND_OVERRIDE_ENV)
    if override:
        if override in {"threads", "dask"}:
            return override
        log.warning(
            "%s=%r is not a recognized backend; falling back to autodetection.",
            _BACKEND_OVERRIDE_ENV,
            override,
        )

    if environ.get("DASK_SCHEDULER_ADDRESS"):
        return "dask"

    if env is None and _dask_scheduler_is_active():
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
