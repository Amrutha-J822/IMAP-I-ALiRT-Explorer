"""Parallel orchestration for multi-instrument I-ALiRT analysis.

The default backend is :class:`concurrent.futures.ThreadPoolExecutor`, which
is the right pick for this project's workload: ``_analyze_one`` is dominated
by an HTTP fetch against ``ialirt.imap-mission.com`` (I/O-bound), so threads
release the GIL during ``socket.recv`` and we get real concurrency on a
single machine without any extra runtime dependencies.

This module is also designed so we can swap that backend for **Dask** without
changing any caller. The reason that matters: if we ever scale this project
up to chew through the *full* IMAP archive on a research cluster
(e.g. Princeton's HPC running Slurm), local threads stop being enough — we
want to spread the per-instrument work across many compute nodes. Dask's
``distributed`` client is the standard fit because:

  * its ``Client.submit`` / ``Future`` / ``as_completed`` API is a near
    drop-in for ``concurrent.futures``, so the orchestration loop below does
    not have to know which backend is active;
  * with ``dask-jobqueue`` (``SLURMCluster``, ``PBSCluster``, ``LSFCluster``)
    you can spin up a transient cluster of worker processes on the HPC's
    batch scheduler from a login node, hand the resulting cluster object to
    ``dask.distributed.Client``, and the same ``parallel_analyze`` call now
    fans out across the cluster instead of one machine's threads.

To actually use the Dask backend you would install
``pip install 'dask[distributed]'`` (plus ``dask-jobqueue`` for HPC) and
call ``parallel_analyze(..., backend="dask")``. If Dask is not installed,
that call raises a clear error and the default thread backend is unaffected.

Example sketch (not exercised in this repo because the cluster is not
available in CI):

    from dask_jobqueue import SLURMCluster
    from dask.distributed import Client
    cluster = SLURMCluster(cores=4, memory="16GB", queue="cpu", walltime="1:00:00")
    cluster.scale(jobs=8)
    client = Client(cluster)
    results = parallel_analyze(
        ["mag", "swapi", "swe", "hit", "codice_lo", "codice_hi"],
        days=30,
        backend="dask",
        dask_client=client,
    )
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any, Protocol

from ialirt_explorer.analytics import analyze, calibrate_mag, detect_anomalies
from ialirt_explorer.ingestion import IALIRT_INSTRUMENTS, fetch_latest


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
    :class:`dask.distributed.Client` happen to support this exact surface,
    which is why ``parallel_analyze`` does not need to special-case them at
    the orchestration site.
    """

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any: ...


@contextmanager
def _compute_backend(
    backend: str,
    *,
    max_workers: int,
    dask_client: Any | None = None,
) -> Iterator[tuple[_ComputeBackend, Callable[[Iterable[Any]], Iterator[Any]]]]:
    """Yield (executor, as_completed_fn) for the requested backend.

    The returned ``as_completed_fn`` accepts an iterable of futures and
    yields them in completion order, matching the semantics of
    :func:`concurrent.futures.as_completed`. We tunnel it through this
    context because Dask exposes its own ``as_completed`` import path.
    """

    if backend == "threads":
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            yield executor, as_completed
        return

    if backend == "dask":
        try:
            from dask.distributed import Client  # type: ignore[import-not-found]
            from dask.distributed import (  # type: ignore[import-not-found]
                as_completed as dask_as_completed,
            )
        except ImportError as exc:  # pragma: no cover - exercised only when dask is missing
            raise RuntimeError(
                "Dask backend requested but `dask[distributed]` is not installed. "
                "Install with `pip install 'dask[distributed]'` for a local cluster, "
                "or `pip install 'dask[distributed]' dask-jobqueue` if you intend to "
                "run on an HPC scheduler (Slurm/PBS/LSF). See module docstring for "
                "a SLURMCluster example."
            ) from exc

        # Honor a user-supplied client (e.g. one already bound to a SLURMCluster
        # on Princeton's HPC). Only spin up a default local cluster when none
        # was passed in.
        owned_client = dask_client is None
        client = dask_client or Client(n_workers=max_workers, threads_per_worker=1)
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
    backend: str = "threads",
    dask_client: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """Fetch, summarize, and flag multiple instruments concurrently.

    Parameters
    ----------
    instruments:
        Iterable of instrument names. Defaults to every supported instrument.
    days:
        Lookback window passed to :func:`fetch_latest`.
    max_workers:
        Concurrency cap. For ``backend="threads"`` it sizes the
        ``ThreadPoolExecutor``; for ``backend="dask"`` without an explicit
        ``dask_client`` it sizes the default local cluster.
    backend:
        ``"threads"`` (default) uses ``concurrent.futures``; ``"dask"`` uses
        ``dask.distributed`` and is intended for HPC fan-out. The thread
        backend has no extra dependencies.
    dask_client:
        Optional pre-built ``dask.distributed.Client`` (for example, one
        bound to a ``SLURMCluster``). Only consulted when ``backend="dask"``.
    """

    selected = [item.lower() for item in (instruments or IALIRT_INSTRUMENTS.keys())]
    workers = max_workers or min(4, len(selected))
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
