"""Parallel orchestration for multi-instrument I-ALiRT analysis.

Two compute backends are available, both as first-class, *installed* code
paths (not optional behind an ImportError):

  * ``backend="threads"`` (default): ``concurrent.futures.ThreadPoolExecutor``.
    Best fit on a single machine, because ``_analyze_one`` is dominated by
    an HTTP fetch against ``ialirt.imap-mission.com`` (I/O-bound); threads
    release the GIL during ``socket.recv`` so we get real concurrency with
    no extra runtime cost.

  * ``backend="dask"``: ``dask.distributed.Client``. The HPC scale-out path.
    The :mod:`dask` and :mod:`distributed` packages are listed as hard
    dependencies in ``pyproject.toml`` and ``requirements.txt`` precisely so
    this backend is not theoretical — it is exercised by the test suite
    (see ``tests/test_parallel.py::test_parallel_analyze_dask_backend``).

Why bother having Dask installed if threads work fine for the live
single-instance service? Because the moment we want to crunch the *full*
IMAP archive across compute nodes — for example on Princeton's research HPC
running Slurm — local threads stop being enough. Dask's distributed client
is the standard fit because:

  * its ``Client.submit`` / ``Future`` / ``as_completed`` API is a near
    drop-in for ``concurrent.futures``, so the orchestration loop in
    :func:`parallel_analyze` does not have to know which backend is active;
  * with ``dask-jobqueue`` (``SLURMCluster``, ``PBSCluster``, ``LSFCluster``)
    a login-node script can spin up a transient cluster of worker processes
    on the batch scheduler, hand the resulting cluster object to
    ``dask.distributed.Client``, and the same ``parallel_analyze`` call now
    fans out across the cluster instead of one machine's threads.

HPC usage (sketch, requires also ``pip install dask-jobqueue``)::

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
    client.close()
    cluster.close()

The Dask import inside ``_compute_backend`` is intentionally lazy: it keeps
backend cold-start fast for the FastAPI service (which never selects
``backend="dask"`` itself), while still giving callers a real, working
distributed code path the moment they ask for it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any, Protocol

from dask.distributed import Client as DaskClient
from dask.distributed import as_completed as dask_as_completed

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
    backend: str = "threads",
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
