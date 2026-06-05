from __future__ import annotations

import pytest

from ialirt_explorer.parallel import parallel_analyze


def test_parallel_analyze_returns_expected_result_shape() -> None:
    results = parallel_analyze(["mag", "swapi"], days=1, max_workers=2)

    assert set(results) == {"mag", "swapi"}
    for result in results.values():
        assert {"data", "stats", "flagged"} == set(result)
        assert not result["data"].empty
        assert "any_anomaly" in result["flagged"]


def test_parallel_analyze_accepts_explicit_threads_backend() -> None:
    """The default backend must remain reachable via an explicit ``backend=`` arg.

    This is the contract that lets a caller flip to ``backend="dask"`` later
    without changing any other code.
    """

    results = parallel_analyze(
        ["mag"], days=1, max_workers=1, backend="threads"
    )
    assert "mag" in results


def test_parallel_analyze_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        parallel_analyze(["mag"], days=1, backend="kubernetes")


def test_parallel_analyze_dask_backend_runs_end_to_end() -> None:
    """The Dask backend is a real, installed code path — exercise it.

    Spins up an in-process ``dask.distributed.Client`` (no separate worker
    processes, no diagnostics port), runs the same per-instrument pipeline
    that the thread backend runs, and verifies the result shape matches.
    This catches regressions like a bad submit/result contract or a serialization
    issue that would silently break the HPC scale-out path.
    """

    results = parallel_analyze(
        ["mag", "swapi"], days=1, max_workers=2, backend="dask"
    )

    assert set(results) == {"mag", "swapi"}
    for result in results.values():
        assert {"data", "stats", "flagged"} == set(result)
        assert not result["data"].empty
        assert "any_anomaly" in result["flagged"]


def test_parallel_analyze_accepts_external_dask_client() -> None:
    """A pre-built ``dask.distributed.Client`` (the HPC pattern) is honored.

    On Princeton's HPC the real call would build a ``SLURMCluster`` from
    ``dask_jobqueue`` and hand its client in. Here we use a lightweight
    in-process client to verify the dispatch path without bringing up a
    scheduler we own. The function must use the passed client and must not
    shut it down on exit (the caller owns its lifecycle).
    """

    from dask.distributed import Client

    with Client(
        n_workers=1, threads_per_worker=2, processes=False, dashboard_address=None
    ) as client:
        results = parallel_analyze(
            ["mag"], days=1, max_workers=1, backend="dask", dask_client=client
        )
        assert "mag" in results
        # Caller-owned client must still be usable after parallel_analyze returns.
        assert client.status == "running"
