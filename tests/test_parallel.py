from __future__ import annotations

import pytest

from ialirt_explorer.parallel import _detect_backend, parallel_analyze


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


def test_detect_backend_defaults_to_threads_with_no_scheduler() -> None:
    """The capability question is 'is there a scheduler?', and the answer
    on a laptop, in CI, inside our Docker image, and on the Render free
    tier is 'no'. Default has to be threads everywhere those are true."""

    assert _detect_backend(env={}) == "threads"


def test_detect_backend_picks_dask_when_scheduler_address_is_advertised() -> None:
    """``DASK_SCHEDULER_ADDRESS`` is Dask's canonical capability flag —
    set by ``dask-jobqueue.SLURMCluster``, ``dask-kubernetes``,
    ``dask-gateway``, ``dask-yarn``, or a manual ``dask-scheduler``.
    Presence means a real cluster is reachable; we must auto-fan-out.

    This is the entire reason the detection is capability-based instead of
    a list of vendor names: the same single signal works for SLURM,
    Kubernetes, a Docker container with a sidecar scheduler, or any future
    deploy target nobody has thought of yet.
    """

    assert (
        _detect_backend(env={"DASK_SCHEDULER_ADDRESS": "tcp://10.0.0.5:8786"})
        == "dask"
    )


def test_detect_backend_respects_manual_override() -> None:
    """``IALIRT_PARALLEL_BACKEND`` short-circuits all autodetection."""

    assert (
        _detect_backend(
            env={
                "IALIRT_PARALLEL_BACKEND": "threads",
                "DASK_SCHEDULER_ADDRESS": "tcp://10.0.0.5:8786",
            }
        )
        == "threads"
    )

    assert _detect_backend(env={"IALIRT_PARALLEL_BACKEND": "dask"}) == "dask"

    assert (
        _detect_backend(env={"IALIRT_PARALLEL_BACKEND": "kubernetes"}) == "threads"
    )


def test_detect_backend_consults_dask_config_when_using_real_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``env`` is not supplied, the detector falls through to
    ``dask.config`` / ``default_client()``. This is the path that catches
    HPC users who set ``scheduler-address`` in ``~/.config/dask/*.yaml``
    or programmatically with ``dask.config.set(...)``, without exporting
    ``DASK_SCHEDULER_ADDRESS`` to the environment.
    """

    import dask

    monkeypatch.delenv("DASK_SCHEDULER_ADDRESS", raising=False)
    monkeypatch.delenv("IALIRT_PARALLEL_BACKEND", raising=False)

    with dask.config.set({"scheduler-address": "tcp://10.0.0.5:8786"}):
        assert _detect_backend() == "dask"

    assert _detect_backend() == "threads"


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
