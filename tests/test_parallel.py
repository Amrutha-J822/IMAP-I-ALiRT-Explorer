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
