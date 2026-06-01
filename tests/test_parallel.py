from __future__ import annotations

from ialirt_explorer.parallel import parallel_analyze


def test_parallel_analyze_returns_expected_result_shape() -> None:
    results = parallel_analyze(["mag", "swapi"], days=1, max_workers=2)

    assert set(results) == {"mag", "swapi"}
    for result in results.values():
        assert {"data", "stats", "flagged"} == set(result)
        assert not result["data"].empty
        assert "any_anomaly" in result["flagged"]
