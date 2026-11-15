import math

import pytest

from quantbench.stats import percentile, summarize


def test_percentile_matches_numpy_linear():
    xs = [15.0, 20.0, 35.0, 40.0, 50.0]
    # numpy.percentile(xs, [0, 25, 40, 50, 100]) -> 15, 20, 29, 35, 50
    assert percentile(xs, 0) == 15
    assert percentile(xs, 25) == 20
    assert math.isclose(percentile(xs, 40), 29.0)
    assert percentile(xs, 50) == 35
    assert percentile(xs, 100) == 50


def test_percentile_unsorted_input_and_single_sample():
    assert percentile([3.0, 1.0, 2.0], 50) == 2.0
    assert percentile([7.0], 99) == 7.0


@pytest.mark.parametrize("q", [-1, 100.1])
def test_percentile_rejects_bad_q(q):
    with pytest.raises(ValueError):
        percentile([1.0], q)


def test_percentile_rejects_empty():
    with pytest.raises(ValueError):
        percentile([], 50)


def test_summarize_basic():
    s = summarize([1.0, 2.0, 3.0, 4.0])
    assert s.n == 4
    assert s.mean == 2.5
    assert math.isclose(s.std, math.sqrt(5 / 3))  # sample std, ddof=1
    assert (s.min, s.max) == (1.0, 4.0)
    assert s.p50 == 2.5


def test_summarize_single_sample_has_zero_std():
    s = summarize([5.0])
    assert s.std == 0.0 and s.p99 == 5.0


def test_summary_to_dict_keys():
    d = summarize([1.0, 2.0]).to_dict()
    assert set(d) == {"n", "mean", "std", "min", "p50", "p90", "p99", "max"}
