import pytest

from quantbench.timer import measure


def test_measure_counts_calls_and_returns_repeats():
    calls = []
    out = measure(lambda: calls.append(1), warmup=2, repeats=5)
    assert len(out) == 5
    assert len(calls) == 7
    assert all(x >= 0 for x in out)


def test_measure_syncs_around_each_timed_call():
    events = []
    measure(lambda: events.append("run"), warmup=1, repeats=2, sync=lambda: events.append("sync"))
    # warmup run, sync, then (sync, run, sync) per repeat
    assert events == ["run", "sync", "sync", "run", "sync", "sync", "run", "sync"]


@pytest.mark.parametrize("kw", [{"warmup": -1}, {"repeats": 0}])
def test_measure_validates_args(kw):
    with pytest.raises(ValueError):
        measure(lambda: None, **kw)
