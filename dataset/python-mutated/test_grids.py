from __future__ import annotations
import pytest
pytest
from bokeh.models import FixedTicker, LinearAxis
import bokeh.models.grids as bmg

def test_ticker_accepts_number_sequences() -> None:
    if False:
        i = 10
        return i + 15
    g = bmg.Grid(ticker=[-10, 0, 10, 20.7])
    assert isinstance(g.ticker, FixedTicker)
    assert g.ticker.ticks == [-10, 0, 10, 20.7]
    g = bmg.Grid()
    g.ticker = [-10, 0, 10, 20.7]
    assert isinstance(g.ticker, FixedTicker)
    assert g.ticker.ticks == [-10, 0, 10, 20.7]

def test_ticker_accepts_axis() -> None:
    if False:
        for i in range(10):
            print('nop')
    g = bmg.Grid(axis=LinearAxis())
    assert isinstance(g.axis, LinearAxis)
    g = bmg.Grid()
    g.axis = LinearAxis()
    assert isinstance(g.axis, LinearAxis)