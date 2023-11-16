from __future__ import annotations
import pytest
pytest
from bokeh.model import collect_models
from bokeh.models.util import generate_structure_plot
from bokeh.plotting import figure

def test_structure(nx):
    if False:
        while True:
            i = 10
    f = figure(width=400, height=400)
    f.line(x=[1, 2, 3], y=[1, 2, 3])
    K = generate_structure_plot(f)
    assert 45 == len(collect_models(K))