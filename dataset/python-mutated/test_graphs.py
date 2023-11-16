from __future__ import annotations
import pytest
pytest
from bokeh.models.graphs import StaticLayoutProvider

def test_staticlayoutprovider_init_props() -> None:
    if False:
        return 10
    provider = StaticLayoutProvider()
    assert provider.graph_layout == {}