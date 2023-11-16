from __future__ import annotations
import pytest
pytest
from bokeh.models import Circle, ColumnDataSource, MultiLine, StaticLayoutProvider
import bokeh.models.renderers as bmr

class TestGraphRenderer:

    def test_init_props(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        layout_provider = StaticLayoutProvider()
        renderer = bmr.GraphRenderer(layout_provider=layout_provider)
        assert renderer.x_range_name == 'default'
        assert renderer.y_range_name == 'default'
        assert renderer.node_renderer.data_source.data == dict(index=[])
        assert renderer.edge_renderer.data_source.data == dict(start=[], end=[])
        assert renderer.layout_provider == layout_provider

    def test_check_malformed_graph_source_no_errors(self) -> None:
        if False:
            i = 10
            return i + 15
        renderer = bmr.GraphRenderer()
        check = renderer._check_malformed_graph_source()
        assert check == []

    def test_check_malformed_graph_source_no_node_index(self) -> None:
        if False:
            return 10
        node_source = ColumnDataSource()
        node_renderer = bmr.GlyphRenderer(data_source=node_source, glyph=Circle())
        renderer = bmr.GraphRenderer(node_renderer=node_renderer)
        check = renderer._check_malformed_graph_source()
        assert check != []

    def test_check_malformed_graph_source_no_edge_start_or_end(self) -> None:
        if False:
            i = 10
            return i + 15
        edge_source = ColumnDataSource()
        edge_renderer = bmr.GlyphRenderer(data_source=edge_source, glyph=MultiLine())
        renderer = bmr.GraphRenderer(edge_renderer=edge_renderer)
        check = renderer._check_malformed_graph_source()
        assert check != []