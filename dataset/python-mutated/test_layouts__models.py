from __future__ import annotations
import pytest
pytest
from bokeh.models import ColumnDataSource
from bokeh.models.ui import Tooltip
from bokeh.plotting import figure
from bokeh.models.layouts import Row, Column, LayoutDOM, TabPanel

def check_props(layout: LayoutDOM):
    if False:
        i = 10
        return i + 15
    assert layout.width is None
    assert layout.height is None
    assert layout.children == []

def check_props_with_sizing_mode(layout: LayoutDOM):
    if False:
        while True:
            i = 10
    assert layout.width is None
    assert layout.height is None
    assert layout.children == []
    assert layout.sizing_mode is None

def check_children_prop(layout_callable: type[Row | Column]):
    if False:
        print('Hello World!')
    components = [Row(), Column(), figure()]
    layout1 = layout_callable(*components)
    assert layout1.children == components
    layout2 = layout_callable(children=components)
    assert layout2.children == components
    with pytest.raises(ValueError):
        layout_callable(children=[ColumnDataSource()])

def test_Row() -> None:
    if False:
        return 10
    check_props_with_sizing_mode(Row())
    check_children_prop(Row)

def test_Column() -> None:
    if False:
        return 10
    check_props_with_sizing_mode(Column())
    check_children_prop(Column)

def test_LayoutDOM_css_classes() -> None:
    if False:
        print('Hello World!')
    m = LayoutDOM()
    assert m.css_classes == []
    m.css_classes = ['foo']
    assert m.css_classes == ['foo']
    m.css_classes = ('bar',)
    assert m.css_classes == ['bar']

def test_LayoutDOM_backgroud() -> None:
    if False:
        while True:
            i = 10
    obj = LayoutDOM(background='#aabbccff')
    assert obj.styles['background-color'] == '#aabbccff'
    obj = LayoutDOM()
    assert 'background-color' not in obj.styles
    obj.background = '#aabbccff'
    assert obj.styles['background-color'] == '#aabbccff'

def test_TabPanel_no_tooltip() -> None:
    if False:
        while True:
            i = 10
    p1 = figure(width=300, height=300)
    panel = TabPanel(child=p1, title='test panel')
    assert panel.title == 'test panel'
    assert panel.child is not None
    assert panel.tooltip is None

def test_TabPanel_tooltip() -> None:
    if False:
        print('Hello World!')
    p1 = figure(width=300, height=300)
    panel = TabPanel(child=p1, title='test panel', tooltip=Tooltip(content='test tooltip'))
    assert panel.tooltip is not None