import pytest
from plotly.graph_objects import *
from plotly.missing_ipywidgets import FigureWidget as FigureWidgetMissingIPywidgets
try:
    import ipywidgets as _ipywidgets
    from packaging.version import Version as _Version
    if _Version(_ipywidgets.__version__) >= _Version('7.0.0'):
        missing_ipywidgets = False
    else:
        raise ImportError()
except Exception:
    missing_ipywidgets = True
if missing_ipywidgets:

    def test_import_figurewidget_without_ipywidgets():
        if False:
            for i in range(10):
                print('nop')
        assert FigureWidget is FigureWidgetMissingIPywidgets
        with pytest.raises(ImportError):
            FigureWidget()
else:

    def test_import_figurewidget_with_ipywidgets():
        if False:
            while True:
                i = 10
        from plotly.graph_objs._figurewidget import FigureWidget as FigureWidgetWithIPywidgets
        assert FigureWidget is FigureWidgetWithIPywidgets
        fig = FigureWidget()
        assert isinstance(fig, FigureWidgetWithIPywidgets)