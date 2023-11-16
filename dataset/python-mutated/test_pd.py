from __future__ import annotations
import pytest
pytest
import pandas as pd
from tests.support.util.api import verify_all
from _util_property import _TestHasProps, _TestModel
import bokeh.core.property.pd as bcpp
ALL = ('PandasDataFrame', 'PandasGroupBy')

class Test_PandasDataFrame:

    def test_valid(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        prop = bcpp.PandasDataFrame()
        assert prop.is_valid(pd.DataFrame())

    def test_invalid(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        prop = bcpp.PandasDataFrame()
        assert not prop.is_valid(None)
        assert not prop.is_valid(1.0 + 1j)
        assert not prop.is_valid(())
        assert not prop.is_valid([])
        assert not prop.is_valid({})
        assert not prop.is_valid(_TestHasProps())
        assert not prop.is_valid(_TestModel())

class Test_PandasGroupBy:

    def test_valid(self) -> None:
        if False:
            print('Hello World!')
        prop = bcpp.PandasGroupBy()
        assert prop.is_valid(pd.core.groupby.GroupBy(pd.DataFrame()))

    def test_invalid(self) -> None:
        if False:
            print('Hello World!')
        prop = bcpp.PandasGroupBy()
        assert not prop.is_valid(None)
        assert not prop.is_valid(1.0 + 1j)
        assert not prop.is_valid(())
        assert not prop.is_valid([])
        assert not prop.is_valid({})
        assert not prop.is_valid(_TestHasProps())
        assert not prop.is_valid(_TestModel())
Test___all__ = verify_all(bcpp, ALL)