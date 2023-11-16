from __future__ import annotations
import pytest
pytest
from bokeh.core.properties import Dict, Int, Interval, List, Regex, String
from bokeh.core.property.wrappers import PropertyValueDict, PropertyValueList
from tests.support.util.api import verify_all
from _util_property import _TestHasProps, _TestModel
import bokeh.core.property.either as bcpe
ALL = ('Either',)

class Test_Either:

    def test_init(self) -> None:
        if False:
            return 10
        with pytest.raises(TypeError):
            bcpe.Either()

    def test_valid(self) -> None:
        if False:
            print('Hello World!')
        prop = bcpe.Either(Interval(Int, 0, 100), Regex('^x*$'), List(Int))
        assert prop.is_valid(0)
        assert prop.is_valid(1)
        assert prop.is_valid('')
        assert prop.is_valid('xxx')
        assert prop.is_valid([])
        assert prop.is_valid([1, 2, 3])
        assert prop.is_valid(100)
        assert prop.is_valid(False)
        assert prop.is_valid(True)

    def test_invalid(self) -> None:
        if False:
            i = 10
            return i + 15
        prop = bcpe.Either(Interval(Int, 0, 100), Regex('^x*$'), List(Int))
        assert not prop.is_valid(None)
        assert not prop.is_valid(0.0)
        assert not prop.is_valid(1.0)
        assert not prop.is_valid(1.0 + 1j)
        assert not prop.is_valid(())
        assert not prop.is_valid({})
        assert not prop.is_valid(_TestHasProps())
        assert not prop.is_valid(_TestModel())
        assert not prop.is_valid(-100)
        assert not prop.is_valid('yyy')
        assert not prop.is_valid([1, 2, ''])

    def test_has_ref(self) -> None:
        if False:
            i = 10
            return i + 15
        prop = bcpe.Either(Int, Int)
        assert not prop.has_ref

    def test_str(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        prop = bcpe.Either(Int, Int)
        assert str(prop) == 'Either(Int, Int)'

    def test_wrap(self) -> None:
        if False:
            return 10
        prop = bcpe.Either(List(Int), Dict(String, Int))
        wrapped = prop.wrap([10, 20])
        assert isinstance(wrapped, PropertyValueList)
        assert prop.wrap(wrapped) is wrapped
        wrapped = prop.wrap({'foo': 10})
        assert isinstance(wrapped, PropertyValueDict)
        assert prop.wrap(wrapped) is wrapped
Test___all__ = verify_all(bcpe, ALL)