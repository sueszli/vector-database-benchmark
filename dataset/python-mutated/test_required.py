from __future__ import annotations
import pytest
pytest
from bokeh.core.properties import Instance, Int, List
from tests.support.util.api import verify_all
from _util_property import _TestHasProps, _TestModel
import bokeh.core.property.required as bcpr
ALL = ('Required',)

class Test_Required:

    def test_init(self) -> None:
        if False:
            i = 10
            return i + 15
        with pytest.raises(TypeError):
            bcpr.Required()

    def test_valid(self) -> None:
        if False:
            print('Hello World!')
        prop = bcpr.Required(List(Int))
        assert prop.is_valid([])
        assert prop.is_valid([1, 2, 3])

    def test_invalid(self) -> None:
        if False:
            print('Hello World!')
        prop = bcpr.Required(List(Int))
        assert not prop.is_valid(None)
        assert not prop.is_valid(-100)
        assert not prop.is_valid('yyy')
        assert not prop.is_valid([1, 2, ''])
        assert not prop.is_valid(())
        assert not prop.is_valid({})
        assert not prop.is_valid(_TestHasProps())
        assert not prop.is_valid(_TestModel())

    def test_has_ref(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        prop0 = bcpr.Required(Int)
        assert not prop0.has_ref
        prop1 = bcpr.Required(Instance(_TestModel))
        assert prop1.has_ref

    def test_str(self) -> None:
        if False:
            i = 10
            return i + 15
        prop = bcpr.Required(List(Int))
        assert str(prop) == 'Required(List(Int))'
Test___all__ = verify_all(bcpr, ALL)