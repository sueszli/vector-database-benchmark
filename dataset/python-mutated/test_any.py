from __future__ import annotations
import pytest
pytest
from tests.support.util.api import verify_all
from _util_property import _TestHasProps, _TestModel
import bokeh.core.property.any as bcpa
ALL = ('Any', 'AnyRef')

class Test_Any:

    def test_valid(self) -> None:
        if False:
            print('Hello World!')
        prop = bcpa.Any()
        assert prop.is_valid(None)
        assert prop.is_valid(False)
        assert prop.is_valid(True)
        assert prop.is_valid(0)
        assert prop.is_valid(1)
        assert prop.is_valid(0.0)
        assert prop.is_valid(1.0)
        assert prop.is_valid(1.0 + 1j)
        assert prop.is_valid('')
        assert prop.is_valid(())
        assert prop.is_valid([])
        assert prop.is_valid({})
        assert prop.is_valid(_TestHasProps())
        assert prop.is_valid(_TestModel())

    def test_invalid(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_has_ref(self) -> None:
        if False:
            print('Hello World!')
        prop = bcpa.Any()
        assert not prop.has_ref

class Test_AnyRef:

    def test_valid(self) -> None:
        if False:
            while True:
                i = 10
        prop = bcpa.AnyRef()
        assert prop.is_valid(None)
        assert prop.is_valid(False)
        assert prop.is_valid(True)
        assert prop.is_valid(0)
        assert prop.is_valid(1)
        assert prop.is_valid(0.0)
        assert prop.is_valid(1.0)
        assert prop.is_valid(1.0 + 1j)
        assert prop.is_valid('')
        assert prop.is_valid(())
        assert prop.is_valid([])
        assert prop.is_valid({})
        assert prop.is_valid(_TestHasProps())
        assert prop.is_valid(_TestModel())

    def test_invalid(self) -> None:
        if False:
            return 10
        pass

    def test_has_ref(self) -> None:
        if False:
            return 10
        prop = bcpa.AnyRef()
        assert prop.has_ref
Test___all__ = verify_all(bcpa, ALL)