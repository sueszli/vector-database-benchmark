from __future__ import annotations
import pytest
pytest
from tests.support.util.api import verify_all
from _util_property import _TestHasProps, _TestModel
import bokeh.core.property.string as bcpr
ALL = ('Regex', 'MathString')

class Test_Regex:

    def test_init(self) -> None:
        if False:
            while True:
                i = 10
        with pytest.raises(TypeError):
            bcpr.Regex()

    def test_valid(self) -> None:
        if False:
            while True:
                i = 10
        prop = bcpr.Regex('^x*$')
        assert prop.is_valid('')
        assert prop.is_valid('x')

    def test_invalid(self) -> None:
        if False:
            print('Hello World!')
        prop = bcpr.Regex('^x*$')
        assert not prop.is_valid('xy')
        assert not prop.is_valid(None)
        assert not prop.is_valid(False)
        assert not prop.is_valid(True)
        assert not prop.is_valid(0)
        assert not prop.is_valid(1)
        assert not prop.is_valid(0.0)
        assert not prop.is_valid(1.0)
        assert not prop.is_valid(1.0 + 1j)
        assert not prop.is_valid(())
        assert not prop.is_valid([])
        assert not prop.is_valid({})
        assert not prop.is_valid(_TestHasProps())
        assert not prop.is_valid(_TestModel())

    def test_has_ref(self) -> None:
        if False:
            return 10
        prop = bcpr.Regex('')
        assert not prop.has_ref

    def test_str(self) -> None:
        if False:
            while True:
                i = 10
        prop = bcpr.Regex('')
        assert str(prop).startswith('Regex(')
Test___all__ = verify_all(bcpr, ALL)