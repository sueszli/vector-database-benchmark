from __future__ import annotations
import pytest
pytest
from tests.support.util.api import verify_all
from _util_property import _TestHasProps, _TestModel
import bokeh.core.property.json as bcpj
ALL = ('JSON',)

class Test_JSON:

    def test_valid(self) -> None:
        if False:
            print('Hello World!')
        prop = bcpj.JSON()
        assert prop.is_valid('[]')
        assert prop.is_valid('[{"foo": 10}]')

    def test_invalid(self) -> None:
        if False:
            i = 10
            return i + 15
        prop = bcpj.JSON()
        assert not prop.is_valid(None)
        assert not prop.is_valid('')
        assert not prop.is_valid('foo')
        assert not prop.is_valid('[]]')
        assert not prop.is_valid("[{'foo': 10}]")
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
        prop = bcpj.JSON()
        assert not prop.has_ref

    def test_str(self) -> None:
        if False:
            while True:
                i = 10
        prop = bcpj.JSON()
        assert str(prop) == 'JSON'
Test___all__ = verify_all(bcpj, ALL)