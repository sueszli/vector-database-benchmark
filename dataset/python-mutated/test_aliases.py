from __future__ import annotations
import pytest
pytest
import datetime
import numpy as np
import pandas as pd
from tests.support.util.api import verify_all
from _util_property import _TestHasProps, _TestModel
import bokeh.core.property.aliases as bcpc
ALL = ('CoordinateLike',)

class Test_CoordinateLike:

    def test_valid(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        prop = bcpc.CoordinateLike()
        assert prop.is_valid(-1.0)
        assert prop.is_valid(-1)
        assert prop.is_valid(0)
        assert prop.is_valid(1)
        assert prop.is_valid(0.0)
        assert prop.is_valid(1.0)
        assert prop.is_valid('2020-01-11T13:00:00')
        assert prop.is_valid('2020-01-11')
        assert prop.is_valid(datetime.datetime.now())
        assert prop.is_valid(datetime.time(10, 12))
        assert prop.is_valid(np.datetime64('2020-01-11'))
        assert prop.is_valid(pd.Timestamp('2010-01-11'))
        assert prop.is_valid('')
        assert prop.is_valid(('', ''))
        assert prop.is_valid(('', '', ''))
        assert prop.is_valid(False)
        assert prop.is_valid(True)

    def test_invalid(self) -> None:
        if False:
            print('Hello World!')
        prop = bcpc.CoordinateLike()
        assert not prop.is_valid(None)
        assert not prop.is_valid(1.0 + 1j)
        assert not prop.is_valid(())
        assert not prop.is_valid([])
        assert not prop.is_valid({})
        assert not prop.is_valid(_TestHasProps())
        assert not prop.is_valid(_TestModel())
Test___all__ = verify_all(bcpc, ALL)