from collections.abc import Generator
import contextlib
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core import accessor

def test_dirname_mixin() -> None:
    if False:
        i = 10
        return i + 15

    class X(accessor.DirNamesMixin):
        x = 1
        y: int

        def __init__(self) -> None:
            if False:
                return 10
            self.z = 3
    result = [attr_name for attr_name in dir(X()) if not attr_name.startswith('_')]
    assert result == ['x', 'z']

@contextlib.contextmanager
def ensure_removed(obj, attr) -> Generator[None, None, None]:
    if False:
        return 10
    "Ensure that an attribute added to 'obj' during the test is\n    removed when we're done\n    "
    try:
        yield
    finally:
        try:
            delattr(obj, attr)
        except AttributeError:
            pass
        obj._accessors.discard(attr)

class MyAccessor:

    def __init__(self, obj) -> None:
        if False:
            return 10
        self.obj = obj
        self.item = 'item'

    @property
    def prop(self):
        if False:
            i = 10
            return i + 15
        return self.item

    def method(self):
        if False:
            print('Hello World!')
        return self.item

@pytest.mark.parametrize('obj, registrar', [(pd.Series, pd.api.extensions.register_series_accessor), (pd.DataFrame, pd.api.extensions.register_dataframe_accessor), (pd.Index, pd.api.extensions.register_index_accessor)])
def test_register(obj, registrar):
    if False:
        return 10
    with ensure_removed(obj, 'mine'):
        before = set(dir(obj))
        registrar('mine')(MyAccessor)
        o = obj([]) if obj is not pd.Series else obj([], dtype=object)
        assert o.mine.prop == 'item'
        after = set(dir(obj))
        assert before ^ after == {'mine'}
        assert 'mine' in obj._accessors

def test_accessor_works():
    if False:
        i = 10
        return i + 15
    with ensure_removed(pd.Series, 'mine'):
        pd.api.extensions.register_series_accessor('mine')(MyAccessor)
        s = pd.Series([1, 2])
        assert s.mine.obj is s
        assert s.mine.prop == 'item'
        assert s.mine.method() == 'item'

def test_overwrite_warns():
    if False:
        for i in range(10):
            print('nop')
    match = '.*MyAccessor.*fake.*Series.*'
    with tm.assert_produces_warning(UserWarning, match=match):
        with ensure_removed(pd.Series, 'fake'):
            setattr(pd.Series, 'fake', 123)
            pd.api.extensions.register_series_accessor('fake')(MyAccessor)
            s = pd.Series([1, 2])
            assert s.fake.prop == 'item'

def test_raises_attribute_error():
    if False:
        i = 10
        return i + 15
    with ensure_removed(pd.Series, 'bad'):

        @pd.api.extensions.register_series_accessor('bad')
        class Bad:

            def __init__(self, data) -> None:
                if False:
                    i = 10
                    return i + 15
                raise AttributeError('whoops')
        with pytest.raises(AttributeError, match='whoops'):
            pd.Series([], dtype=object).bad