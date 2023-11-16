from __future__ import annotations
import pytest
pytest
from bokeh.core.has_props import HasProps
from bokeh.core.properties import Int, Override, String
from tests.support.util.api import verify_all
import bokeh.core.property.include as bcpi
ALL = ('Include',)

class IsDelegate(HasProps):
    x = Int(12)
    y = String('hello')

class Test_Include:

    def test_include_with_prefix(self) -> None:
        if False:
            return 10

        class IncludesDelegateWithPrefix(HasProps):
            z = bcpi.Include(IsDelegate, prefix='z')
        o = IncludesDelegateWithPrefix()
        assert o.z_x == 12
        assert o.z_y == 'hello'
        assert not hasattr(o, 'z')
        assert not hasattr(o, 'x')
        assert not hasattr(o, 'y')
        assert 'z' not in o.properties_with_values(include_defaults=True)
        assert 'x' not in o.properties_with_values(include_defaults=True)
        assert 'y' not in o.properties_with_values(include_defaults=True)
        assert 'z_x' in o.properties_with_values(include_defaults=True)
        assert 'z_y' in o.properties_with_values(include_defaults=True)
        assert 'z_x' not in o.properties_with_values(include_defaults=False)
        assert 'z_y' not in o.properties_with_values(include_defaults=False)

    def test_include_without_prefix(self) -> None:
        if False:
            print('Hello World!')

        class IncludesDelegateWithoutPrefix(HasProps):
            z = bcpi.Include(IsDelegate)
        o = IncludesDelegateWithoutPrefix()
        assert o.x == 12
        assert o.y == 'hello'
        assert not hasattr(o, 'z')
        assert 'x' in o.properties_with_values(include_defaults=True)
        assert 'y' in o.properties_with_values(include_defaults=True)
        assert 'x' not in o.properties_with_values(include_defaults=False)
        assert 'y' not in o.properties_with_values(include_defaults=False)

    def test_include_without_prefix_using_override(self) -> None:
        if False:
            for i in range(10):
                print('nop')

        class IncludesDelegateWithoutPrefixUsingOverride(HasProps):
            z = bcpi.Include(IsDelegate)
            y = Override(default='world')
        o = IncludesDelegateWithoutPrefixUsingOverride()
        assert o.x == 12
        assert o.y == 'world'
        assert not hasattr(o, 'z')
        assert 'x' in o.properties_with_values(include_defaults=True)
        assert 'y' in o.properties_with_values(include_defaults=True)
        assert 'x' not in o.properties_with_values(include_defaults=False)
        assert 'y' not in o.properties_with_values(include_defaults=False)
Test___all__ = verify_all(bcpi, ALL)