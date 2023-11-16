from panda3d.core import WindowProperties
import pytest

def test_winprops_ctor():
    if False:
        i = 10
        return i + 15
    props = WindowProperties()
    assert not props.is_any_specified()

def test_winprops_copy_ctor():
    if False:
        while True:
            i = 10
    props = WindowProperties()
    props.set_size(1, 2)
    props2 = WindowProperties(props)
    assert props == props2
    assert props2.get_size() == (1, 2)
    with pytest.raises(TypeError):
        WindowProperties(None)

def test_winprops_ctor_kwargs():
    if False:
        print('Hello World!')
    props = WindowProperties(size=(1, 2), origin=3)
    assert props.has_size()
    assert props.get_size() == (1, 2)
    assert props.has_origin()
    assert props.get_origin() == (3, 3)
    with pytest.raises(TypeError):
        WindowProperties(swallow_type='african')
    with pytest.raises(TypeError):
        WindowProperties(size='invalid')

def test_winprops_size_staticmethod():
    if False:
        for i in range(10):
            print('nop')
    props = WindowProperties.size(1, 2)
    assert props.has_size()
    assert props.get_size() == (1, 2)
    props = WindowProperties.size((1, 2))
    assert props.has_size()
    assert props.get_size() == (1, 2)

def test_winprops_size_property():
    if False:
        print('Hello World!')
    props = WindowProperties()
    props.set_size(1, 2)
    assert props.size == (1, 2)
    props.clear_size()
    assert props.size is None
    props.size = (4, 5)
    assert props.get_size() == (4, 5)
    props.size = None
    assert not props.has_size()

def test_winprops_maximized_property():
    if False:
        while True:
            i = 10
    props = WindowProperties()
    props.set_maximized(True)
    assert props.maximized == True
    props.clear_maximized()
    assert props.maximized is None
    props.maximized = True
    assert props.get_maximized() == True
    props.maximized = None
    assert not props.has_maximized()