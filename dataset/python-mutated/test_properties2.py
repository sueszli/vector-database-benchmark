"""
Test advanced component properties.
"""
from flexx.util.testing import run_tests_if_main, skipif, skip, raises
from flexx.event.both_tester import run_in_both, this_is_js
from flexx import event
loop = event.loop

class MyObject(event.Component):
    floatpair = event.FloatPairProp(settable=True)
    enum1 = event.EnumProp(('foo', 'bar', 'spam'), settable=True)
    enum2 = event.EnumProp(('foo', 'bar', 'spam'), 'bar', settable=True)
    color = event.ColorProp('cyan', settable=True)

@run_in_both(MyObject)
def test_property_FloatPair():
    if False:
        return 10
    '\n    [0.0, 0.0]\n    [42.0, 42.0]\n    [3.2, 4.2]\n    ==\n    ? two values, not 3\n    ? 1st value cannot be\n    ? 2nd value cannot be\n    append failed\n    ----------\n    [0, 0]\n    [42, 42]\n    [3.2, 4.2]\n    ==\n    ? two values, not 3\n    ? 1st value cannot be\n    ? 2nd value cannot be\n    append failed\n    '
    m = MyObject()
    print(list(m.floatpair))
    m.set_floatpair(42)
    loop.iter()
    print(list(m.floatpair))
    m.set_floatpair((3.2, 4.2))
    loop.iter()
    print(list(m.floatpair))
    print('==')
    m.set_floatpair((3.2, 4.2, 1))
    loop.iter()
    m.set_floatpair(('hi', 1))
    loop.iter()
    m.set_floatpair((1, 'hi'))
    loop.iter()
    try:
        m.floatpair.append(9)
    except Exception:
        print('append failed')

@run_in_both(MyObject)
def test_property_Enum():
    if False:
        print('Hello World!')
    "\n    FOO\n    BAR\n    SPAM\n    FOO\n    ? TypeError\n    ? Invalid value for enum 'enum1': EGGS\n    "
    m = MyObject()
    print(m.enum1)
    print(m.enum2)
    m = MyObject(enum1='spam')
    print(m.enum1)
    m.set_enum1('foo')
    loop.iter()
    print(m.enum1)
    m.set_enum1(3)
    loop.iter()
    m.set_enum1('eggs')
    loop.iter()

@run_in_both(MyObject)
def test_property_Color1():
    if False:
        i = 10
        return i + 15
    '\n    #00ffff 1.0\n    [0.0, 1.0, 1.0, 1.0]\n    rgba(0,255,255,1)\n    rgba(0,255,255,0.25)\n    ----------\n    #00ffff 1\n    [0, 1, 1, 1]\n    rgba(0,255,255,1)\n    rgba(0,255,255,0.25)\n    '
    m = MyObject()
    print(m.color.hex, m.color.alpha)
    print(list(m.color.t))
    print(m.color.css)
    m.set_color((0, 1, 1, 0.25))
    loop.iter()
    print(m.color.css)

@run_in_both(MyObject)
def test_property_Color2():
    if False:
        return 10
    '\n    ? #00ffff 1\n    ? #ff8800 1\n    ? #f48404 1\n    ? #ff8800 0.5\n    ? #f48404 0.5\n    xx\n    ? #00ff00 1\n    ? #ffff00 0.5\n    xx\n    ? #ffff00 1\n    ? #ff00ff 1\n    xx\n    ? #ff0000 1\n    ? #00ff00 0.5\n    '
    m = MyObject()
    print(m.color.hex, m.color.alpha)
    m.set_color('#f80')
    loop.iter()
    print(m.color.hex, m.color.alpha)
    m.set_color('#f48404')
    loop.iter()
    print(m.color.hex, m.color.alpha)
    m.set_color('#f808')
    loop.iter()
    print(m.color.hex, m.color.alpha)
    m.set_color('#f4840488')
    loop.iter()
    print(m.color.hex, m.color.alpha)
    print('xx')
    m.set_color('rgb(0, 255, 0)')
    loop.iter()
    print(m.color.hex, m.color.alpha)
    m.set_color('rgba(255, 255, 0, 0.5)')
    loop.iter()
    print(m.color.hex, m.color.alpha)
    print('xx')
    m.set_color('yellow')
    loop.iter()
    print(m.color.hex, m.color.alpha)
    m.set_color('magenta')
    loop.iter()
    print(m.color.hex, m.color.alpha)
    print('xx')
    m.set_color((1, 0, 0, 1))
    loop.iter()
    print(m.color.hex, m.color.alpha)
    m.set_color((0, 1, 0, 0.5))
    loop.iter()
    print(m.color.hex, m.color.alpha)
run_tests_if_main()