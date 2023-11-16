"""
Tests for `__init_subclass__` related functionality.
"""
import attr

def test_init_subclass_vanilla(slots):
    if False:
        i = 10
        return i + 15
    '\n    `super().__init_subclass__` can be used if the subclass is not an attrs\n    class both with dict and slotted classes.\n    '

    @attr.s(slots=slots)
    class Base:

        def __init_subclass__(cls, param, **kw):
            if False:
                for i in range(10):
                    print('nop')
            super().__init_subclass__(**kw)
            cls.param = param

    class Vanilla(Base, param='foo'):
        pass
    assert 'foo' == Vanilla().param

def test_init_subclass_attrs():
    if False:
        while True:
            i = 10
    '\n    `__init_subclass__` works with attrs classes as long as slots=False.\n    '

    @attr.s(slots=False)
    class Base:

        def __init_subclass__(cls, param, **kw):
            if False:
                for i in range(10):
                    print('nop')
            super().__init_subclass__(**kw)
            cls.param = param

    @attr.s
    class Attrs(Base, param='foo'):
        pass
    assert 'foo' == Attrs().param

def test_init_subclass_slots_workaround():
    if False:
        return 10
    '\n    `__init_subclass__` works with modern APIs if care is taken around classes\n    existing twice.\n    '
    subs = {}

    @attr.define
    class Base:

        def __init_subclass__(cls):
            if False:
                for i in range(10):
                    print('nop')
            subs[cls.__qualname__] = cls

    @attr.define
    class Sub1(Base):
        x: int

    @attr.define
    class Sub2(Base):
        y: int
    assert (Sub1, Sub2) == tuple(subs.values())