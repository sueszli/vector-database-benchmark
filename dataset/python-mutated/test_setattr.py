import pickle
import pytest
import attr
from attr import setters
from attr.exceptions import FrozenAttributeError
from attr.validators import instance_of, matches_re

@attr.s(frozen=True)
class Frozen:
    x = attr.ib()

@attr.s
class WithOnSetAttrHook:
    x = attr.ib(on_setattr=lambda *args: None)

class TestSetAttr:

    def test_change(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The return value of a hook overwrites the value. But they are not run\n        on __init__.\n        '

        def hook(*a, **kw):
            if False:
                return 10
            return 'hooked!'

        @attr.s
        class Hooked:
            x = attr.ib(on_setattr=hook)
            y = attr.ib()
        h = Hooked('x', 'y')
        assert 'x' == h.x
        assert 'y' == h.y
        h.x = 'xxx'
        h.y = 'yyy'
        assert 'yyy' == h.y
        assert 'hooked!' == h.x

    def test_frozen_attribute(self):
        if False:
            while True:
                i = 10
        '\n        Frozen attributes raise FrozenAttributeError, others are not affected.\n        '

        @attr.s
        class PartiallyFrozen:
            x = attr.ib(on_setattr=setters.frozen)
            y = attr.ib()
        pf = PartiallyFrozen('x', 'y')
        pf.y = 'yyy'
        assert 'yyy' == pf.y
        with pytest.raises(FrozenAttributeError):
            pf.x = 'xxx'
        assert 'x' == pf.x

    @pytest.mark.parametrize('on_setattr', [setters.validate, [setters.validate], setters.pipe(setters.validate)])
    def test_validator(self, on_setattr):
        if False:
            return 10
        "\n        Validators are run and they don't alter the value.\n        "

        @attr.s(on_setattr=on_setattr)
        class ValidatedAttribute:
            x = attr.ib()
            y = attr.ib(validator=[instance_of(str), matches_re('foo.*qux')])
        va = ValidatedAttribute(42, 'foobarqux')
        with pytest.raises(TypeError) as ei:
            va.y = 42
        assert 'foobarqux' == va.y
        assert ei.value.args[0].startswith("'y' must be <")
        with pytest.raises(ValueError) as ei:
            va.y = 'quxbarfoo'
        assert ei.value.args[0].startswith("'y' must match regex '")
        assert 'foobarqux' == va.y
        va.y = 'foobazqux'
        assert 'foobazqux' == va.y

    def test_pipe(self):
        if False:
            return 10
        '\n        Multiple hooks are possible, in that case the last return value is\n        used. They can be supplied using the pipe functions or by passing a\n        list to on_setattr.\n        '
        s = [setters.convert, lambda _, __, nv: nv + 1]

        @attr.s
        class Piped:
            x1 = attr.ib(converter=int, on_setattr=setters.pipe(*s))
            x2 = attr.ib(converter=int, on_setattr=s)
        p = Piped('41', '22')
        assert 41 == p.x1
        assert 22 == p.x2
        p.x1 = '41'
        p.x2 = '22'
        assert 42 == p.x1
        assert 23 == p.x2

    def test_make_class(self):
        if False:
            print('Hello World!')
        '\n        on_setattr of make_class gets forwarded.\n        '
        C = attr.make_class('C', {'x': attr.ib()}, on_setattr=setters.frozen)
        c = C(1)
        with pytest.raises(FrozenAttributeError):
            c.x = 2

    def test_no_validator_no_converter(self):
        if False:
            return 10
        '\n        validate and convert tolerate missing validators and converters.\n        '

        @attr.s(on_setattr=[setters.convert, setters.validate])
        class C:
            x = attr.ib()
        c = C(1)
        c.x = 2
        assert 2 == c.x

    def test_validate_respects_run_validators_config(self):
        if False:
            return 10
        "\n        If run validators is off, validate doesn't run them.\n        "

        @attr.s(on_setattr=setters.validate)
        class C:
            x = attr.ib(validator=attr.validators.instance_of(int))
        c = C(1)
        attr.set_run_validators(False)
        c.x = '1'
        assert '1' == c.x
        attr.set_run_validators(True)
        with pytest.raises(TypeError) as ei:
            c.x = '1'
        assert ei.value.args[0].startswith("'x' must be <")

    def test_frozen_on_setattr_class_is_caught(self):
        if False:
            while True:
                i = 10
        '\n        @attr.s(on_setattr=X, frozen=True) raises an ValueError.\n        '
        with pytest.raises(ValueError) as ei:

            @attr.s(frozen=True, on_setattr=setters.validate)
            class C:
                x = attr.ib()
        assert "Frozen classes can't use on_setattr." == ei.value.args[0]

    def test_frozen_on_setattr_attribute_is_caught(self):
        if False:
            i = 10
            return i + 15
        '\n        attr.ib(on_setattr=X) on a frozen class raises an ValueError.\n        '
        with pytest.raises(ValueError) as ei:

            @attr.s(frozen=True)
            class C:
                x = attr.ib(on_setattr=setters.validate)
        assert "Frozen classes can't use on_setattr." == ei.value.args[0]

    def test_setattr_reset_if_no_custom_setattr(self, slots):
        if False:
            print('Hello World!')
        '\n        If a class with an active setattr is subclassed and no new setattr\n        is generated, the __setattr__ is set to object.__setattr__.\n\n        We do the double test because of Python 2.\n        '

        def boom(*args):
            if False:
                print('Hello World!')
            pytest.fail('Must not be called.')

        @attr.s
        class Hooked:
            x = attr.ib(on_setattr=boom)

        @attr.s(slots=slots)
        class NoHook(WithOnSetAttrHook):
            x = attr.ib()
        assert NoHook.__setattr__ == object.__setattr__
        assert 1 == NoHook(1).x
        assert Hooked.__attrs_own_setattr__
        assert not NoHook.__attrs_own_setattr__
        assert WithOnSetAttrHook.__attrs_own_setattr__

    def test_setattr_inherited_do_not_reset(self, slots):
        if False:
            return 10
        '\n        If we inherit a __setattr__ that has been written by the user, we must\n        not reset it unless necessary.\n        '

        class A:
            """
            Not an attrs class on purpose to prevent accidental resets that
            would render the asserts meaningless.
            """

            def __setattr__(self, *args):
                if False:
                    return 10
                pass

        @attr.s(slots=slots)
        class B(A):
            pass
        assert B.__setattr__ == A.__setattr__

        @attr.s(slots=slots)
        class C(B):
            pass
        assert C.__setattr__ == A.__setattr__

    def test_pickling_retains_attrs_own(self, slots):
        if False:
            print('Hello World!')
        '\n        Pickling/Unpickling does not lose ownership information about\n        __setattr__.\n        '
        i = WithOnSetAttrHook(1)
        assert True is i.__attrs_own_setattr__
        i2 = pickle.loads(pickle.dumps(i))
        assert True is i2.__attrs_own_setattr__
        WOSAH = pickle.loads(pickle.dumps(WithOnSetAttrHook))
        assert True is WOSAH.__attrs_own_setattr__

    def test_slotted_class_can_have_custom_setattr(self):
        if False:
            while True:
                i = 10
        "\n        A slotted class can define a custom setattr and it doesn't get\n        overwritten.\n\n        Regression test for #680.\n        "

        @attr.s(slots=True)
        class A:

            def __setattr__(self, key, value):
                if False:
                    for i in range(10):
                        print('nop')
                raise SystemError
        with pytest.raises(SystemError):
            A().x = 1

    @pytest.mark.xfail(raises=attr.exceptions.FrozenAttributeError)
    def test_slotted_confused(self):
        if False:
            print('Hello World!')
        "\n        If we have a in-between non-attrs class, setattr reset detection\n        should still work, but currently doesn't.\n\n        It works with dict classes because we can look the finished class and\n        patch it.  With slotted classes we have to deduce it ourselves.\n        "

        @attr.s(slots=True)
        class A:
            x = attr.ib(on_setattr=setters.frozen)

        class B(A):
            pass

        @attr.s(slots=True)
        class C(B):
            x = attr.ib()
        C(1).x = 2

    def test_setattr_auto_detect_if_no_custom_setattr(self, slots):
        if False:
            print('Hello World!')
        "\n        It's possible to remove the on_setattr hook from an attribute and\n        therefore write a custom __setattr__.\n        "
        assert 1 == WithOnSetAttrHook(1).x

        @attr.s(auto_detect=True, slots=slots)
        class RemoveNeedForOurSetAttr(WithOnSetAttrHook):
            x = attr.ib()

            def __setattr__(self, name, val):
                if False:
                    i = 10
                    return i + 15
                object.__setattr__(self, name, val * 2)
        i = RemoveNeedForOurSetAttr(1)
        assert not RemoveNeedForOurSetAttr.__attrs_own_setattr__
        assert 2 == i.x

    def test_setattr_restore_respects_auto_detect(self, slots):
        if False:
            return 10
        '\n        If __setattr__ should be restored but the user supplied its own and\n        set auto_detect, leave is alone.\n        '

        @attr.s(auto_detect=True, slots=slots)
        class CustomSetAttr:

            def __setattr__(self, _, __):
                if False:
                    i = 10
                    return i + 15
                pass
        assert CustomSetAttr.__setattr__ != object.__setattr__

    def test_setattr_auto_detect_frozen(self, slots):
        if False:
            for i in range(10):
                print('nop')
        '\n        frozen=True together with a detected custom __setattr__ are rejected.\n        '
        with pytest.raises(ValueError, match="Can't freeze a class with a custom __setattr__."):

            @attr.s(auto_detect=True, slots=slots, frozen=True)
            class CustomSetAttr(Frozen):

                def __setattr__(self, _, __):
                    if False:
                        print('Hello World!')
                    pass

    def test_setattr_auto_detect_on_setattr(self, slots):
        if False:
            print('Hello World!')
        '\n        on_setattr attributes together with a detected custom __setattr__ are\n        rejected.\n        '
        with pytest.raises(ValueError, match="Can't combine custom __setattr__ with on_setattr hooks."):

            @attr.s(auto_detect=True, slots=slots)
            class HookAndCustomSetAttr:
                x = attr.ib(on_setattr=lambda *args: None)

                def __setattr__(self, _, __):
                    if False:
                        i = 10
                        return i + 15
                    pass

    @pytest.mark.parametrize('a_slots', [True, False])
    @pytest.mark.parametrize('b_slots', [True, False])
    @pytest.mark.parametrize('c_slots', [True, False])
    def test_setattr_inherited_do_not_reset_intermediate(self, a_slots, b_slots, c_slots):
        if False:
            while True:
                i = 10
        '\n        A user-provided intermediate __setattr__ is not reset to\n        object.__setattr__.\n\n        This only can work with auto_detect activated, such that attrs can know\n        that there is a user-provided __setattr__.\n        '

        @attr.s(slots=a_slots)
        class A:
            x = attr.ib(on_setattr=setters.frozen)

        @attr.s(slots=b_slots, auto_detect=True)
        class B(A):
            x = attr.ib(on_setattr=setters.NO_OP)

            def __setattr__(self, key, value):
                if False:
                    i = 10
                    return i + 15
                raise SystemError

        @attr.s(slots=c_slots)
        class C(B):
            pass
        assert getattr(A, '__attrs_own_setattr__', False) is True
        assert getattr(B, '__attrs_own_setattr__', False) is False
        assert getattr(C, '__attrs_own_setattr__', False) is False
        with pytest.raises(SystemError):
            C(1).x = 3

    def test_docstring(self):
        if False:
            return 10
        '\n        Generated __setattr__ has a useful docstring.\n        '
        assert 'Method generated by attrs for class WithOnSetAttrHook.' == WithOnSetAttrHook.__setattr__.__doc__