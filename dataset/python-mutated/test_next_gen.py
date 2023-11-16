"""
Integration tests for next-generation APIs.
"""
import re
from contextlib import contextmanager
from functools import partial
import pytest
import attr as _attr
import attrs

@attrs.define
class C:
    x: str
    y: int

class TestNextGen:

    def test_simple(self):
        if False:
            print('Hello World!')
        '\n        Instantiation works.\n        '
        C('1', 2)

    def test_field_type(self):
        if False:
            return 10
        '\n        Make class with attrs.field and type parameter.\n        '
        classFields = {'testint': attrs.field(type=int)}
        A = attrs.make_class('A', classFields)
        assert int == attrs.fields(A).testint.type

    def test_no_slots(self):
        if False:
            print('Hello World!')
        '\n        slots can be deactivated.\n        '

        @attrs.define(slots=False)
        class NoSlots:
            x: int
        ns = NoSlots(1)
        assert {'x': 1} == ns.__dict__

    def test_validates(self):
        if False:
            return 10
        '\n        Validators at __init__ and __setattr__ work.\n        '

        @attrs.define
        class Validated:
            x: int = attrs.field(validator=attrs.validators.instance_of(int))
        v = Validated(1)
        with pytest.raises(TypeError):
            Validated(None)
        with pytest.raises(TypeError):
            v.x = '1'

    def test_no_order(self):
        if False:
            while True:
                i = 10
        '\n        Order is off by default but can be added.\n        '
        with pytest.raises(TypeError):
            C('1', 2) < C('2', 3)

        @attrs.define(order=True)
        class Ordered:
            x: int
        assert Ordered(1) < Ordered(2)

    def test_override_auto_attribs_true(self):
        if False:
            print('Hello World!')
        "\n        Don't guess if auto_attrib is set explicitly.\n\n        Having an unannotated attrs.ib/attrs.field fails.\n        "
        with pytest.raises(attrs.exceptions.UnannotatedAttributeError):

            @attrs.define(auto_attribs=True)
            class ThisFails:
                x = attrs.field()
                y: int

    def test_override_auto_attribs_false(self):
        if False:
            return 10
        "\n        Don't guess if auto_attrib is set explicitly.\n\n        Annotated fields that don't carry an attrs.ib are ignored.\n        "

        @attrs.define(auto_attribs=False)
        class NoFields:
            x: int
            y: int
        assert NoFields() == NoFields()

    def test_auto_attribs_detect(self):
        if False:
            return 10
        '\n        define correctly detects if a class lacks type annotations.\n        '

        @attrs.define
        class OldSchool:
            x = attrs.field()
        assert OldSchool(1) == OldSchool(1)

        @attrs.define()
        class OldSchool2:
            x = attrs.field()
        assert OldSchool2(1) == OldSchool2(1)

    def test_auto_attribs_detect_fields_and_annotations(self):
        if False:
            while True:
                i = 10
        '\n        define infers auto_attribs=True if fields have type annotations\n        '

        @attrs.define
        class NewSchool:
            x: int
            y: list = attrs.field()

            @y.validator
            def _validate_y(self, attribute, value):
                if False:
                    i = 10
                    return i + 15
                if value < 0:
                    raise ValueError('y must be positive')
        assert NewSchool(1, 1) == NewSchool(1, 1)
        with pytest.raises(ValueError):
            NewSchool(1, -1)
        assert list(attrs.fields_dict(NewSchool).keys()) == ['x', 'y']

    def test_auto_attribs_partially_annotated(self):
        if False:
            print('Hello World!')
        '\n        define infers auto_attribs=True if any type annotations are found\n        '

        @attrs.define
        class NewSchool:
            x: int
            y: list
            z = 10
        assert NewSchool(1, []) == NewSchool(1, [])
        assert list(attrs.fields_dict(NewSchool).keys()) == ['x', 'y']
        assert NewSchool.z == 10
        assert 'z' in NewSchool.__dict__

    def test_auto_attribs_detect_annotations(self):
        if False:
            i = 10
            return i + 15
        '\n        define correctly detects if a class has type annotations.\n        '

        @attrs.define
        class NewSchool:
            x: int
        assert NewSchool(1) == NewSchool(1)

        @attrs.define()
        class NewSchool2:
            x: int
        assert NewSchool2(1) == NewSchool2(1)

    def test_exception(self):
        if False:
            i = 10
            return i + 15
        '\n        Exceptions are detected and correctly handled.\n        '

        @attrs.define
        class E(Exception):
            msg: str
            other: int
        with pytest.raises(E) as ei:
            raise E('yolo', 42)
        e = ei.value
        assert ('yolo', 42) == e.args
        assert 'yolo' == e.msg
        assert 42 == e.other

    def test_frozen(self):
        if False:
            i = 10
            return i + 15
        '\n        attrs.frozen freezes classes.\n        '

        @attrs.frozen
        class F:
            x: str
        f = F(1)
        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            f.x = 2

    def test_auto_detect_eq(self):
        if False:
            return 10
        '\n        auto_detect=True works for eq.\n\n        Regression test for #670.\n        '

        @attrs.define
        class C:

            def __eq__(self, o):
                if False:
                    for i in range(10):
                        print('nop')
                raise ValueError()
        with pytest.raises(ValueError):
            C() == C()

    def test_subclass_frozen(self):
        if False:
            while True:
                i = 10
        "\n        It's possible to subclass an `attrs.frozen` class and the frozen-ness\n        is inherited.\n        "

        @attrs.frozen
        class A:
            a: int

        @attrs.frozen
        class B(A):
            b: int

        @attrs.define(on_setattr=attrs.setters.NO_OP)
        class C(B):
            c: int
        assert B(1, 2) == B(1, 2)
        assert C(1, 2, 3) == C(1, 2, 3)
        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            A(1).a = 1
        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            B(1, 2).a = 1
        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            B(1, 2).b = 2
        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            C(1, 2, 3).c = 3

    def test_catches_frozen_on_setattr(self):
        if False:
            return 10
        '\n        Passing frozen=True and on_setattr hooks is caught, even if the\n        immutability is inherited.\n        '

        @attrs.define(frozen=True)
        class A:
            pass
        with pytest.raises(ValueError, match="Frozen classes can't use on_setattr."):

            @attrs.define(frozen=True, on_setattr=attrs.setters.validate)
            class B:
                pass
        with pytest.raises(ValueError, match=re.escape("Frozen classes can't use on_setattr (frozen-ness was inherited).")):

            @attrs.define(on_setattr=attrs.setters.validate)
            class C(A):
                pass

    @pytest.mark.parametrize('decorator', [partial(_attr.s, frozen=True, slots=True, auto_exc=True), attrs.frozen, attrs.define, attrs.mutable])
    def test_discard_context(self, decorator):
        if False:
            return 10
        '\n        raise from None works.\n\n        Regression test for #703.\n        '

        @decorator
        class MyException(Exception):
            x: str = attrs.field()
        with pytest.raises(MyException) as ei:
            try:
                raise ValueError()
            except ValueError:
                raise MyException('foo') from None
        assert 'foo' == ei.value.x
        assert ei.value.__cause__ is None

    @pytest.mark.parametrize('decorator', [partial(_attr.s, frozen=True, slots=True, auto_exc=True), attrs.frozen, attrs.define, attrs.mutable])
    def test_setting_traceback_on_exception(self, decorator):
        if False:
            return 10
        '\n        contextlib.contextlib (re-)sets __traceback__ on raised exceptions.\n\n        Ensure that works, as well as if done explicitly\n        '

        @decorator
        class MyException(Exception):
            pass

        @contextmanager
        def do_nothing():
            if False:
                return 10
            yield
        with do_nothing(), pytest.raises(MyException) as ei:
            raise MyException()
        assert isinstance(ei.value, MyException)
        ei.value.__traceback__ = ei.value.__traceback__

    def test_converts_and_validates_by_default(self):
        if False:
            print('Hello World!')
        '\n        If no on_setattr is set, assume setters.convert, setters.validate.\n        '

        @attrs.define
        class C:
            x: int = attrs.field(converter=int)

            @x.validator
            def _v(self, _, value):
                if False:
                    for i in range(10):
                        print('nop')
                if value < 10:
                    raise ValueError('must be >=10')
        inst = C(10)
        inst.x = '11'
        assert 11 == inst.x
        with pytest.raises(ValueError, match='must be >=10'):
            inst.x = '9'

    def test_mro_ng(self):
        if False:
            print('Hello World!')
        '\n        Attributes and methods are looked up the same way in NG by default.\n\n        See #428\n        '

        @attrs.define
        class A:
            x: int = 10

            def xx(self):
                if False:
                    print('Hello World!')
                return 10

        @attrs.define
        class B(A):
            y: int = 20

        @attrs.define
        class C(A):
            x: int = 50

            def xx(self):
                if False:
                    i = 10
                    return i + 15
                return 50

        @attrs.define
        class D(B, C):
            pass
        d = D()
        assert d.x == d.xx()

class TestAsTuple:

    def test_smoke(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        `attrs.astuple` only changes defaults, so we just call it and compare.\n        '
        inst = C('foo', 42)
        assert attrs.astuple(inst) == _attr.astuple(inst)

class TestAsDict:

    def test_smoke(self):
        if False:
            i = 10
            return i + 15
        '\n        `attrs.asdict` only changes defaults, so we just call it and compare.\n        '
        inst = C('foo', {(1,): 42})
        assert attrs.asdict(inst) == _attr.asdict(inst, retain_collection_types=True)

class TestImports:
    """
    Verify our re-imports and mirroring works.
    """

    def test_converters(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Importing from attrs.converters works.\n        '
        from attrs.converters import optional
        assert optional is _attr.converters.optional

    def test_exceptions(self):
        if False:
            while True:
                i = 10
        '\n        Importing from attrs.exceptions works.\n        '
        from attrs.exceptions import FrozenError
        assert FrozenError is _attr.exceptions.FrozenError

    def test_filters(self):
        if False:
            while True:
                i = 10
        '\n        Importing from attrs.filters works.\n        '
        from attrs.filters import include
        assert include is _attr.filters.include

    def test_setters(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Importing from attrs.setters works.\n        '
        from attrs.setters import pipe
        assert pipe is _attr.setters.pipe

    def test_validators(self):
        if False:
            print('Hello World!')
        '\n        Importing from attrs.validators works.\n        '
        from attrs.validators import and_
        assert and_ is _attr.validators.and_