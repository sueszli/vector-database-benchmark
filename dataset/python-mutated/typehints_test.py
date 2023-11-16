"""Unit tests for the type-hint objects and decorators."""
import collections.abc
import functools
import sys
import typing
import unittest
from apache_beam import Map
from apache_beam import PTransform
from apache_beam.pvalue import PBegin
from apache_beam.pvalue import PCollection
from apache_beam.pvalue import PDone
from apache_beam.transforms.core import DoFn
from apache_beam.typehints import KV
from apache_beam.typehints import Any
from apache_beam.typehints import Iterable
from apache_beam.typehints import Tuple
from apache_beam.typehints import TypeCheckError
from apache_beam.typehints import Union
from apache_beam.typehints import native_type_compatibility
from apache_beam.typehints import typehints
from apache_beam.typehints import with_input_types
from apache_beam.typehints import with_output_types
from apache_beam.typehints.decorators import GeneratorWrapper
from apache_beam.typehints.decorators import _check_instance_type
from apache_beam.typehints.decorators import _interleave_type_check
from apache_beam.typehints.decorators import _positional_arg_hints
from apache_beam.typehints.decorators import get_signature
from apache_beam.typehints.decorators import get_type_hints
from apache_beam.typehints.decorators import getcallargs_forhints
from apache_beam.typehints.typehints import is_consistent_with
from apache_beam.typehints.typehints import visit_inner_types

def check_or_interleave(hint, value, var):
    if False:
        i = 10
        return i + 15
    if hint is None:
        return value
    elif isinstance(hint, typehints.IteratorHint.IteratorTypeConstraint):
        return _interleave_type_check(hint, var)(value)
    _check_instance_type(hint, value, var)
    return value

def check_type_hints(f):
    if False:
        print('Hello World!')

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if False:
            print('Hello World!')
        hints = get_type_hints(f)
        if hints.input_types:
            input_hints = getcallargs_forhints(f, *hints.input_types[0], **hints.input_types[1])
            inputs = get_signature(f).bind(*args, **kwargs).arguments
            for (var, hint) in input_hints.items():
                value = inputs[var]
                new_value = check_or_interleave(hint, value, var)
                if new_value is not value:
                    if var in kwargs:
                        kwargs[var] = new_value
                    else:
                        args = list(args)
                        for (ix, pvar) in enumerate(get_signature(f).parameters):
                            if pvar == var:
                                args[ix] = new_value
                                break
                        else:
                            raise NotImplementedError('Iterable in nested argument %s' % var)
        res = f(*args, **kwargs)
        return check_or_interleave(hints.simple_output_type('typecheck'), res, None)
    return wrapper

class DummyTestClass1(object):
    pass

class DummyTestClass2(object):
    pass

class SuperClass(object):
    pass

class SubClass(SuperClass):
    pass
T = typing.TypeVar('T')

class NonBuiltInGeneric(typing.NamedTuple('Entry', [('Field1', T), ('Field2', T)]), typing.Generic[T]):
    pass

class TypeHintTestCase(unittest.TestCase):

    def assertCompatible(self, base, sub):
        if False:
            return 10
        (base, sub) = native_type_compatibility.convert_to_beam_types([base, sub])
        self.assertTrue(is_consistent_with(sub, base), '%s is not consistent with %s' % (sub, base))

    def assertNotCompatible(self, base, sub):
        if False:
            i = 10
            return i + 15
        (base, sub) = native_type_compatibility.convert_to_beam_type([base, sub])
        self.assertFalse(is_consistent_with(sub, base), '%s is consistent with %s' % (sub, base))

class TypeVariableTestCase(TypeHintTestCase):

    def test_eq_with_name_check(self):
        if False:
            for i in range(10):
                print('nop')
        use_name_in_eq = True
        self.assertNotEqual(typehints.Set[typehints.TypeVariable('T', use_name_in_eq)], typehints.Set[typehints.TypeVariable('T2', use_name_in_eq)])

    def test_eq_without_name_check(self):
        if False:
            for i in range(10):
                print('nop')
        use_name_in_eq = False
        self.assertEqual(typehints.Set[typehints.TypeVariable('T', use_name_in_eq)], typehints.Set[typehints.TypeVariable('T2', use_name_in_eq)])

    def test_eq_with_different_name_checks(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(typehints.Set[typehints.TypeVariable('T', True)], typehints.Set[typehints.TypeVariable('T2', False)])
        self.assertEqual(typehints.Set[typehints.TypeVariable('T', False)], typehints.Set[typehints.TypeVariable('T2', True)])

class AnyTypeConstraintTestCase(TypeHintTestCase):

    def test_any_compatibility(self):
        if False:
            print('Hello World!')
        self.assertCompatible(typehints.Any, typehints.List[int])
        self.assertCompatible(typehints.Any, DummyTestClass1)
        self.assertCompatible(typehints.Union[int, bool], typehints.Any)
        self.assertCompatible(typehints.Optional[int], typehints.Any)
        self.assertCompatible(typehints.Tuple[int], typehints.Any)
        self.assertCompatible(typehints.KV[int, str], typehints.Any)
        self.assertCompatible(typehints.Dict[int, bool], typehints.Any)
        self.assertCompatible(typehints.Set[int], typehints.Any)
        self.assertCompatible(typehints.FrozenSet[int], typehints.Any)
        self.assertCompatible(typehints.Iterable[int], typehints.Any)
        self.assertCompatible(typehints.Iterator[int], typehints.Any)
        self.assertCompatible(typehints.Generator[int], typehints.Any)
        self.assertCompatible(object, typehints.Any)
        self.assertCompatible(typehints.Any, object)

    def test_repr(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual('Any', repr(typehints.Any))

    def test_type_check(self):
        if False:
            for i in range(10):
                print('nop')
        typehints.Any.type_check(1)
        typehints.Any.type_check([1, 2, 3])
        typehints.Any.type_check(DummyTestClass1())

class UnionHintTestCase(TypeHintTestCase):

    def test_getitem_must_be_valid_type_param_cant_be_object_instance(self):
        if False:
            return 10
        with self.assertRaises(TypeError) as e:
            typehints.Union[5]
        self.assertEqual('Cannot create Union without a sequence of types.', e.exception.args[0])

    def test_getitem_must_be_valid_type_param(self):
        if False:
            for i in range(10):
                print('nop')
        t = [2, 3]
        with self.assertRaises(TypeError) as e:
            typehints.Union[t]
        self.assertEqual('All parameters to a Union hint must be a non-sequence, a type, or a TypeConstraint. 2 is an instance of int.', e.exception.args[0])

    def test_getitem_duplicates_ignored(self):
        if False:
            while True:
                i = 10
        hint = typehints.Union[int, int, str]
        self.assertEqual(len(hint.union_types), 2)

    def test_getitem_nested_unions_flattened(self):
        if False:
            i = 10
            return i + 15
        hint = typehints.Union[typehints.Union[int, str], typehints.Union[float, bool]]
        self.assertTrue(len(hint.union_types) == 4)
        self.assertTrue(all((t in hint.union_types for t in (int, str, float, bool))))

    def test_union_hint_compatibility(self):
        if False:
            return 10
        self.assertCompatible(typehints.Union[int, float], int)
        self.assertCompatible(typehints.Union[int, str], typehints.Union[str, int])
        self.assertCompatible(typehints.Union[int, float, str], typehints.Union[str, int])
        self.assertCompatible(typehints.Union[DummyTestClass1, str], typehints.Union[DummyTestClass1, str])
        self.assertCompatible(typehints.Union[int, str], typehints.Union[str, typehints.Union[int, str]])
        self.assertNotCompatible(typehints.Union[float, bool], typehints.Union[int, bool])
        self.assertNotCompatible(typehints.Union[bool, str], typehints.Union[float, bool, int])

    def test_nested_compatibility(self):
        if False:
            i = 10
            return i + 15
        self.assertCompatible(Union[int, Tuple[Any, int]], Tuple[int, int])
        self.assertCompatible(Union[int, Tuple[Any, Any]], Union[Tuple[int, Any], Tuple[Any, int]])
        self.assertCompatible(Union[int, SuperClass], SubClass)
        self.assertCompatible(Union[int, float, SuperClass], Union[int, SubClass])
        self.assertCompatible(int, Union[()])
        self.assertNotCompatible(Union[int, SubClass], SuperClass)
        self.assertNotCompatible(Union[int, float, SubClass], Union[int, SuperClass])
        self.assertNotCompatible(Union[int, SuperClass], Union[int, float, SubClass])
        self.assertCompatible(Tuple[Any, Any], Union[Tuple[str, int], Tuple[str, float]])

    def test_union_hint_repr(self):
        if False:
            for i in range(10):
                print('nop')
        hint = typehints.Union[DummyTestClass1, str]
        self.assertIn(str(hint), ["Union[<class 'str'>, <class 'apache_beam.typehints.typehints_test.DummyTestClass1'>]", "Union[<class 'apache_beam.typehints.typehints_test.DummyTestClass1'>, <class 'str'>]"])

    def test_union_hint_enforcement_composite_type_in_union(self):
        if False:
            while True:
                i = 10
        o = DummyTestClass1()
        hint = typehints.Union[int, DummyTestClass1]
        self.assertIsNone(hint.type_check(4))
        self.assertIsNone(hint.type_check(o))

    def test_union_hint_enforcement_part_of_union(self):
        if False:
            i = 10
            return i + 15
        hint = typehints.Union[int, str]
        self.assertIsNone(hint.type_check(5))
        self.assertIsNone(hint.type_check('test'))

    def test_union_hint_enforcement_not_part_of_union(self):
        if False:
            while True:
                i = 10
        hint = typehints.Union[int, float]
        with self.assertRaises(TypeError) as e:
            hint.type_check('test')
        self.assertEqual('Union[<class \'float\'>, <class \'int\'>] type-constraint violated. Expected an instance of one of: ("<class \'float\'>", "<class \'int\'>"), received str instead.', e.exception.args[0])

    def test_dict_union(self):
        if False:
            return 10
        hint = Union[typehints.Dict[Any, int], typehints.Dict[Union[()], Union[()]]]
        self.assertEqual(typehints.Dict[Any, int], hint)

    def test_empty_union(self):
        if False:
            while True:
                i = 10
        self.assertEqual(typehints.Union[()], typehints.Union[typehints.Union[()], typehints.Union[()]])
        self.assertEqual(int, typehints.Union[typehints.Union[()], int])

    def test_match_type_variables(self):
        if False:
            i = 10
            return i + 15
        A = typehints.TypeVariable('A')
        B = typehints.TypeVariable('B')
        self.assertEqual(typehints.Union[A, int].match_type_variables(str), {A: str})
        self.assertEqual(typehints.Union[A, int].match_type_variables(int), {})
        self.assertEqual(typehints.Union[A, B, int].match_type_variables(str), {})
        self.assertEqual(typehints.Tuple[A, typehints.Union[A, B, int]].match_type_variables(typehints.Tuple[str, float]), {A: str})
        self.assertEqual(typehints.Union[Tuple[str, A], Tuple[float, B]].match_type_variables(typehints.Tuple[Any, int]), {})
        self.assertEqual(typehints.Union[Tuple[str, A], Tuple[float, A]].match_type_variables(typehints.Tuple[Any, int]), {A: int})
        self.assertEqual(typehints.Union[Tuple[str, A], Tuple[float, B]].match_type_variables(typehints.Tuple[str, int]), {A: int})

    def test_bind_type_variables(self):
        if False:
            print('Hello World!')
        A = typehints.TypeVariable('A')
        B = typehints.TypeVariable('B')
        hint = typehints.Union[A, B, int]
        self.assertEqual(hint.bind_type_variables({A: str, B: float}), typehints.Union[str, float, int])
        self.assertEqual(hint.bind_type_variables({A: str, B: int}), typehints.Union[str, int])
        self.assertEqual(hint.bind_type_variables({A: int, B: int}), int)

    def test_visit_inner_types(self):
        if False:
            while True:
                i = 10
        A = typehints.TypeVariable('A')
        B = typehints.TypeVariable('B')
        hint = typehints.Tuple[Tuple[A, A], B, int]
        user_data = object()
        nodes = []

        def visitor(hint, arg):
            if False:
                while True:
                    i = 10
            self.assertIs(arg, user_data)
            nodes.append(hint)
        visit_inner_types(hint, visitor, user_data)
        self.assertEqual(nodes, [hint, Tuple[A, A], A, A, B, int])

class OptionalHintTestCase(TypeHintTestCase):

    def test_getitem_sequence_not_allowed(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(TypeError) as e:
            typehints.Optional[int, str]
        self.assertTrue(e.exception.args[0].startswith('An Option type-hint only accepts a single type parameter.'))

    def test_getitem_proxy_to_union(self):
        if False:
            print('Hello World!')
        hint = typehints.Optional[int]
        self.assertTrue(isinstance(hint, typehints.UnionHint.UnionConstraint))

    def test_is_optional(self):
        if False:
            return 10
        hint1 = typehints.Optional[int]
        self.assertTrue(typehints.is_nullable(hint1))
        hint2 = typehints.UnionConstraint({int, bytes})
        self.assertFalse(typehints.is_nullable(hint2))
        hint3 = typehints.UnionConstraint({int, bytes, type(None)})
        self.assertFalse(typehints.is_nullable(hint3))

class TupleHintTestCase(TypeHintTestCase):

    def test_getitem_invalid_ellipsis_type_param(self):
        if False:
            print('Hello World!')
        error_msg = 'Ellipsis can only be used to type-hint an arbitrary length tuple of containing a single type: Tuple[A, ...].'
        with self.assertRaises(TypeError) as e:
            typehints.Tuple[int, int, ...]
        self.assertEqual(error_msg, e.exception.args[0])
        with self.assertRaises(TypeError) as e:
            typehints.Tuple[...]
        self.assertEqual(error_msg, e.exception.args[0])

    def test_getitem_params_must_be_type_or_constraint(self):
        if False:
            while True:
                i = 10
        expected_error_prefix = 'All parameters to a Tuple hint must be'
        with self.assertRaises(TypeError) as e:
            typehints.Tuple[5, [1, 3]]
        self.assertTrue(e.exception.args[0].startswith(expected_error_prefix))
        if sys.version_info < (3, 9):
            with self.assertRaises(TypeError) as e:
                typehints.Tuple[list, dict]
            self.assertTrue(e.exception.args[0].startswith(expected_error_prefix))
        else:
            try:
                typehints.Tuple[list, dict]
            except TypeError:
                self.fail('built-in composite raised TypeError unexpectedly')

    def test_compatibility_arbitrary_length(self):
        if False:
            print('Hello World!')
        self.assertNotCompatible(typehints.Tuple[int, int], typehints.Tuple[int, ...])
        self.assertCompatible(typehints.Tuple[int, ...], typehints.Tuple[int, int])
        self.assertCompatible(typehints.Tuple[Any, ...], typehints.Tuple[int, float])
        self.assertCompatible(typehints.Tuple[SuperClass, ...], typehints.Tuple[SubClass, SuperClass])
        self.assertCompatible(typehints.Iterable[int], typehints.Tuple[int, ...])
        self.assertCompatible(typehints.Iterable[SuperClass], typehints.Tuple[SubClass, ...])

    def test_compatibility(self):
        if False:
            print('Hello World!')
        self.assertCompatible(typehints.Tuple[int, str], typehints.Tuple[int, str])
        self.assertCompatible(typehints.Tuple[int, Any], typehints.Tuple[int, str])
        self.assertCompatible(typehints.Tuple[int, str], typehints.Tuple[int, Any])
        self.assertCompatible(typehints.Tuple[typehints.Union[int, str], bool], typehints.Tuple[typehints.Union[int, str], bool])
        self.assertCompatible(typehints.Tuple[typehints.Union[str, int], int], typehints.Tuple[typehints.Union[int, str], int])
        self.assertCompatible(typehints.Tuple[SuperClass, int], typehints.Tuple[SubClass, int])
        self.assertNotCompatible(typehints.Tuple[int, int], typehints.Tuple[int, int, int])

    def test_raw_tuple(self):
        if False:
            return 10
        self.assertCompatible(tuple, typehints.Tuple[int])
        self.assertCompatible(tuple, typehints.Tuple[int, float])
        self.assertCompatible(tuple, typehints.Tuple[int, ...])

    def test_repr(self):
        if False:
            return 10
        hint = typehints.Tuple[int, str, float]
        self.assertEqual("Tuple[<class 'int'>, <class 'str'>, <class 'float'>]", str(hint))
        hint = typehints.Tuple[DummyTestClass1, DummyTestClass2]
        self.assertEqual("Tuple[<class 'apache_beam.typehints.typehints_test.DummyTestClass1'>, <class 'apache_beam.typehints.typehints_test.DummyTestClass2'>]", str(hint))
        hint = typehints.Tuple[float, ...]
        self.assertEqual("Tuple[<class 'float'>, ...]", str(hint))

    def test_type_check_must_be_tuple(self):
        if False:
            print('Hello World!')
        hint = typehints.Tuple[int, str]
        expected_error_prefix = 'Tuple type constraint violated. Valid object'
        invalid_instances = ([1, 2, 3], {4: 'f'}, 9, 'test', None)
        for t in invalid_instances:
            with self.assertRaises(TypeError) as e:
                hint.type_check(t)
            self.assertTrue(e.exception.args[0].startswith(expected_error_prefix))

    def test_type_check_must_have_same_arity(self):
        if False:
            return 10
        hint = typehints.Tuple[int, int]
        t = (1, 2, 3)
        with self.assertRaises(TypeError) as e:
            hint.type_check(t)
        self.assertEqual('Passed object instance is of the proper type, but differs in length from the hinted type. Expected a tuple of length 2, received a tuple of length 3.', e.exception.args[0])

    def test_type_check_invalid_simple_types(self):
        if False:
            return 10
        hint = typehints.Tuple[str, bool]
        with self.assertRaises(TypeError) as e:
            hint.type_check((4, False))
        self.assertEqual("Tuple[<class 'str'>, <class 'bool'>] hint type-constraint violated. The type of element #0 in the passed tuple is incorrect. Expected an instance of type <class 'str'>, instead received an instance of type int.", e.exception.args[0])

    def test_type_check_invalid_composite_type(self):
        if False:
            return 10
        hint = typehints.Tuple[DummyTestClass1, DummyTestClass2]
        t = (DummyTestClass2(), DummyTestClass1())
        with self.assertRaises(TypeError) as e:
            hint.type_check(t)
        self.assertEqual("Tuple[<class 'apache_beam.typehints.typehints_test.DummyTestClass1'>, <class 'apache_beam.typehints.typehints_test.DummyTestClass2'>] hint type-constraint violated. The type of element #0 in the passed tuple is incorrect. Expected an instance of type <class 'apache_beam.typehints.typehints_test.DummyTestClass1'>, instead received an instance of type DummyTestClass2.", e.exception.args[0])

    def test_type_check_valid_simple_types(self):
        if False:
            for i in range(10):
                print('nop')
        hint = typehints.Tuple[float, bool]
        self.assertIsNone(hint.type_check((4.3, True)))
        hint = typehints.Tuple[int]
        self.assertIsNone(hint.type_check((1,)))

    def test_type_check_valid_composite_types(self):
        if False:
            return 10
        hint = typehints.Tuple[typehints.Tuple[int, str], typehints.Tuple[int, bool]]
        self.assertIsNone(hint.type_check(((4, 'test'), (4, True))))

    def test_type_check_valid_simple_type_arbitrary_length(self):
        if False:
            while True:
                i = 10
        hint = typehints.Tuple[int, ...]
        t = (1, 2, 3, 4)
        self.assertIsNone(hint.type_check(t))

    def test_type_check_valid_composite_type_arbitrary_length(self):
        if False:
            for i in range(10):
                print('nop')
        hint = typehints.Tuple[typehints.List[str], ...]
        t = (['h', 'e'], ['l', 'l'], ['o'])
        self.assertIsNone(hint.type_check(t))

    def test_type_check_invalid_simple_type_arbitrary_length(self):
        if False:
            for i in range(10):
                print('nop')
        hint = typehints.Tuple[str, ...]
        t = ('t', 'e', 5, 't')
        with self.assertRaises(TypeError) as e:
            hint.type_check(t)
        self.assertEqual("Tuple[<class 'str'>, ...] hint type-constraint violated. The type of element #2 in the passed <class 'tuple'> is incorrect. Expected an instance of type <class 'str'>, instead received an instance of type int.", e.exception.args[0])

    def test_type_check_invalid_composite_type_arbitrary_length(self):
        if False:
            return 10
        hint = typehints.Tuple[typehints.List[int], ...]
        t = ([1, 2], 'e', 's', 't')
        with self.assertRaises(TypeError) as e:
            hint.type_check(t)
        self.assertEqual("Tuple[List[<class 'int'>], ...] hint type-constraint violated. The type of element #1 in the passed tuple is incorrect: List type-constraint violated. Valid object instance must be of type 'list'. Instead, an instance of 'str' was received.", e.exception.args[0])

    def test_normalize_with_builtin_tuple(self):
        if False:
            print('Hello World!')
        if sys.version_info >= (3, 9):
            expected_beam_type = typehints.Tuple[int, int]
            converted_beam_type = typehints.normalize(tuple[int, int], False)
            self.assertEqual(converted_beam_type, expected_beam_type)

    def test_builtin_and_type_compatibility(self):
        if False:
            while True:
                i = 10
        if sys.version_info >= (3, 9):
            self.assertCompatible(tuple, typing.Tuple)
            self.assertCompatible(tuple[int, int], typing.Tuple[int, int])

class ListHintTestCase(TypeHintTestCase):

    def test_getitem_invalid_composite_type_param(self):
        if False:
            print('Hello World!')
        with self.assertRaises(TypeError):
            typehints.List[4]

    def test_list_constraint_compatibility(self):
        if False:
            print('Hello World!')
        hint1 = typehints.List[typehints.Tuple[int, str]]
        hint2 = typehints.List[typehints.Tuple[float, bool]]
        self.assertCompatible(hint1, hint1)
        self.assertNotCompatible(hint1, hint2)
        self.assertCompatible(typehints.List[SuperClass], typehints.List[SubClass])

    def test_list_repr(self):
        if False:
            return 10
        hint = typehints.List[typehints.Tuple[DummyTestClass1, DummyTestClass2]]
        self.assertEqual("List[Tuple[<class 'apache_beam.typehints.typehints_test.DummyTestClass1'>, <class 'apache_beam.typehints.typehints_test.DummyTestClass2'>]]", repr(hint))

    def test_enforce_list_type_constraint_valid_simple_type(self):
        if False:
            return 10
        hint = typehints.List[int]
        self.assertIsNone(hint.type_check([1, 2, 3]))

    def test_enforce_list_type_constraint_valid_composite_type(self):
        if False:
            for i in range(10):
                print('nop')
        hint = typehints.List[DummyTestClass1]
        l = [DummyTestClass1(), DummyTestClass1()]
        self.assertIsNone(hint.type_check(l))

    def test_enforce_list_type_constraint_invalid_simple_type(self):
        if False:
            print('Hello World!')
        hint = typehints.List[int]
        l = ['f', 'd', 'm']
        with self.assertRaises(TypeError) as e:
            hint.type_check(l)
        self.assertEqual("List[<class 'int'>] hint type-constraint violated. The type of element #0 in the passed <class 'list'> is incorrect. Expected an instance of type <class 'int'>, instead received an instance of type str.", e.exception.args[0])

    def test_enforce_list_type_constraint_invalid_composite_type(self):
        if False:
            i = 10
            return i + 15
        hint = typehints.List[typehints.Tuple[int, int]]
        l = [('f', 1), ('m', 5)]
        with self.assertRaises(TypeError) as e:
            hint.type_check(l)
        self.assertEqual("List[Tuple[<class 'int'>, <class 'int'>]] hint type-constraint violated. The type of element #0 in the passed list is incorrect: Tuple[<class 'int'>, <class 'int'>] hint type-constraint violated. The type of element #0 in the passed tuple is incorrect. Expected an instance of type <class 'int'>, instead received an instance of type str.", e.exception.args[0])

    def test_normalize_with_builtin_list(self):
        if False:
            return 10
        if sys.version_info >= (3, 9):
            expected_beam_type = typehints.List[int]
            converted_beam_type = typehints.normalize(list[int], False)
            self.assertEqual(converted_beam_type, expected_beam_type)

    def test_builtin_and_type_compatibility(self):
        if False:
            while True:
                i = 10
        if sys.version_info >= (3, 9):
            self.assertCompatible(list, typing.List)
            self.assertCompatible(list[int], typing.List[int])

    def test_is_typing_generic(self):
        if False:
            return 10
        self.assertTrue(typehints.is_typing_generic(typing.List[str]))

    def test_builtin_is_typing_generic(self):
        if False:
            while True:
                i = 10
        if sys.version_info >= (3, 9):
            self.assertTrue(typehints.is_typing_generic(list[str]))

class KVHintTestCase(TypeHintTestCase):

    def test_getitem_param_must_be_tuple(self):
        if False:
            return 10
        with self.assertRaises(TypeError) as e:
            typehints.KV[4]
        self.assertEqual('Parameter to KV type-hint must be a tuple of types: KV[.., ..].', e.exception.args[0])

    def test_getitem_param_must_have_length_2(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(TypeError) as e:
            typehints.KV[int, str, bool]
        self.assertEqual('Length of parameters to a KV type-hint must be exactly 2. Passed parameters: ({}, {}, {}), have a length of 3.'.format(int, str, bool), e.exception.args[0])

    def test_getitem_proxy_to_tuple(self):
        if False:
            print('Hello World!')
        hint = typehints.KV[int, str]
        self.assertTrue(isinstance(hint, typehints.Tuple.TupleConstraint))

    def test_enforce_kv_type_constraint(self):
        if False:
            i = 10
            return i + 15
        hint = typehints.KV[str, typehints.Tuple[int, int, int]]
        t = ('test', (1, 2, 3))
        self.assertIsNone(hint.type_check(t))

class DictHintTestCase(TypeHintTestCase):

    def test_getitem_param_must_be_tuple(self):
        if False:
            return 10
        with self.assertRaises(TypeError) as e:
            typehints.Dict[4]
        self.assertEqual('Parameter to Dict type-hint must be a tuple of types: Dict[.., ..].', e.exception.args[0])

    def test_getitem_param_must_have_length_2(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(TypeError) as e:
            typehints.Dict[float, int, bool]
        self.assertEqual('Length of parameters to a Dict type-hint must be exactly 2. Passed parameters: ({}, {}, {}), have a length of 3.'.format(float, int, bool), e.exception.args[0])

    def test_key_type_must_be_valid_composite_param(self):
        if False:
            while True:
                i = 10
        if sys.version_info < (3, 9):
            with self.assertRaises(TypeError):
                typehints.Dict[list, int]
        else:
            try:
                typehints.Tuple[list, int]
            except TypeError:
                self.fail('built-in composite raised TypeError unexpectedly')

    def test_value_type_must_be_valid_composite_param(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(TypeError):
            typehints.Dict[str, 5]

    def test_compatibility(self):
        if False:
            for i in range(10):
                print('nop')
        hint1 = typehints.Dict[int, str]
        hint2 = typehints.Dict[bool, int]
        hint3 = typehints.Dict[int, typehints.List[typehints.Tuple[str, str, str]]]
        hint4 = typehints.Dict[int, int]
        self.assertCompatible(hint1, hint1)
        self.assertCompatible(hint3, hint3)
        self.assertNotCompatible(hint3, 4)
        self.assertNotCompatible(hint2, hint1)
        self.assertNotCompatible(hint1, hint4)

    def test_repr(self):
        if False:
            print('Hello World!')
        hint3 = typehints.Dict[int, typehints.List[typehints.Tuple[str, str, str]]]
        self.assertEqual("Dict[<class 'int'>, List[Tuple[<class 'str'>, <class 'str'>, <class 'str'>]]]", repr(hint3))

    def test_type_checks_not_dict(self):
        if False:
            return 10
        hint = typehints.Dict[int, str]
        l = [1, 2]
        with self.assertRaises(TypeError) as e:
            hint.type_check(l)
        self.assertEqual('Dict type-constraint violated. All passed instances must be of type dict. [1, 2] is of type list.', e.exception.args[0])

    def test_type_check_invalid_key_type(self):
        if False:
            return 10
        hint = typehints.Dict[typehints.Tuple[int, int, int], typehints.List[str]]
        d = {(1, 2): ['m', '1', '2', '3']}
        with self.assertRaises((TypeError, TypeError)) as e:
            hint.type_check(d)
        self.assertEqual("Dict[Tuple[<class 'int'>, <class 'int'>, <class 'int'>], List[<class 'str'>]] hint key-type constraint violated. All keys should be of type Tuple[<class 'int'>, <class 'int'>, <class 'int'>]. Instead: Passed object instance is of the proper type, but differs in length from the hinted type. Expected a tuple of length 3, received a tuple of length 2.", e.exception.args[0])

    def test_type_check_invalid_value_type(self):
        if False:
            return 10
        hint = typehints.Dict[str, typehints.Dict[int, str]]
        d = {'f': [1, 2, 3]}
        with self.assertRaises(TypeError) as e:
            hint.type_check(d)
        self.assertEqual("Dict[<class 'str'>, Dict[<class 'int'>, <class 'str'>]] hint value-type constraint violated. All values should be of type Dict[<class 'int'>, <class 'str'>]. Instead: Dict type-constraint violated. All passed instances must be of type dict. [1, 2, 3] is of type list.", e.exception.args[0])

    def test_type_check_valid_simple_type(self):
        if False:
            while True:
                i = 10
        hint = typehints.Dict[int, str]
        d = {4: 'f', 9: 'k'}
        self.assertIsNone(hint.type_check(d))

    def test_type_check_valid_composite_type(self):
        if False:
            i = 10
            return i + 15
        hint = typehints.Dict[typehints.Tuple[str, str], typehints.List[int]]
        d = {('f', 'k'): [1, 2, 3], ('m', 'r'): [4, 6, 9]}
        self.assertIsNone(hint.type_check(d))

    def test_match_type_variables(self):
        if False:
            return 10
        S = typehints.TypeVariable('S')
        T = typehints.TypeVariable('T')
        hint = typehints.Dict[S, T]
        self.assertEqual({S: int, T: str}, hint.match_type_variables(typehints.Dict[int, str]))

    def test_normalize_with_builtin_dict(self):
        if False:
            i = 10
            return i + 15
        if sys.version_info >= (3, 9):
            expected_beam_type = typehints.Dict[str, int]
            converted_beam_type = typehints.normalize(dict[str, int], False)
            self.assertEqual(converted_beam_type, expected_beam_type)

    def test_builtin_and_type_compatibility(self):
        if False:
            while True:
                i = 10
        if sys.version_info >= (3, 9):
            self.assertCompatible(dict, typing.Dict)
            self.assertCompatible(dict[str, int], typing.Dict[str, int])
            self.assertCompatible(dict[str, list[int]], typing.Dict[str, typing.List[int]])

class BaseSetHintTest:

    class CommonTests(TypeHintTestCase):

        def test_getitem_invalid_composite_type_param(self):
            if False:
                for i in range(10):
                    print('nop')
            if sys.version_info < (3, 9):
                with self.assertRaises(TypeError) as e:
                    self.beam_type[list]
                self.assertEqual('Parameter to a {} hint must be a non-sequence, a type, or a TypeConstraint. {} is an instance of type.'.format(self.string_type, list), e.exception.args[0])
            else:
                try:
                    self.beam_type[list]
                except TypeError:
                    self.fail('built-in composite raised TypeError unexpectedly')

        def test_non_typing_generic(self):
            if False:
                print('Hello World!')
            testCase = DummyTestClass1()
            self.assertFalse(typehints.is_typing_generic(testCase))

        def test_compatibility(self):
            if False:
                i = 10
                return i + 15
            hint1 = self.beam_type[typehints.List[str]]
            hint2 = self.beam_type[typehints.Tuple[int, int]]
            self.assertCompatible(hint1, hint1)
            self.assertNotCompatible(hint2, hint1)

        def test_repr(self):
            if False:
                print('Hello World!')
            hint = self.beam_type[typehints.List[bool]]
            self.assertEqual("{}[List[<class 'bool'>]]".format(self.string_type), repr(hint))

        def test_type_check_must_be_set(self):
            if False:
                i = 10
                return i + 15
            hint = self.beam_type[str]
            with self.assertRaises(TypeError) as e:
                hint.type_check(4)
                self.assertEqual("{} type-constraint violated. Valid object instance must be of type '{}'. Instead, an instance of 'int' was received.".format(self.py_type, self.py_type.__name__), e.exception.args[0])

        def test_type_check_invalid_elem_type(self):
            if False:
                for i in range(10):
                    print('nop')
            hint = self.beam_type[float]
            with self.assertRaises(TypeError):
                hint.type_check('f')

        def test_type_check_valid_elem_simple_type(self):
            if False:
                return 10
            hint = self.beam_type[str]
            s = self.py_type(['f', 'm', 'k'])
            self.assertIsNone(hint.type_check(s))

        def test_type_check_valid_elem_composite_type(self):
            if False:
                return 10
            hint = self.beam_type[typehints.Union[int, str]]
            s = self.py_type([9, 'm', 'k'])
            self.assertIsNone(hint.type_check(s))

class SetHintTestCase(BaseSetHintTest.CommonTests):
    py_type = set
    beam_type = typehints.Set
    string_type = 'Set'

    def test_builtin_compatibility(self):
        if False:
            i = 10
            return i + 15
        if sys.version_info >= (3, 9):
            self.assertCompatible(set[int], collections.abc.Set[int])
            self.assertCompatible(set[int], collections.abc.MutableSet[int])

    def test_collections_compatibility(self):
        if False:
            print('Hello World!')
        if sys.version_info >= (3, 9):
            self.assertCompatible(collections.abc.Set[int], collections.abc.MutableSet[int])
            self.assertCompatible(collections.abc.MutableSet[int], collections.abc.Set[int])

class FrozenSetHintTestCase(BaseSetHintTest.CommonTests):
    py_type = frozenset
    beam_type = typehints.FrozenSet
    string_type = 'FrozenSet'

class CollectionHintTestCase(TypeHintTestCase):

    def test_type_constraint_compatibility(self):
        if False:
            return 10
        self.assertCompatible(typehints.Collection[int], typehints.Set[int])
        self.assertCompatible(typehints.Iterable[int], typehints.Collection[int])
        self.assertCompatible(typehints.Collection[int], typehints.FrozenSet[int])
        self.assertCompatible(typehints.Collection[typehints.Any], typehints.Collection[int])
        self.assertCompatible(typehints.Collection[int], typehints.Tuple[int])
        self.assertCompatible(typehints.Any, typehints.Collection[str])
        self.assertCompatible(typehints.Collection[str], typehints.List[str])

    def test_one_way_compatibility(self):
        if False:
            while True:
                i = 10
        self.assertNotCompatible(typehints.Set[int], typehints.Collection[int])
        self.assertNotCompatible(typehints.FrozenSet[int], typehints.Collection[int])
        self.assertNotCompatible(typehints.Tuple[int], typehints.Collection[int])
        self.assertNotCompatible(typehints.Collection[int], typehints.Iterable[int])
        self.assertNotCompatible(typehints.List[str], typehints.Collection[str])

    def test_getitem_invalid_composite_type_param(self):
        if False:
            return 10
        with self.assertRaises(TypeError) as e:
            typehints.Collection[5]
        self.assertEqual('Parameter to a Collection hint must be a non-sequence, a type, or a TypeConstraint. 5 is an instance of int.', e.exception.args[0])

class IterableHintTestCase(TypeHintTestCase):

    def test_getitem_invalid_composite_type_param(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(TypeError) as e:
            typehints.Iterable[5]
        self.assertEqual('Parameter to an Iterable hint must be a non-sequence, a type, or a TypeConstraint. 5 is an instance of int.', e.exception.args[0])

    def test_compatibility(self):
        if False:
            i = 10
            return i + 15
        self.assertCompatible(typehints.Iterable[int], typehints.List[int])
        self.assertCompatible(typehints.Iterable[int], typehints.Set[int])
        self.assertCompatible(typehints.Iterable[typehints.Any], typehints.List[typehints.Tuple[int, bool]])
        self.assertCompatible(typehints.Iterable[int], typehints.Iterable[int])
        self.assertCompatible(typehints.Iterable[typehints.Union[int, str]], typehints.Iterable[typehints.Union[int, str]])
        self.assertNotCompatible(typehints.Iterable[str], typehints.Iterable[bool])
        self.assertCompatible(typehints.Iterable[int], typehints.List[int])
        self.assertCompatible(typehints.Iterable[int], typehints.Set[int])
        self.assertCompatible(typehints.Iterable[typehints.Any], typehints.List[typehints.Tuple[int, bool]])
        self.assertCompatible(typehints.Iterable[int], typehints.Collection[int])

    def test_tuple_compatibility(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertCompatible(typehints.Iterable[int], typehints.Tuple[int, ...])
        self.assertCompatible(typehints.Iterable[SuperClass], typehints.Tuple[SubClass, ...])
        self.assertCompatible(typehints.Iterable[int], typehints.Tuple[int, int])
        self.assertCompatible(typehints.Iterable[Any], typehints.Tuple[int, float])
        self.assertCompatible(typehints.Iterable[typehints.Union[int, float]], typehints.Tuple[int, ...])
        self.assertCompatible(typehints.Iterable[typehints.Union[int, float]], typehints.Tuple[int, float])
        self.assertCompatible(typehints.Iterable[typehints.Union[int, float]], typehints.Tuple[int, float, int])

    def test_repr(self):
        if False:
            while True:
                i = 10
        hint = typehints.Iterable[typehints.Set[str]]
        self.assertEqual("Iterable[Set[<class 'str'>]]", repr(hint))

    def test_type_check_must_be_iterable(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(TypeError) as e:
            hint = typehints.Iterable[int]
            hint.type_check(5)
        self.assertEqual("Iterable type-constraint violated. Valid object instance must be of type 'iterable'. Instead, an instance of 'int' was received.", e.exception.args[0])

    def test_type_check_violation_invalid_simple_type(self):
        if False:
            print('Hello World!')
        hint = typehints.Iterable[float]
        l = set([1, 2, 3, 4])
        with self.assertRaises(TypeError):
            hint.type_check(l)

    def test_type_check_violation_valid_simple_type(self):
        if False:
            i = 10
            return i + 15
        hint = typehints.Iterable[str]
        l = ('t', 'e', 's', 't')
        self.assertIsNone(hint.type_check(l))

    def test_type_check_violation_invalid_composite_type(self):
        if False:
            i = 10
            return i + 15
        hint = typehints.Iterable[typehints.List[int]]
        l = [['t', 'e'], ['s', 't']]
        with self.assertRaises(TypeError):
            hint.type_check(l)

    def test_type_check_violation_valid_composite_type(self):
        if False:
            i = 10
            return i + 15
        hint = typehints.Iterable[typehints.List[int]]
        l = [[1, 2], [3, 4, 5]]
        self.assertIsNone(hint.type_check(l))

class TestGeneratorWrapper(TypeHintTestCase):

    def test_functions_as_regular_generator(self):
        if False:
            for i in range(10):
                print('nop')

        def count(n):
            if False:
                while True:
                    i = 10
            for i in range(n):
                yield i
        l = []
        interleave_func = lambda x: l.append(x)
        wrapped_gen = GeneratorWrapper(count(4), interleave_func)
        self.assertEqual(0, next(wrapped_gen))
        self.assertEqual((1, 2, 3), tuple(wrapped_gen))
        self.assertEqual([0, 1, 2, 3], l)

class GeneratorHintTestCase(TypeHintTestCase):

    def test_repr(self):
        if False:
            return 10
        hint = typehints.Iterator[typehints.Set[str]]
        self.assertEqual("Iterator[Set[<class 'str'>]]", repr(hint))

    def test_compatibility(self):
        if False:
            while True:
                i = 10
        self.assertCompatible(typehints.Iterator[int], typehints.Iterator[int])
        self.assertNotCompatible(typehints.Iterator[str], typehints.Iterator[float])

    def test_conversion(self):
        if False:
            while True:
                i = 10
        self.assertCompatible(typehints.Iterator[int], typehints.Generator[int])
        self.assertCompatible(typehints.Iterator[int], typehints.Generator[int, None, None])

    def test_generator_return_hint_invalid_yield_type(self):
        if False:
            return 10

        @check_type_hints
        @with_output_types(typehints.Iterator[int])
        def all_upper(s):
            if False:
                i = 10
                return i + 15
            for e in s:
                yield e.upper()
        with self.assertRaises(TypeCheckError) as e:
            next(all_upper('hello'))
        self.assertEqual("Type-hint for return type violated: Iterator[<class 'int'>] hint type-constraint violated. Expected a iterator of type <class 'int'>. Instead received a iterator of type str.", e.exception.args[0])

    def test_generator_argument_hint_invalid_yield_type(self):
        if False:
            while True:
                i = 10

        def wrong_yield_gen():
            if False:
                return 10
            for e in ['a', 'b']:
                yield e

        @check_type_hints
        @with_input_types(a=typehints.Iterator[int])
        def increment(a):
            if False:
                for i in range(10):
                    print('nop')
            return [e + 1 for e in a]
        with self.assertRaises(TypeCheckError) as e:
            increment(wrong_yield_gen())
        self.assertEqual("Type-hint for argument: 'a' violated: Iterator[<class 'int'>] hint type-constraint violated. Expected a iterator of type <class 'int'>. Instead received a iterator of type str.", e.exception.args[0])

class TakesDecoratorTestCase(TypeHintTestCase):

    def test_must_be_primitive_type_or_constraint(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(TypeError) as e:
            t = [1, 2]

            @with_input_types(a=t)
            def unused_foo(a):
                if False:
                    while True:
                        i = 10
                pass
        self.assertEqual('All type hint arguments must be a non-sequence, a type, or a TypeConstraint. [1, 2] is an instance of list.', e.exception.args[0])
        with self.assertRaises(TypeError) as e:
            t = 5

            @check_type_hints
            @with_input_types(a=t)
            def unused_foo(a):
                if False:
                    return 10
                pass
        self.assertEqual('All type hint arguments must be a non-sequence, a type, or a TypeConstraint. 5 is an instance of int.', e.exception.args[0])

    def test_basic_type_assertion(self):
        if False:
            i = 10
            return i + 15

        @check_type_hints
        @with_input_types(a=int)
        def foo(a):
            if False:
                return 10
            return a + 1
        with self.assertRaises(TypeCheckError) as e:
            m = 'a'
            foo(m)
        self.assertEqual("Type-hint for argument: 'a' violated. Expected an instance of {}, instead found an instance of {}.".format(int, type(m)), e.exception.args[0])

    def test_composite_type_assertion(self):
        if False:
            while True:
                i = 10

        @check_type_hints
        @with_input_types(a=typehints.List[int])
        def foo(a):
            if False:
                while True:
                    i = 10
            a.append(1)
            return a
        with self.assertRaises(TypeCheckError) as e:
            m = ['f', 'f']
            foo(m)
            self.assertEqual("Type-hint for argument: 'a' violated: List[<class 'int'>] hint type-constraint violated. The type of element #0 in the passed list is incorrect. Expected an instance of type int, instead received an instance of type str.", e.exception.args[0])

    def test_valid_simple_type_arguments(self):
        if False:
            print('Hello World!')

        @with_input_types(a=str)
        def upper(a):
            if False:
                return 10
            return a.upper()
        self.assertEqual('M', upper('m'))

    def test_any_argument_type_hint(self):
        if False:
            print('Hello World!')

        @check_type_hints
        @with_input_types(a=typehints.Any)
        def foo(a):
            if False:
                print('Hello World!')
            return 4
        self.assertEqual(4, foo('m'))

    def test_valid_mix_positional_and_keyword_arguments(self):
        if False:
            for i in range(10):
                print('nop')

        @check_type_hints
        @with_input_types(typehints.List[int], elem=typehints.List[int])
        def combine(container, elem):
            if False:
                for i in range(10):
                    print('nop')
            return container + elem
        self.assertEqual([1, 2, 3], combine([1, 2], [3]))

    def test_invalid_only_positional_arguments(self):
        if False:
            for i in range(10):
                print('nop')

        @check_type_hints
        @with_input_types(int, int)
        def sub(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return a - b
        with self.assertRaises(TypeCheckError) as e:
            m = 'two'
            sub(1, m)
        self.assertEqual("Type-hint for argument: 'b' violated. Expected an instance of {}, instead found an instance of {}.".format(int, type(m)), e.exception.args[0])

    def test_valid_only_positional_arguments(self):
        if False:
            i = 10
            return i + 15

        @with_input_types(int, int)
        def add(a, b):
            if False:
                while True:
                    i = 10
            return a + b
        self.assertEqual(3, add(1, 2))

class InputDecoratorTestCase(TypeHintTestCase):

    def test_valid_hint(self):
        if False:
            for i in range(10):
                print('nop')

        @with_input_types(int, int)
        def unused_add(a, b):
            if False:
                while True:
                    i = 10
            return a + b

        @with_input_types(int, b=int)
        def unused_add2(a, b):
            if False:
                i = 10
                return i + 15
            return a + b

        @with_input_types(a=int, b=int)
        def unused_add3(a, b):
            if False:
                return 10
            return a + b

    def test_invalid_kw_hint(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(TypeError, '\\[1, 2\\]'):

            @with_input_types(a=[1, 2])
            def unused_foo(a):
                if False:
                    while True:
                        i = 10
                pass

    def test_invalid_pos_hint(self):
        if False:
            return 10
        with self.assertRaisesRegex(TypeError, '\\[1, 2\\]'):

            @with_input_types([1, 2])
            def unused_foo(a):
                if False:
                    return 10
                pass

class OutputDecoratorTestCase(TypeHintTestCase):

    def test_valid_hint(self):
        if False:
            print('Hello World!')

        @with_output_types(int)
        def unused_foo():
            if False:
                i = 10
                return i + 15
            return 5

        @with_output_types(None)
        def unused_foo():
            if False:
                i = 10
                return i + 15
            return 5

        @with_output_types(Tuple[int, str])
        def unused_foo():
            if False:
                return 10
            return (5, 'bar')

    def test_no_kwargs_accepted(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(ValueError, 'must be positional'):

            @with_output_types(m=int)
            def unused_foo():
                if False:
                    print('Hello World!')
                return 5

    def test_must_be_primitive_type_or_type_constraint(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(TypeError):

            @with_output_types(5)
            def unused_foo():
                if False:
                    print('Hello World!')
                pass
        with self.assertRaises(TypeError):

            @with_output_types([1, 2])
            def unused_foo():
                if False:
                    for i in range(10):
                        print('nop')
                pass

    def test_must_be_single_return_type(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ValueError):

            @with_output_types(int, str)
            def unused_foo():
                if False:
                    return 10
                return (4, 'f')

    def test_type_check_violation(self):
        if False:
            print('Hello World!')

        @check_type_hints
        @with_output_types(int)
        def foo(a):
            if False:
                for i in range(10):
                    print('nop')
            return 'test'
        with self.assertRaises(TypeCheckError) as e:
            m = 4
            foo(m)
        self.assertEqual('Type-hint for return type violated. Expected an instance of {}, instead found an instance of {}.'.format(int, type('test')), e.exception.args[0])

    def test_type_check_simple_type(self):
        if False:
            print('Hello World!')

        @check_type_hints
        @with_output_types(str)
        def upper(a):
            if False:
                i = 10
                return i + 15
            return a.upper()
        self.assertEqual('TEST', upper('test'))

    def test_type_check_composite_type(self):
        if False:
            for i in range(10):
                print('nop')

        @check_type_hints
        @with_output_types(typehints.List[typehints.Tuple[int, int]])
        def bar():
            if False:
                for i in range(10):
                    print('nop')
            return [(i, i + 1) for i in range(5)]
        self.assertEqual([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], bar())

    def test_any_return_type_hint(self):
        if False:
            return 10

        @check_type_hints
        @with_output_types(typehints.Any)
        def bar():
            if False:
                print('Hello World!')
            return 'foo'
        self.assertEqual('foo', bar())

class CombinedReturnsAndTakesTestCase(TypeHintTestCase):

    def test_enable_and_disable_type_checking_takes(self):
        if False:
            return 10

        @with_input_types(a=int)
        def int_to_str(a):
            if False:
                while True:
                    i = 10
            return str(a)
        self.assertEqual('a', int_to_str('a'))

        @check_type_hints
        @with_input_types(a=int)
        def int_to_str(a):
            if False:
                i = 10
                return i + 15
            return str(a)
        with self.assertRaises(TypeCheckError):
            int_to_str('a')

    def test_enable_and_disable_type_checking_returns(self):
        if False:
            while True:
                i = 10

        @with_output_types(str)
        def int_to_str(a):
            if False:
                return 10
            return a
        self.assertEqual(9, int_to_str(9))

        @check_type_hints
        @with_output_types(str)
        def int_to_str(a):
            if False:
                for i in range(10):
                    print('nop')
            return a
        with self.assertRaises(TypeCheckError):
            int_to_str(9)

    def test_valid_mix_pos_and_keyword_with_both_orders(self):
        if False:
            i = 10
            return i + 15

        @with_input_types(str, start=int)
        @with_output_types(str)
        def to_upper_with_slice(string, start):
            if False:
                return 10
            return string.upper()[start:]
        self.assertEqual('ELLO', to_upper_with_slice('hello', 1))

    def test_simple_takes_and_returns_hints(self):
        if False:
            return 10

        @check_type_hints
        @with_output_types(str)
        @with_input_types(a=str)
        def to_lower(a):
            if False:
                print('Hello World!')
            return a.lower()
        self.assertEqual('m', to_lower('M'))
        with self.assertRaises(TypeCheckError):
            to_lower(5)

        @check_type_hints
        @with_output_types(str)
        @with_input_types(a=str)
        def to_lower(a):
            if False:
                print('Hello World!')
            return 9
        with self.assertRaises(TypeCheckError):
            to_lower('a')

    def test_composite_takes_and_returns_hints(self):
        if False:
            print('Hello World!')

        @check_type_hints
        @with_input_types(it=typehints.List[int])
        @with_output_types(typehints.List[typehints.Tuple[int, int]])
        def expand_ints(it):
            if False:
                print('Hello World!')
            return [(i, i + 1) for i in it]
        self.assertEqual([(0, 1), (1, 2), (2, 3)], expand_ints(list(range(3))))
        with self.assertRaises(TypeCheckError):
            expand_ints('t e s t'.split())

        @check_type_hints
        @with_output_types(typehints.List[typehints.Tuple[int, int]])
        @with_input_types(it=typehints.List[int])
        def expand_ints(it):
            if False:
                i = 10
                return i + 15
            return [str(i) for i in it]
        with self.assertRaises(TypeCheckError):
            expand_ints(list(range(2)))

class DecoratorHelpers(TypeHintTestCase):

    def test_hint_helper(self):
        if False:
            return 10
        self.assertTrue(is_consistent_with(Any, int))
        self.assertTrue(is_consistent_with(int, Any))
        self.assertTrue(is_consistent_with(str, object))
        self.assertFalse(is_consistent_with(object, str))
        self.assertTrue(is_consistent_with(str, Union[str, int]))
        self.assertFalse(is_consistent_with(Union[str, int], str))

    def test_positional_arg_hints(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(typehints.Any, _positional_arg_hints('x', {}))
        self.assertEqual(int, _positional_arg_hints('x', {'x': int}))
        self.assertEqual(typehints.Tuple[int, typehints.Any], _positional_arg_hints(['x', 'y'], {'x': int}))

    def test_getcallargs_forhints(self):
        if False:
            return 10

        def func(a, b_c, *d):
            if False:
                print('Hello World!')
            return (a, b_c, d)
        self.assertEqual({'a': Any, 'b_c': Any, 'd': Tuple[Any, ...]}, getcallargs_forhints(func, *[Any, Any]))
        self.assertEqual({'a': Any, 'b_c': Any, 'd': Tuple[Union[int, str], ...]}, getcallargs_forhints(func, *[Any, Any, str, int]))
        self.assertEqual({'a': int, 'b_c': Tuple[str, Any], 'd': Tuple[Any, ...]}, getcallargs_forhints(func, *[int, Tuple[str, Any]]))
        self.assertEqual({'a': Any, 'b_c': Any, 'd': Tuple[str, ...]}, getcallargs_forhints(func, *[Any, Any, Tuple[str, ...]]))
        self.assertEqual({'a': Any, 'b_c': Any, 'd': Tuple[Union[Tuple[str, ...], int], ...]}, getcallargs_forhints(func, *[Any, Any, Tuple[str, ...], int]))

    def test_getcallargs_forhints_builtins(self):
        if False:
            i = 10
            return i + 15
        if sys.version_info < (3, 7):
            self.assertEqual({'_': str, '__unknown__varargs': Tuple[Any, ...], '__unknown__keywords': typehints.Dict[Any, Any]}, getcallargs_forhints(str.upper, str))
            self.assertEqual({'_': str, '__unknown__varargs': Tuple[str, ...], '__unknown__keywords': typehints.Dict[Any, Any]}, getcallargs_forhints(str.strip, str, str))
            self.assertEqual({'_': str, '__unknown__varargs': Tuple[typehints.List[int], ...], '__unknown__keywords': typehints.Dict[Any, Any]}, getcallargs_forhints(str.join, str, typehints.List[int]))
        else:
            self.assertEqual({'self': str}, getcallargs_forhints(str.upper, str))
            self.assertEqual({'self': str, 'chars': Any}, getcallargs_forhints(str.strip, str))
            self.assertEqual({'self': str, 'iterable': typehints.List[int]}, getcallargs_forhints(str.join, str, typehints.List[int]))

class TestGetYieldedType(unittest.TestCase):

    def test_iterables(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(int, typehints.get_yielded_type(typehints.Iterable[int]))
        self.assertEqual(int, typehints.get_yielded_type(typehints.Iterator[int]))
        self.assertEqual(int, typehints.get_yielded_type(typehints.Generator[int]))
        self.assertEqual(int, typehints.get_yielded_type(typehints.List[int]))
        self.assertEqual(typehints.Union[int, str], typehints.get_yielded_type(typehints.Tuple[int, str]))
        self.assertEqual(int, typehints.get_yielded_type(typehints.Set[int]))
        self.assertEqual(int, typehints.get_yielded_type(typehints.FrozenSet[int]))

    def test_not_iterable(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(ValueError, 'not iterable'):
            typehints.get_yielded_type(int)

class TestCoerceToKvType(TypeHintTestCase):

    def test_coercion_success(self):
        if False:
            return 10
        cases = [((Any,), typehints.KV[Any, Any]), ((typehints.KV[Any, Any],), typehints.KV[Any, Any]), ((typehints.Tuple[str, int],), typehints.KV[str, int])]
        for (args, expected) in cases:
            self.assertEqual(typehints.coerce_to_kv_type(*args), expected)
            self.assertCompatible(args[0], expected)

    def test_coercion_fail(self):
        if False:
            while True:
                i = 10
        cases = [((str, 'label', 'producer'), 'producer.*compatible'), ((Tuple[str],), 'two components'), ((Union[str, int],), 'compatible'), ((Union,), 'compatible'), ((typehints.List[Any],), 'compatible')]
        for (args, regex) in cases:
            with self.assertRaisesRegex(ValueError, regex):
                typehints.coerce_to_kv_type(*args)

class TestParDoAnnotations(unittest.TestCase):

    def test_with_side_input(self):
        if False:
            i = 10
            return i + 15

        class MyDoFn(DoFn):

            def process(self, element: float, side_input: str) -> Iterable[KV[str, float]]:
                if False:
                    while True:
                        i = 10
                pass
        th = MyDoFn().get_type_hints()
        self.assertEqual(th.input_types, ((float, str), {}))
        self.assertEqual(th.output_types, ((KV[str, float],), {}))

    def test_pep484_annotations(self):
        if False:
            print('Hello World!')

        class MyDoFn(DoFn):

            def process(self, element: int) -> Iterable[str]:
                if False:
                    for i in range(10):
                        print('nop')
                pass
        th = MyDoFn().get_type_hints()
        self.assertEqual(th.input_types, ((int,), {}))
        self.assertEqual(th.output_types, ((str,), {}))

class TestPTransformAnnotations(unittest.TestCase):

    def test_pep484_annotations(self):
        if False:
            while True:
                i = 10

        class MyPTransform(PTransform):

            def expand(self, pcoll: PCollection[int]) -> PCollection[str]:
                if False:
                    i = 10
                    return i + 15
                return pcoll | Map(lambda num: str(num))
        th = MyPTransform().get_type_hints()
        self.assertEqual(th.input_types, ((int,), {}))
        self.assertEqual(th.output_types, ((str,), {}))

    def test_annotations_without_input_pcollection_wrapper(self):
        if False:
            print('Hello World!')

        class MyPTransform(PTransform):

            def expand(self, pcoll: int) -> PCollection[str]:
                if False:
                    while True:
                        i = 10
                return pcoll | Map(lambda num: str(num))
        error_str = 'This input type hint will be ignored and not used for type-checking purposes. Typically, input type hints for a PTransform are single (or nested) types wrapped by a PCollection, or PBegin. Got: {} instead.'.format(int)
        with self.assertLogs(level='WARN') as log:
            MyPTransform().get_type_hints()
            self.assertIn(error_str, log.output[0])

    def test_annotations_without_output_pcollection_wrapper(self):
        if False:
            while True:
                i = 10

        class MyPTransform(PTransform):

            def expand(self, pcoll: PCollection[int]) -> str:
                if False:
                    print('Hello World!')
                return pcoll | Map(lambda num: str(num))
        error_str = 'This output type hint will be ignored and not used for type-checking purposes. Typically, output type hints for a PTransform are single (or nested) types wrapped by a PCollection, PDone, or None. Got: {} instead.'.format(str)
        with self.assertLogs(level='WARN') as log:
            th = MyPTransform().get_type_hints()
            self.assertIn(error_str, log.output[0])
            self.assertEqual(th.input_types, ((int,), {}))
            self.assertEqual(th.output_types, None)

    def test_annotations_without_input_internal_type(self):
        if False:
            print('Hello World!')

        class MyPTransform(PTransform):

            def expand(self, pcoll: PCollection) -> PCollection[str]:
                if False:
                    print('Hello World!')
                return pcoll | Map(lambda num: str(num))
        th = MyPTransform().get_type_hints()
        self.assertEqual(th.input_types, ((Any,), {}))
        self.assertEqual(th.output_types, ((str,), {}))

    def test_annotations_without_output_internal_type(self):
        if False:
            print('Hello World!')

        class MyPTransform(PTransform):

            def expand(self, pcoll: PCollection[int]) -> PCollection:
                if False:
                    while True:
                        i = 10
                return pcoll | Map(lambda num: str(num))
        th = MyPTransform().get_type_hints()
        self.assertEqual(th.input_types, ((int,), {}))
        self.assertEqual(th.output_types, ((Any,), {}))

    def test_annotations_without_any_internal_type(self):
        if False:
            for i in range(10):
                print('nop')

        class MyPTransform(PTransform):

            def expand(self, pcoll: PCollection) -> PCollection:
                if False:
                    print('Hello World!')
                return pcoll | Map(lambda num: str(num))
        th = MyPTransform().get_type_hints()
        self.assertEqual(th.input_types, ((Any,), {}))
        self.assertEqual(th.output_types, ((Any,), {}))

    def test_annotations_without_input_typehint(self):
        if False:
            while True:
                i = 10

        class MyPTransform(PTransform):

            def expand(self, pcoll) -> PCollection[str]:
                if False:
                    i = 10
                    return i + 15
                return pcoll | Map(lambda num: str(num))
        th = MyPTransform().get_type_hints()
        self.assertEqual(th.input_types, ((Any,), {}))
        self.assertEqual(th.output_types, ((str,), {}))

    def test_annotations_without_output_typehint(self):
        if False:
            print('Hello World!')

        class MyPTransform(PTransform):

            def expand(self, pcoll: PCollection[int]):
                if False:
                    print('Hello World!')
                return pcoll | Map(lambda num: str(num))
        th = MyPTransform().get_type_hints()
        self.assertEqual(th.input_types, ((int,), {}))
        self.assertEqual(th.output_types, ((Any,), {}))

    def test_annotations_without_any_typehints(self):
        if False:
            return 10

        class MyPTransform(PTransform):

            def expand(self, pcoll):
                if False:
                    for i in range(10):
                        print('nop')
                return pcoll | Map(lambda num: str(num))
        th = MyPTransform().get_type_hints()
        self.assertEqual(th.input_types, None)
        self.assertEqual(th.output_types, None)

    def test_annotations_with_pbegin(self):
        if False:
            while True:
                i = 10

        class MyPTransform(PTransform):

            def expand(self, pcoll: PBegin):
                if False:
                    print('Hello World!')
                return pcoll | Map(lambda num: str(num))
        th = MyPTransform().get_type_hints()
        self.assertEqual(th.input_types, ((Any,), {}))
        self.assertEqual(th.output_types, ((Any,), {}))

    def test_annotations_with_pdone(self):
        if False:
            print('Hello World!')

        class MyPTransform(PTransform):

            def expand(self, pcoll) -> PDone:
                if False:
                    i = 10
                    return i + 15
                return pcoll | Map(lambda num: str(num))
        th = MyPTransform().get_type_hints()
        self.assertEqual(th.input_types, ((Any,), {}))
        self.assertEqual(th.output_types, ((Any,), {}))

    def test_annotations_with_none_input(self):
        if False:
            i = 10
            return i + 15

        class MyPTransform(PTransform):

            def expand(self, pcoll: None) -> PCollection[str]:
                if False:
                    print('Hello World!')
                return pcoll | Map(lambda num: str(num))
        error_str = 'This input type hint will be ignored and not used for type-checking purposes. Typically, input type hints for a PTransform are single (or nested) types wrapped by a PCollection, or PBegin. Got: {} instead.'.format(None)
        with self.assertLogs(level='WARN') as log:
            th = MyPTransform().get_type_hints()
            self.assertIn(error_str, log.output[0])
            self.assertEqual(th.input_types, None)
            self.assertEqual(th.output_types, ((str,), {}))

    def test_annotations_with_none_output(self):
        if False:
            for i in range(10):
                print('nop')

        class MyPTransform(PTransform):

            def expand(self, pcoll) -> None:
                if False:
                    print('Hello World!')
                return pcoll | Map(lambda num: str(num))
        th = MyPTransform().get_type_hints()
        self.assertEqual(th.input_types, ((Any,), {}))
        self.assertEqual(th.output_types, ((Any,), {}))

    def test_annotations_with_arbitrary_output(self):
        if False:
            print('Hello World!')

        class MyPTransform(PTransform):

            def expand(self, pcoll) -> str:
                if False:
                    for i in range(10):
                        print('nop')
                return pcoll | Map(lambda num: str(num))
        th = MyPTransform().get_type_hints()
        self.assertEqual(th.input_types, ((Any,), {}))
        self.assertEqual(th.output_types, None)

    def test_annotations_with_arbitrary_input_and_output(self):
        if False:
            while True:
                i = 10

        class MyPTransform(PTransform):

            def expand(self, pcoll: int) -> str:
                if False:
                    return 10
                return pcoll | Map(lambda num: str(num))
        input_error_str = 'This input type hint will be ignored and not used for type-checking purposes. Typically, input type hints for a PTransform are single (or nested) types wrapped by a PCollection, or PBegin. Got: {} instead.'.format(int)
        output_error_str = 'This output type hint will be ignored and not used for type-checking purposes. Typically, output type hints for a PTransform are single (or nested) types wrapped by a PCollection, PDone, or None. Got: {} instead.'.format(str)
        with self.assertLogs(level='WARN') as log:
            th = MyPTransform().get_type_hints()
            self.assertIn(input_error_str, log.output[0])
            self.assertIn(output_error_str, log.output[1])
            self.assertEqual(th.input_types, None)
            self.assertEqual(th.output_types, None)

    def test_typing_module_annotations_are_converted_to_beam_annotations(self):
        if False:
            while True:
                i = 10

        class MyPTransform(PTransform):

            def expand(self, pcoll: PCollection[typing.Dict[str, str]]) -> PCollection[typing.Dict[str, str]]:
                if False:
                    print('Hello World!')
                return pcoll
        th = MyPTransform().get_type_hints()
        self.assertEqual(th.input_types, ((typehints.Dict[str, str],), {}))
        self.assertEqual(th.input_types, ((typehints.Dict[str, str],), {}))

    def test_nested_typing_annotations_are_converted_to_beam_annotations(self):
        if False:
            for i in range(10):
                print('nop')

        class MyPTransform(PTransform):

            def expand(self, pcoll: PCollection[typing.Union[int, typing.Any, typing.Dict[str, float]]]) -> PCollection[typing.Union[int, typing.Any, typing.Dict[str, float]]]:
                if False:
                    return 10
                return pcoll
        th = MyPTransform().get_type_hints()
        self.assertEqual(th.input_types, ((typehints.Union[int, typehints.Any, typehints.Dict[str, float]],), {}))
        self.assertEqual(th.input_types, ((typehints.Union[int, typehints.Any, typehints.Dict[str, float]],), {}))

    def test_mixed_annotations_are_converted_to_beam_annotations(self):
        if False:
            return 10

        class MyPTransform(PTransform):

            def expand(self, pcoll: typing.Any) -> typehints.Any:
                if False:
                    return 10
                return pcoll
        th = MyPTransform().get_type_hints()
        self.assertEqual(th.input_types, ((typehints.Any,), {}))
        self.assertEqual(th.input_types, ((typehints.Any,), {}))

    def test_pipe_operator_as_union(self):
        if False:
            for i in range(10):
                print('nop')
        if sys.version_info.major == 3 and sys.version_info.minor >= 10:
            type_a = int | float
            type_b = typing.Union[int, float]
            self.assertEqual(native_type_compatibility.convert_to_beam_type(type_a), native_type_compatibility.convert_to_beam_type(type_b))

class TestNonBuiltInGenerics(unittest.TestCase):

    def test_no_error_thrown(self):
        if False:
            while True:
                i = 10
        input = NonBuiltInGeneric[str]
        output = typehints.normalize(input)
        self.assertEqual(input, output)
if __name__ == '__main__':
    unittest.main()