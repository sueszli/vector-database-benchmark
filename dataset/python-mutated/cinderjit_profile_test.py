import os
import sys
import threading
import unittest
from collections import Counter
from contextlib import contextmanager
from test import cinder_support
try:
    import cinder
except ModuleNotFoundError:
    cinder = None
try:
    import cinderjit
except ModuleNotFoundError:
    cinderjit = None

def check_profiling():
    if False:
        i = 10
        return i + 15
    var = os.getenv('CINDER_JIT_PROFILE_TEST_PROFILING', None)
    return var not in (None, '', '0')
PROFILING = check_profiling()
TESTING = not (PROFILING or cinderjit is None)

class ProfileTest(unittest.TestCase):

    @contextmanager
    def assertDeopts(self, deopt_specs):
        if False:
            print('Hello World!')
        'Assert that the protected region of code has exactly as many deopts as the\n        given specs. Specs are given as dict keys in deopt_specs, with the\n        values indicating how many deopts of that kind to expect.\n\n        Specs should be tuples of key-value pair tuples, and can reference any\n        of the keys and values in\n        cinderjit.get_and_clear_runtime_stats()["deopt"]. For example, to match\n        two GuardType guard failures, pass:\n\n        {(("reason", "GuardFailure"), ("description", "GuardType)): 2}\n\n        or to assert no deopts, pass an empty dict:\n\n        {}\n        '
        cinderjit.get_and_clear_runtime_stats()
        yield
        deopts = cinderjit.get_and_clear_runtime_stats()['deopt']

        def deopt_matches_spec(deopt, spec):
            if False:
                return 10
            for (key, val) in spec:
                if deopt.get(key) != val:
                    return False
            return True
        found_specs = Counter()
        for deopt in deopts:
            deopt = {**deopt['normal'], **deopt['int']}
            if deopt['filename'] != __file__:
                continue
            for spec in deopt_specs:
                if deopt_matches_spec(deopt, spec):
                    found_specs[spec] += deopt['count']
                    break
            else:
                self.fail(f"Deopt event '{deopt}' doesn't match any given specs")
        for (spec, expected_count) in deopt_specs.items():
            found_count = found_specs[spec]
            self.assertEqual(found_count, expected_count, f'Deopt spec {spec}')

class BinaryOpTests(ProfileTest):

    def test_long_power(self):
        if False:
            while True:
                i = 10

        def do_pow(a, b):
            if False:
                return 10
            return a ** b
        self.assertEqual(do_pow(2, 2), 4)
        if TESTING:
            with self.assertDeopts({}):
                self.assertEqual(do_pow(2, 8), 256)

    def assert_index_error(self):
        if False:
            i = 10
            return i + 15
        return self.assertRaisesRegex(IndexError, "cannot fit 'int' into an index-sized integer")

    def test_raising_str_subscr(self):
        if False:
            while True:
                i = 10
        s = 'abcdefu'
        with self.assert_index_error():
            return s[sys.maxsize + 1]

    def test_raising_list_subscr(self):
        if False:
            for i in range(10):
                print('nop')
        l = [1, 2, 3]
        with self.assert_index_error():
            return l[sys.maxsize + 1]

    def test_raising_tuple_subscr(self):
        if False:
            for i in range(10):
                print('nop')
        t = (1, 2, 3)
        with self.assert_index_error():
            return t[sys.maxsize + 1]

    def test_raising_str_repeat(self):
        if False:
            while True:
                i = 10
        s = '123'
        with self.assertRaisesRegex(OverflowError, "cannot fit 'int' into an index-sized integer"):
            return s * (sys.maxsize + 1)

class NewThreadTests(ProfileTest):

    def test_create_thread(self):
        if False:
            return 10
        x = 5

        def thread_body():
            if False:
                while True:
                    i = 10
            nonlocal x
            x = 10
        self.assertEqual(x, 5)
        t = threading.Thread(target=thread_body)
        t.start()
        t.join()
        self.assertEqual(x, 10)

@unittest.skipIf(cinder is None, 'tests cinder-specific functionality')
class GetProfilesTests(ProfileTest):

    @cinder_support.runInSubprocess
    def test_get_and_clear_type_profiles(self):
        if False:
            while True:
                i = 10
        if not PROFILING:
            return
        cinder.get_and_clear_type_profiles()
        result = 0
        for i in range(10):
            result += i
        self.assertEqual(result, 45)
        profiles = cinder.get_and_clear_type_profiles()
        for hit in profiles:
            self.assertIn('normal', hit)
            normal = hit['normal']
            self.assertIn('func_qualname', normal)
            if normal['func_qualname'] == 'GetProfilesTests.test_get_and_clear_type_profiles' and normal.get('opname') == 'INPLACE_ADD':
                break
        else:
            self.fail("Didn't find expected profile hit in results")

@cinder_support.failUnlessJITCompiled
def run_cls_meth(cls):
    if False:
        while True:
            i = 10
    return cls.cls_meth()

@cinder_support.failUnlessJITCompiled
def run_static_meth(cls):
    if False:
        while True:
            i = 10
    return cls.static_meth()

@cinder_support.failUnlessJITCompiled
def run_regular_meth(cls):
    if False:
        for i in range(10):
            print('nop')
    return cls.regular_meth()

class LoadTypeMethodCacheTests(ProfileTest):

    def _new_cls_meth(cls):
        if False:
            print('Hello World!')
        return 2
    new_cls_meth = classmethod(_new_cls_meth)

    def _new_static_meth():
        if False:
            for i in range(10):
                print('nop')
        return 1
    new_static_meth = staticmethod(_new_static_meth)

    def _prime_cache(self, obj, method_runner, expected_output):
        if False:
            print('Hello World!')
        self.assertEqual(method_runner(obj), expected_output)
        self.assertEqual(method_runner(obj), expected_output)

    def test_type_modified(self):
        if False:
            for i in range(10):
                print('nop')

        class Oracle:

            @classmethod
            def cls_meth(cls):
                if False:
                    while True:
                        i = 10
                return 42

            @staticmethod
            def static_meth():
                if False:
                    for i in range(10):
                        print('nop')
                return 41

            def regular_meth():
                if False:
                    for i in range(10):
                        print('nop')
                return 40
        obj = Oracle
        self._prime_cache(obj, run_cls_meth, 42)
        Oracle.cls_meth = self.new_cls_meth
        self.assertEqual(run_cls_meth(obj), 2)
        self._prime_cache(obj, run_static_meth, 41)
        Oracle.static_meth = self.new_static_meth
        self.assertEqual(run_static_meth(obj), 1)
        self._prime_cache(obj, run_regular_meth, 40)

        def _new_regular_meth():
            if False:
                i = 10
                return i + 15
            return 0
        Oracle.regular_meth = _new_regular_meth
        self.assertEqual(run_regular_meth(obj), 0)

    def test_base_type_modified(self):
        if False:
            for i in range(10):
                print('nop')

        class Base:

            @classmethod
            def cls_meth(cls):
                if False:
                    for i in range(10):
                        print('nop')
                return 42

            @staticmethod
            def static_meth():
                if False:
                    i = 10
                    return i + 15
                return 41

            def regular_meth():
                if False:
                    for i in range(10):
                        print('nop')
                return 40

        class Derived(Base):
            pass
        obj = Derived
        self._prime_cache(obj, run_cls_meth, 42)
        Base.cls_meth = self.new_cls_meth
        self.assertEqual(run_cls_meth(obj), 2)
        self._prime_cache(obj, run_static_meth, 41)
        Base.static_meth = self.new_static_meth
        self.assertEqual(run_static_meth(obj), 1)
        self._prime_cache(obj, run_regular_meth, 40)

        def _new_regular_meth():
            if False:
                i = 10
                return i + 15
            return 0
        Base.regular_meth = _new_regular_meth
        self.assertEqual(run_regular_meth(obj), 0)

    def test_cache_invalidation_when_second_base_type_modified(self):
        if False:
            while True:
                i = 10

        class Base1:
            pass

        class Base2:

            @classmethod
            def cls_meth(cls):
                if False:
                    print('Hello World!')
                return 42

            @staticmethod
            def static_meth():
                if False:
                    print('Hello World!')
                return 41

            def regular_meth():
                if False:
                    while True:
                        i = 10
                return 40

        class Derived(Base1, Base2):
            pass
        obj = Derived
        self._prime_cache(obj, run_cls_meth, 42)
        Base2.cls_meth = self.new_cls_meth
        self.assertEqual(run_cls_meth(obj), 2)
        self._prime_cache(obj, run_static_meth, 41)
        Base2.static_meth = self.new_static_meth
        self.assertEqual(run_static_meth(obj), 1)
        self._prime_cache(obj, run_regular_meth, 40)

        def _new_regular_meth():
            if False:
                return 10
            return 0
        Base2.regular_meth = _new_regular_meth
        self.assertEqual(run_regular_meth(obj), 0)

    def test_cache_invalidation_when_changing_bases(self):
        if False:
            return 10

        class Base1:

            @classmethod
            def cls_meth(cls):
                if False:
                    print('Hello World!')
                return 42

            @staticmethod
            def static_meth():
                if False:
                    for i in range(10):
                        print('nop')
                return 41

            def regular_meth():
                if False:
                    i = 10
                    return i + 15
                return 40

        class Derived(Base1):
            pass

        class Base2:

            @classmethod
            def cls_meth(cls):
                if False:
                    return 10
                return 2

            @staticmethod
            def static_meth():
                if False:
                    while True:
                        i = 10
                return 1

            def regular_meth():
                if False:
                    i = 10
                    return i + 15
                return 0
        self._prime_cache(Derived, run_cls_meth, 42)
        self._prime_cache(Derived, run_static_meth, 41)
        self._prime_cache(Derived, run_regular_meth, 40)
        Derived.__bases__ = (Base2,)
        self.assertEqual(run_cls_meth(Derived), 2)
        self.assertEqual(run_static_meth(Derived), 1)
        self.assertEqual(run_regular_meth(Derived), 0)

    def test_dyanmic_type_method_lookup(self):
        if False:
            for i in range(10):
                print('nop')

        class A:

            @classmethod
            def foo(cls):
                if False:
                    print('Hello World!')
                return 'A'

        class B:

            @classmethod
            def foo(cls):
                if False:
                    return 10
                return 'B'

        def call_foo(ty):
            if False:
                return 10
            return ty.foo()
        call_foo(A)
        call_foo(B)
        self.assertEqual(call_foo(A), 'A')
        self.assertEqual(call_foo(B), 'B')

class LoadAttrTests(ProfileTest):

    def make_slot_type(caller_name, name, slots, bases=None):
        if False:
            return 10

        def init(self, **kwargs):
            if False:
                return 10
            for (key, val) in kwargs.items():
                setattr(self, key, val)
        if bases is None:
            bases = (object,)
            slots = slots + ['__dict__']
        return type(name, bases, {'__init__': init, '__slots__': slots, '__qualname__': f'LoadAttrTests.{caller_name}.<locals>.{name}'})
    BasicSlotAttr = make_slot_type('test_load_from_slot', 'BasicSlotAttr', ['b', 'c'])
    OtherSlotAttr = make_slot_type('test_load_attr_from_slot', 'OtherSlotAttr', ['a', 'b', 'c', 'd'])

    def test_load_attr_from_slot(self):
        if False:
            while True:
                i = 10

        def get_a(o):
            if False:
                print('Hello World!')
            return o.a

        def get_b(o):
            if False:
                print('Hello World!')
            return o.b

        def get_c(o):
            if False:
                i = 10
                return i + 15
            return o.c

        def get_d(o):
            if False:
                for i in range(10):
                    print('nop')
            return o.d
        o1 = self.BasicSlotAttr(b='bee', c='see')
        self.assertEqual(get_b(o1), 'bee')
        self.assertEqual(get_c(o1), 'see')
        if TESTING:
            with self.assertDeopts({}):
                self.assertEqual(get_b(o1), 'bee')
                self.assertEqual(get_c(o1), 'see')
            o2 = self.OtherSlotAttr(a='aaa', b='bbb', c='ccc', d='ddd')
            with self.assertDeopts({(('reason', 'GuardFailure'), ('description', 'GuardType')): 2}):
                self.assertEqual(get_a(o2), 'aaa')
                self.assertEqual(get_b(o2), 'bbb')
                self.assertEqual(get_c(o2), 'ccc')
                self.assertEqual(get_d(o2), 'ddd')
    ModifiedSlotAttr = make_slot_type('test_modify_type_and_load_attr_from_slot', 'ModifiedSlotAttr', ['a', 'b'])

    def test_modify_type_and_load_attr_from_slot(self):
        if False:
            while True:
                i = 10
        o = self.ModifiedSlotAttr(a=123, b=456)
        o.__dict__ = {'a': 'shadowed a'}
        self.assertEqual(o.a, 123)

        def get_attr(o):
            if False:
                while True:
                    i = 10
            return o.a
        self.assertEqual(get_attr(o), 123)
        if TESTING:
            with self.assertDeopts({}):
                self.assertEqual(get_attr(o), 123)
                o.a = 789
                self.assertEqual(get_attr(o), 789)
            descr_saved = self.ModifiedSlotAttr.a
            del self.ModifiedSlotAttr.a
            with self.assertDeopts({(('reason', 'GuardFailure'), ('description', 'member descriptor attribute')): 3}):
                self.assertEqual(get_attr(o), 'shadowed a')
                o.a = 'another a'
                self.assertEqual(get_attr(o), 'another a')
                self.ModifiedSlotAttr.a = descr_saved
                self.assertEqual(get_attr(o), 789)
    ModifiedOtherAttr = make_slot_type('test_modify_other_type_member', 'ModifiedOtherAttr', ['foo', 'bar'])

    def test_modify_other_type_member(self):
        if False:
            for i in range(10):
                print('nop')
        o = self.ModifiedOtherAttr(foo='foo', bar='bar')

        def get_foo(o):
            if False:
                i = 10
                return i + 15
            return o.foo

        def get_bar(o):
            if False:
                i = 10
                return i + 15
            return o.bar
        self.assertEqual(get_foo(o), 'foo')
        self.assertEqual(get_bar(o), 'bar')
        if TESTING:
            self.ModifiedOtherAttr.bar = 5
            with self.assertDeopts({}):
                self.assertEqual(get_foo(o), 'foo')
            with self.assertDeopts({(('reason', 'GuardFailure'), ('description', 'member descriptor attribute')): 1}):
                self.assertEqual(get_bar(o), 5)

    class EmptyBase:
        pass
    FakeSlotType = make_slot_type('test_fake_slot_type', 'FakeSlotType', ['a', 'b'], bases=(EmptyBase,))
    FakeSlotType.c = FakeSlotType.a
    OtherFakeSlotType = make_slot_type('test_fake_slot_type', 'OtherFakeSlotType', ['a', 'b'], bases=(EmptyBase,))
    OtherFakeSlotType.c = OtherFakeSlotType.b

    def test_fake_slot_type(self):
        if False:
            print('Hello World!')
        'Test __class__ assignment where the new type has a compatible layout but the\n        "slot" we specialized on changed anyway, because it was aliasing a real\n        slot.\n        '
        o1 = self.FakeSlotType(a='a_1', b='b_1')

        def get_attrs(o, do_assign=False):
            if False:
                while True:
                    i = 10
            a = o.a
            if do_assign:
                o.__class__ = self.OtherFakeSlotType
            c = o.c
            return f'{a}-{c}'
        self.assertEqual(get_attrs(o1), 'a_1-a_1')
        if TESTING:
            with self.assertDeopts({(('reason', 'GuardFailure'), ('description', 'member descriptor attribute'), ('guilty_type', 'test_cinderx.cinderjit_profile_test:OtherFakeSlotType')): 1}):
                self.assertEqual(get_attrs(o1, True), 'a_1-b_1')

    def test_load_attr_from_split_dict(self):
        if False:
            while True:
                i = 10

        class Point:

            def __init__(self, x, y, break_dict_order=False):
                if False:
                    return 10
                if break_dict_order:
                    self.w = 'oops'
                self.x = x
                self.y = y

        class OtherPoint:

            @property
            def x(self):
                if False:
                    while True:
                        i = 10
                return 78

            @property
            def y(self):
                if False:
                    print('Hello World!')
                return 90
        p = Point(123, 456)
        op = OtherPoint()
        p_dict = Point(11, 22)
        p_dict.__dict__ = {'a': 1, 'b': 2, 'y': 3, 'x': 4}
        self.assertEqual(p.x, 123)

        def get_x(o):
            if False:
                while True:
                    i = 10
            return o.x

        def get_y(o):
            if False:
                return 10
            return o.y
        self.assertEqual(get_x(p), 123)
        self.assertEqual(get_y(p), 456)
        if TESTING:
            with self.assertDeopts({}):
                self.assertEqual(get_x(p), 123)
                self.assertEqual(get_y(p), 456)
            Point.foo = 'whatever'
            with self.assertDeopts({}):
                self.assertEqual(get_x(p), 123)
                self.assertEqual(get_y(p), 456)
            with self.assertDeopts({(('reason', 'GuardFailure'), ('description', 'ht_cached_keys comparison')): 2}):
                self.assertEqual(get_x(p_dict), 4)
                self.assertEqual(get_y(p_dict), 3)
            p2 = Point('eks', 'why', break_dict_order=True)
            with self.assertDeopts({(('reason', 'GuardFailure'), ('description', 'SplitDictDeoptPatcher')): 2}):
                self.assertEqual(get_x(p2), 'eks')
                self.assertEqual(get_y(p2), 'why')
            with self.assertDeopts({(('reason', 'GuardFailure'), ('description', 'GuardType')): 2}):
                self.assertEqual(get_x(op), 78)
                self.assertEqual(get_y(op), 90)

    def test_load_attr_from_split_dict_overwrite_type_attr(self):
        if False:
            return 10

        class Point:

            def __init__(self, x, y):
                if False:
                    while True:
                        i = 10
                self.x = x
                self.y = y

        def get_x(o):
            if False:
                return 10
            return o.x

        def get_y(o):
            if False:
                i = 10
                return i + 15
            return o.y
        p = Point(111, 222)
        self.assertEqual(get_x(p), 111)
        self.assertEqual(get_y(p), 222)
        if TESTING:
            with self.assertDeopts({}):
                self.assertEqual(get_x(p), 111)
                self.assertEqual(get_y(p), 222)
            Point.x = property(lambda self: 333)
            with self.assertDeopts({(('reason', 'GuardFailure'), ('description', 'SplitDictDeoptPatcher')): 1}):
                self.assertEqual(get_x(p), 333)
            with self.assertDeopts({}):
                self.assertEqual(get_y(p), 222)

    def test_load_attr_from_replaced_property(self):
        if False:
            i = 10
            return i + 15

        class C:

            @property
            def x(self):
                if False:
                    i = 10
                    return i + 15
                return 'hello from property'

        def load_x(o):
            if False:
                i = 10
                return i + 15
            return o.x
        c = C()
        self.assertEqual(load_x(c), 'hello from property')
        if TESTING:
            C.x = 'goodbye'
            with self.assertDeopts({(('reason', 'GuardFailure'), ('description', 'property attribute')): 1}):
                self.assertEqual(load_x(c), 'goodbye')

    def test_load_attr_from_replaced_descriptor(self):
        if False:
            i = 10
            return i + 15

        class Descr:

            def __get__(self, instance, owner=None):
                if False:
                    print('Hello World!')
                return 'descr!'

            def __set__(self, instance, value):
                if False:
                    while True:
                        i = 10
                raise RuntimeError

        class C:
            attr = Descr()

        def load_attr(o):
            if False:
                for i in range(10):
                    print('nop')
            return o.attr
        c = C()
        self.assertEqual(load_attr(c), 'descr!')
        if TESTING:
            del Descr.__get__
            with self.assertDeopts({(('reason', 'GuardFailure'), ('description', 'tp_descr_get/tp_descr_set')): 1}):
                self.assertIs(load_attr(c), C.attr)
            C.attr = 'not descr'
            with self.assertDeopts({(('reason', 'GuardFailure'), ('description', 'generic descriptor attribute')): 1}):
                self.assertEqual(load_attr(c), 'not descr')

class Duck:

    def speak(self):
        if False:
            print('Hello World!')
        return 'quack'

class ShimmedDuck(Duck):
    pass
GLOBAL_DUCK = Duck() if PROFILING else ShimmedDuck()

class FailingGuardTest(ProfileTest):

    def test_always_failing_guard(self):
        if False:
            while True:
                i = 10
        self.assertEqual(GLOBAL_DUCK.speak(), 'quack')