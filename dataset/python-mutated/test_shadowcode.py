import builtins
import gc
import sys
import unittest
import weakref
from collections import UserDict
from test.support.script_helper import assert_python_ok, run_python_until_end
from unittest import skipIf
from test.cinder_support import CINDERJIT_ENABLED
from types import FunctionType
import cinder
from cinder import cached_property, StrictModule, strict_module_patch
REPETITION = 100

class ShadowError(Exception):
    pass
knobs = cinder.getknobs()
if 'shadowcode' in knobs:
    cinder.setknobs({'shadowcode': True})
    cinder.setknobs({'polymorphiccache': True})

def skip_ret_code_check_for_leaking_test_in_asan_mode(*args, **env_vars):
    if False:
        print('Hello World!')
    if cinder._built_with_asan:
        (res, _) = run_python_until_end(*args, **env_vars)
        return res
    else:
        return assert_python_ok(*args, **env_vars)

class ShadowCodeTests(unittest.TestCase):

    def test_type_error(self):
        if False:
            print('Hello World!')

        class Desc:

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.error = False

            def __get__(self, inst, ctx):
                if False:
                    while True:
                        i = 10
                if self.error:
                    raise ShadowError()
                return 42
        desc = Desc()

        class C:
            prop = desc

        def f(x):
            if False:
                while True:
                    i = 10
            return x.prop
        for _ in range(REPETITION):
            self.assertEqual(f(C), 42)
        desc.error = True
        for _ in range(REPETITION):
            self.assertRaises(ShadowError, f, C)

    def test_module_error(self):
        if False:
            i = 10
            return i + 15
        sys.prop = 42

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return x.prop
        for _ in range(REPETITION):
            self.assertEqual(f(sys), 42)
        del sys.prop
        for _ in range(REPETITION):
            self.assertRaises(AttributeError, f, sys)

    def test_load_attr_no_dict_descr_error(self):
        if False:
            while True:
                i = 10

        class Desc:

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.error = False

            def __get__(self, inst, ctx):
                if False:
                    return 10
                if self.error:
                    raise ShadowError()
                return 42
        desc = Desc()

        class C:
            __slots__ = ()
            prop = desc

        def f(x):
            if False:
                i = 10
                return i + 15
            return x.prop
        a = C()
        for _ in range(REPETITION):
            self.assertEqual(f(a), 42)
        desc.error = True
        for _ in range(REPETITION):
            self.assertRaises(ShadowError, f, a)

    def test_load_attr_dict_descr_error(self):
        if False:
            i = 10
            return i + 15

        class Desc:

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                self.error = False

            def __get__(self, inst, ctx):
                if False:
                    for i in range(10):
                        print('nop')
                if self.error:
                    raise ShadowError()
                return 42
        desc = Desc()

        class C:
            prop = desc

        def f(x):
            if False:
                while True:
                    i = 10
            return x.prop
        a = C()
        a.foo = 100
        a.bar = 200
        b = C()
        b.quox = 100
        c = C()
        c.blah = 300
        for _ in range(REPETITION):
            self.assertEqual(f(c), 42)
        desc.error = True
        for _ in range(REPETITION):
            self.assertRaises(ShadowError, f, c)

    def test_load_attr_dict_no_item(self):
        if False:
            i = 10
            return i + 15

        class C:
            pass

        def f(x):
            if False:
                i = 10
                return i + 15
            return x.prop
        a = C()
        a.foo = 100
        a.bar = 200
        b = C()
        b.quox = 100
        c = C()
        c.prop = 42
        for _ in range(REPETITION):
            self.assertEqual(f(c), 42)
        for _ in range(REPETITION):
            self.assertRaises(AttributeError, f, b)

    def test_split_dict_append(self):
        if False:
            i = 10
            return i + 15
        "add a property to a split dictionary that aliases is a descriptor\nproperty after we've already cached the non-existance of the split property"

        class C:

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.x = 1
            prop = 42

        def f(x):
            if False:
                while True:
                    i = 10
            return x.prop
        a = C()
        for _ in range(REPETITION):
            self.assertEqual(f(a), 42)
        a.prop = 100
        for _ in range(REPETITION):
            self.assertEqual(f(a), 100)

    def test_class_overflow(self):
        if False:
            for i in range(10):
                print('nop')

        def make_class():
            if False:
                print('Hello World!')

            class C:

                def __init__(self):
                    if False:
                        for i in range(10):
                            print('nop')
                    self.prop = 1
            return C

        def f(x):
            if False:
                print('Hello World!')
            return x.prop
        a = make_class()()
        for _ in range(REPETITION):
            self.assertEqual(f(a), 1)
        for _ in range(300):
            a = make_class()()
            self.assertEqual(f(a), 1)

    def test_dict(self):
        if False:
            print('Hello World!')

        class C:
            pass
        a = C()
        a.foo = 1
        a.bar = 2
        b = C()
        b.bar = 1
        b.baz = 2

        def f(x):
            if False:
                return 10
            return x.bar
        for _i in range(REPETITION):
            self.assertEqual(f(b), 1)
        C.bar = property(lambda self: 42)
        for _i in range(REPETITION):
            self.assertEqual(f(b), 42)

    def test_split_dict_no_descr(self):
        if False:
            for i in range(10):
                print('nop')

        class C:

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.foo = 1
                self.bar = 2
                self.baz = 3
                self.quox = 3
                self.foo1 = 1
                self.bar2 = 2
                self.baz3 = 3
                self.quox4 = 3
        a = C()
        b = C()

        def f(x):
            if False:
                while True:
                    i = 10
            return x.foo
        for _i in range(REPETITION):
            self.assertEqual(f(a), 1)
        a.foo = 2
        for _i in range(REPETITION):
            self.assertEqual(f(a), 2)
        for _i in range(REPETITION):
            self.assertEqual(f(b), 1)
        C.foo = property(lambda self: 100)
        for _i in range(REPETITION):
            self.assertEqual(f(b), 100)
        C.foo = 100
        for _i in range(REPETITION):
            self.assertEqual(f(b), 1)

    def test_split_dict_descr(self):
        if False:
            i = 10
            return i + 15

        class C:
            foo = 100

            def __init__(self, foo=True):
                if False:
                    for i in range(10):
                        print('nop')
                self.bar = 2
                self.baz = 3
                self.quox = 3
                self.foo1 = 1
                self.bar2 = 2
                self.baz3 = 3
                self.quox4 = 3
                if foo:
                    self.foo = 1
        a = C()
        b = C(False)

        def f(x):
            if False:
                print('Hello World!')
            return x.foo
        for _i in range(REPETITION):
            self.assertEqual(f(a), 1)
        for _i in range(REPETITION):
            self.assertEqual(f(b), 100)
        C.foo = property(lambda self: 100)
        for _i in range(REPETITION):
            self.assertEqual(f(b), 100)

    def test_dict_descr_set_no_get(self):
        if False:
            for i in range(10):
                print('nop')

        class Descr:

            def __set__(self, obj, value):
                if False:
                    return 10
                pass

        class C:
            x = Descr()
        a = C()
        a.foo = 1
        b = C()
        b.bar = 2
        c_no_x = C()
        c_x = C()
        c_x.__dict__['x'] = 42

        def f(c):
            if False:
                for i in range(10):
                    print('nop')
            return c.x
        for _i in range(REPETITION):
            self.assertIs(f(c_no_x), C.x)
            self.assertEqual(f(c_x), 42)

    def test_split_dict_descr_set_no_get(self):
        if False:
            i = 10
            return i + 15

        class Descr:

            def __set__(self, obj, value):
                if False:
                    return 10
                pass

        class C:
            x = Descr()
        c_no_x = C()
        c_x = C()
        c_x.__dict__['x'] = 42

        def f(c):
            if False:
                while True:
                    i = 10
            return c.x
        for _i in range(REPETITION):
            self.assertIs(f(c_no_x), C.x)
            self.assertEqual(f(c_x), 42)

    def test_module(self):
        if False:
            i = 10
            return i + 15
        version = sys.version

        def f():
            if False:
                i = 10
                return i + 15
            return sys.version
        for _i in range(REPETITION):
            self.assertEqual(f(), version)
        sys.version = '2.8'
        try:
            for _i in range(REPETITION):
                self.assertEqual(f(), '2.8')
        finally:
            sys.version = version

    def test_type_attr_metaattr(self):
        if False:
            print('Hello World!')

        class MC(type):
            x = 100

        class C(metaclass=MC):
            x = 42

        def f(x):
            if False:
                i = 10
                return i + 15
            return x.x
        for _i in range(REPETITION):
            self.assertEqual(f(C), 42)

    def test_type_attr_no_double_invoke(self):
        if False:
            for i in range(10):
                print('nop')
        'verify that a descriptor only gets invoked once when it raises'

        class Desc:

            def __init__(self):
                if False:
                    return 10
                self.error = False
                self.calls = 0

            def __get__(self, inst, ctx):
                if False:
                    i = 10
                    return i + 15
                self.calls += 1
                if self.error:
                    raise ShadowError()
                return 42
        desc = Desc()

        class C:
            prop = desc

        def f(x):
            if False:
                i = 10
                return i + 15
            return x.prop
        for _i in range(REPETITION):
            self.assertEqual(f(C), 42)
        self.assertEqual(desc.calls, REPETITION)
        desc.error = True
        self.assertRaises(ShadowError, f, C)
        self.assertEqual(desc.calls, REPETITION + 1)

    def test_no_dict_descr_builtin(self):
        if False:
            while True:
                i = 10
        x = 42

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return x.real
        for _i in range(REPETITION):
            self.assertEqual(f(x), 42)

    def test_no_dict_descr_user(self):
        if False:
            return 10

        class C:
            __slots__ = ()

            @property
            def abc(self):
                if False:
                    i = 10
                    return i + 15
                return 42
        x = C()

        def f(x):
            if False:
                return 10
            return x.abc
        for _i in range(REPETITION):
            self.assertEqual(f(x), 42)
        C.abc = property(lambda self: 100)
        for _i in range(REPETITION):
            self.assertEqual(f(x), 100)

    def test_no_dict(self):
        if False:
            i = 10
            return i + 15

        class C:
            __slots__ = ()
            abc = 42
        x = C()

        def f(x):
            if False:
                print('Hello World!')
            return x.abc
        for _i in range(REPETITION):
            self.assertEqual(f(x), 42)
        C.abc = 100
        for _i in range(REPETITION):
            self.assertEqual(f(x), 100)

    def test_dict_descr(self):
        if False:
            for i in range(10):
                print('nop')
        'shadowing a class member should give the instance'

        class C:

            def x(self):
                if False:
                    print('Hello World!')
                return 1

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.x = 1
        a = C()

        def f(x):
            if False:
                return 10
            self.assertEqual(x.x, 1)
        for _i in range(REPETITION):
            f(a)

    def test_dict_descr_2(self):
        if False:
            while True:
                i = 10
        'getting a descriptor should return a new instance'

        class C:

            def x(self):
                if False:
                    print('Hello World!')
                return 1
        a = C()

        def f(x):
            if False:
                return 10
            return x.x
        items = []
        for _i in range(REPETITION):
            items.append(f(a))
        self.assertEqual(len({id(item) for item in items}), REPETITION)

    def test_dict_descr_3(self):
        if False:
            print('Hello World!')

        class C:

            def __init__(self, order):
                if False:
                    print('Hello World!')
                if order:
                    self.x = 1
                    self.y = 2
                else:
                    self.y = 1
                    self.x = 2

            def z(self):
                if False:
                    return 10
                return 42
        a = C(0)
        a = C(1)

        def f(x):
            if False:
                print('Hello World!')
            self.assertEqual(a.z(), 42)
        for _ in range(100):
            f(a)

    def test_type_attr(self):
        if False:
            for i in range(10):
                print('nop')

        class C:
            x = 1

        def f(x, expected):
            if False:
                while True:
                    i = 10
            self.assertEqual(x.x, expected)
        for _ in range(REPETITION):
            f(C, 1)
        C.x = 2
        for _ in range(REPETITION):
            f(C, 2)

    def test_instance_attr(self):
        if False:
            for i in range(10):
                print('nop')
        "LOAD_ATTR_DICT_DESCR -> LOAD_ATTR_NO_DICT\nWe generate a cached opcode that handles a dict, then transition over to one\nthat doesn't need a dict lookup"

        class C:

            def f(self):
                if False:
                    i = 10
                    return i + 15
                return 42
        a = C()

        def f(x):
            if False:
                return 10
            return x.f

        def g(x):
            if False:
                i = 10
                return i + 15
            return x.f
        for _ in range(REPETITION):
            self.assertEqual(f(a)(), 42)
        C.f = property(lambda x: 100)
        for _ in range(REPETITION):
            self.assertEqual(g(a), 100)
        f(a)

    def test_megamorphic(self):
        if False:
            print('Hello World!')

        class C:
            x = 0

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return x.x
        a = C()
        for i in range(REPETITION):
            self.assertEqual(f(a), i)
            C.x += 1

    def test_modify_class(self):
        if False:
            print('Hello World!')
        for i in range(REPETITION):

            class lazy_classproperty(object):

                def __init__(self, fget):
                    if False:
                        for i in range(10):
                            print('nop')
                    self._fget = fget
                    self.__doc__ = fget.__doc__
                    self.__name__ = fget.__name__
                    self.count = 0

                def __get__(self, obj, obj_cls_type):
                    if False:
                        while True:
                            i = 10
                    value = self._fget(obj_cls_type)
                    self.count += 1
                    if self.count == i:
                        setattr(obj_cls_type, self.__name__, value)
                    return value

            class C:

                @lazy_classproperty
                def f(cls):
                    if False:
                        i = 10
                        return i + 15
                    return 42
            a = C()
            exec("\ndef f(x):\n    z = x.f\n    if z != 42: self.fail('err')\n        ", locals(), globals())
            for _ in range(REPETITION * 2):
                f(C)

    def test_extended_arg(self):
        if False:
            for i in range(10):
                print('nop')
        'tests patching an opcode with EXTENDED_ARG and inserting a nop in\nplace of the extended arg opcode'

        class C:

            def __init__(self):
                if False:
                    return 10
                self.ext = 0
                for i in range(256):
                    setattr(self, 'x' + hex(i)[2:], i)
        f = self.make_large_func(args='x, false = False', add='x.x{}', size=256, skip=8)
        a = C()
        for _ in range(REPETITION):
            self.assertEqual(f(a), 32612)

    def test_cache_global_reuse(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a, b):
            if False:
                while True:
                    i = 10
            return min(a, b) + min(a, b) + min(a, b)
        for i in range(REPETITION):
            self.assertEqual(f(i, i + 1), i * 3)

    def make_large_func(self, globals=None, args='', add='x{}', start=0, size=300, skip=None):
        if False:
            for i in range(10):
                print('nop')
        code = 'def f(' + args + '):\n    res = 0\n'
        if skip:
            code += '    if false:\n'
        for i in range(start, size):
            indent = '    '
            if skip and i < skip:
                indent += '    '
            code += indent + 'res += ' + add.format(hex(i)[2:]) + '\n'
        code += '    return res'
        locals = {}
        exec(code, globals if globals is not None else {}, locals)
        return locals['f']

    def test_global_cache_exhausted(self):
        if False:
            i = 10
            return i + 15
        globals = {}
        for i in range(300):
            globals['x' + hex(i)[2:]] = i
        f = self.make_large_func(globals)
        for _ in range(REPETITION):
            self.assertEqual(f(), 44850)

    def test_global_invalidate_builtins(self):
        if False:
            return 10
        global X
        X = 1

        def f():
            if False:
                while True:
                    i = 10
            return X
        for i in range(REPETITION):
            self.assertEqual(f(), 1)
        try:
            builtins.__dict__[42] = 42
        finally:
            del builtins.__dict__[42]

    def test_reoptimize_no_caches(self):
        if False:
            print('Hello World!')
        "we limit caches to 256 per method.  If we take a EXTENDED_ARG cache, optimize it,\n        and then don't have any other spaces for caches, we fail to replace the cache.  We\n        should maintain the ref count on the previous cache correctly."
        COUNT = 400
        TOTAL = 79800
        klass = '\nclass C:\n    def __init__(self, flag=False):\n        if flag:\n            self.foo = 42\n' + '\n'.join((f'        self.x{i} = {i}' for i in range(COUNT)))
        d = {}
        exec(klass, globals(), d)
        accesses = '\n'.join((f'        if min < {i} < max: res += inst.x{i}' for i in range(COUNT)))
        func = f'\ndef f(min, max, inst, path=False):\n    res = 0\n    if path:\n{accesses}\n    else:\n{accesses}\n    return res\n'
        exec(func, globals(), d)
        C = d['C']
        a = C()
        f = d['f']
        for i in range(REPETITION):
            self.assertEqual(f(260, 270, a), 2385)
            self.assertEqual(f(260, 270, a, True), 2385)
        self.assertEqual(f(0, COUNT, a), TOTAL)
        self.assertEqual(f(0, COUNT, a, True), TOTAL)
        a = C(True)
        for i in range(REPETITION):
            self.assertEqual(f(260, 262, a), 261)
        self.assertEqual(f(0, COUNT, a), TOTAL)

    def test_cache_exhausted(self):
        if False:
            for i in range(10):
                print('nop')
        'tests running out of cache instances'

        class C:

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.ext = 0
                for i in range(256):
                    setattr(self, 'x' + hex(i)[2:], i)
        f = self.make_large_func(args='x', add='x.x{}', size=256)
        a = C()
        for _ in range(REPETITION):
            self.assertEqual(f(a), 32640)

    def test_l2_cache_hit_afer_exhaustion(self):
        if False:
            for i in range(10):
                print('nop')
        'tests running out of cache instances, and then having another\nfunction grab those instances from the L2 cache'

        class C:

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.ext = 0
                for i in range(300):
                    setattr(self, 'x' + hex(i)[2:], i)
        f = self.make_large_func(args='x', add='x.x{}', size=300)
        a = C()
        for _ in range(REPETITION):
            self.assertEqual(f(a), 44850)
        f = self.make_large_func(args='x', add='x.x{}', start=256, size=300)
        for _ in range(REPETITION):
            self.assertEqual(f(a), 12210)

    def test_modify_descriptor(self):
        if False:
            return 10
        "changing a descriptor into a plain old value shouldn't crash"

        class mydesc(object):

            def __get__(self, inst, ctx):
                if False:
                    for i in range(10):
                        print('nop')
                return 42

            def __repr__(self):
                if False:
                    while True:
                        i = 10
                return 'mydesc'

        class myobj:
            __slots__ = []
            desc = mydesc()

        def f(x):
            if False:
                return 10
            return x.desc
        for i in range(REPETITION):
            self.assertEqual(42, f(myobj()))
        del mydesc.__get__
        self.assertEqual(repr(f(myobj())), 'mydesc')

    def test_type_resurrection(self):
        if False:
            i = 10
            return i + 15

        class metafin(type):

            def __del__(self):
                if False:
                    print('Hello World!')
                nonlocal C
                C = self

        class C(metaclass=metafin):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.abc = 42
                self.foo = 200

        def f(x):
            if False:
                return 10
            return x.abc

        def g(x):
            if False:
                while True:
                    i = 10
            return x.foo
        a = C()
        for _ in range(REPETITION):
            self.assertEqual(f(a), 42)
        if not CINDERJIT_ENABLED:
            self.assertNotEqual(len(weakref.getweakrefs(C)), 0)
        del a, C, metafin
        gc.collect()
        self.assertEqual(len(weakref.getweakrefs(C)), 0)
        a = C()
        C.abc = property(lambda x: 100)
        self.assertEqual(f(a), 100)
        for _ in range(REPETITION):
            self.assertEqual(g(a), 200)
        if not CINDERJIT_ENABLED:
            self.assertNotEqual(len(weakref.getweakrefs(C)), 0)

    def test_type_resurrection_2(self):
        if False:
            while True:
                i = 10

        class metafin(type):

            def __del__(self):
                if False:
                    while True:
                        i = 10
                nonlocal C
                C = self

        class C(metaclass=metafin):
            abc = 42

        def f(x):
            if False:
                print('Hello World!')
            return x.abc
        a = C()
        for _ in range(REPETITION):
            self.assertEqual(f(a), 42)
        del a, C, metafin
        gc.collect()
        a = C()
        C.abc = 100
        self.assertEqual(f(a), 100)

    def test_descriptor_ends_split_dict(self):
        if False:
            return 10
        for x in range(REPETITION):
            mutating = False

            class myprop:

                def __init__(self, func):
                    if False:
                        return 10
                    self.func = func

                def __get__(self, inst, ctx):
                    if False:
                        print('Hello World!')
                    return self.func(inst)

            class myclass(object):

                def __init__(self):
                    if False:
                        for i in range(10):
                            print('nop')
                    self.quox = 100
                    self.baz = 200

                @myprop
                def abc(self):
                    if False:
                        while True:
                            i = 10
                    if mutating and 'quox' in self.__dict__:
                        del self.quox
                    return self.baz
            l = g = {}
            exec('\ndef f(x):\n    return x.abc', l, g)
            f = l['f']
            inst = myclass()
            for i in range(REPETITION):
                if i == x:
                    mutating = True
                self.assertEqual(f(inst), 200)

    def test_eq_side_effects(self):
        if False:
            while True:
                i = 10
        'dict key which overrides __eq__ and mutates the class during a get'
        for x in range(REPETITION):
            mutating = False

            class funkyattr:

                def __init__(self, name):
                    if False:
                        i = 10
                        return i + 15
                    self.name = name
                    self.hash = hash(name)

                def __eq__(self, other):
                    if False:
                        return 10
                    if mutating:
                        if hasattr(myobj, 'foo'):
                            del myobj.foo
                        return False
                    if isinstance(other, str):
                        return other == self.name
                    return self is other

                def __hash__(self):
                    if False:
                        print('Hello World!')
                    return self.hash

            class myobj:
                foo = 2000
            inst = myobj()
            inst.__dict__[funkyattr('foo')] = 42

            def f(x):
                if False:
                    print('Hello World!')
                return x.foo
            for i in range(REPETITION):
                if i == x:
                    mutating = True
                res = f(inst)
                if i == x:
                    self.assertEqual(res, 2000, repr(i))
                    mutating = False
                else:
                    self.assertEqual(res, 42)
                res = f(inst)
                self.assertEqual(res, 42)

    def test_knob(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            knobs = cinder.getknobs()
            self.assertEqual(knobs['shadowcode'], True)
            cinder.setknobs({'shadowcode': False})
            knobs = cinder.getknobs()
            self.assertEqual(knobs['shadowcode'], False)
        finally:
            cinder.setknobs({'shadowcode': True})
            knobs = cinder.getknobs()
            self.assertEqual(knobs['shadowcode'], True)

    def test_store_attr_dict(self):
        if False:
            for i in range(10):
                print('nop')

        class C:

            def __init__(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                if x is True:
                    self.x = 1
                    self.y = 2
                elif x is False:
                    self.y = 2
                    self.x = 1

        def f(x):
            if False:
                return 10
            x.z = 100

        class D:
            pass
        for _ in range(REPETITION):
            a = C(True)
            f(a)
            self.assertEqual(a.z, 100)
            b = C(True)
            f(b)
            self.assertEqual(b.z, 100)
            c = C(None)
            f(c)
            self.assertEqual(c.z, 100)
        x = D()
        f(x)
        self.assertEqual(x.z, 100)

    def test_store_attr_dict_type_change(self):
        if False:
            while True:
                i = 10

        class C:

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.x = 42

        def f(x):
            if False:
                while True:
                    i = 10
            x.z = 100
        for _ in range(REPETITION):
            x = C()
            f(x)
            self.assertEqual(x.z, 100)
        C.foo = 100
        x = C()
        f(x)
        self.assertEqual(x.z, 100)

    def test_store_attr_descr(self):
        if False:
            i = 10
            return i + 15

        class C:

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.x = 42

            @property
            def f(self):
                if False:
                    return 10
                return 42

            @f.setter
            def f(self, value):
                if False:
                    i = 10
                    return i + 15
                self.x = value

        def f(x):
            if False:
                i = 10
                return i + 15
            x.f = 100
        for _ in range(REPETITION):
            x = C()
            f(x)
            self.assertEqual(x.x, 100)

    def test_store_attr_descr_type_change(self):
        if False:
            print('Hello World!')

        class C:

            def __init__(self):
                if False:
                    print('Hello World!')
                self.x = 42

            @property
            def f(self):
                if False:
                    while True:
                        i = 10
                return 42

            @f.setter
            def f(self, value):
                if False:
                    print('Hello World!')
                self.x = value

        def f(x):
            if False:
                return 10
            x.f = 100
        for _ in range(REPETITION):
            x = C()
            f(x)
            self.assertEqual(x.x, 100)

        def setter(self, value):
            if False:
                for i in range(10):
                    print('nop')
            self.y = value
        C.f = property(None, setter)
        x = C()
        f(x)
        self.assertEqual(x.y, 100)

    def test_store_attr_descr_error(self):
        if False:
            return 10
        should_raise = False

        class C:

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.x = 42

            @property
            def f(self):
                if False:
                    while True:
                        i = 10
                return 42

            @f.setter
            def f(self, value):
                if False:
                    for i in range(10):
                        print('nop')
                if should_raise:
                    raise ValueError('no way')
                self.x = value

        def f(x):
            if False:
                i = 10
                return i + 15
            x.f = 100
        for _ in range(REPETITION):
            x = C()
            f(x)
            self.assertEqual(x.x, 100)
        should_raise = True
        with self.assertRaisesRegex(ValueError, 'no way'):
            f(C())

    def test_no_attr(self):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                return 10
            x.foo = 42
        for _ in range(REPETITION):
            with self.assertRaisesRegex(AttributeError, "'object' object has no attribute 'foo'"):
                f(object())

    def test_read_only_attr(self):
        if False:
            print('Hello World!')

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            x.__str__ = 42
        for _ in range(REPETITION):
            with self.assertRaisesRegex(AttributeError, "'object' object attribute '__str__' is read-only"):
                f(object())

    def test_split_dict_creation(self):
        if False:
            while True:
                i = 10

        class C:

            def __init__(self, init):
                if False:
                    i = 10
                    return i + 15
                if init:
                    self.a = 1
                    self.b = 2
                    self.c = 3

        def f(x):
            if False:
                print('Hello World!')
            x.a = 100
        for _ in range(REPETITION):
            x = C(True)
            f(x)
            self.assertEqual(x.a, 100)
        x = C(False)
        f(x)
        self.assertEqual(x.a, 100)

    def test_split_dict_not_split(self):
        if False:
            while True:
                i = 10

        class C:

            def __init__(self, init):
                if False:
                    print('Hello World!')
                if init:
                    self.a = 1
                    self.b = 2
                    self.c = 3

        def f(x):
            if False:
                while True:
                    i = 10
            x.a = 100
        for _ in range(REPETITION):
            x = C(True)
            f(x)
            self.assertEqual(x.a, 100)
        x = C(False)
        x.other = 42
        f(x)
        self.assertEqual(x.a, 100)

    def test_split_replace_existing_attr(self):
        if False:
            print('Hello World!')

        class C:

            def __init__(self):
                if False:
                    print('Hello World!')
                self.a = 1
                self.b = 2
                self.c = 3

        def f(x):
            if False:
                i = 10
                return i + 15
            x.a = 100
        for _ in range(REPETITION):
            x = C()
            f(x)
            self.assertEqual(x.a, 100)

    def test_split_dict_next_attr(self):
        if False:
            while True:
                i = 10

        class C:

            def __init__(self, init):
                if False:
                    i = 10
                    return i + 15
                self.a = 1
                self.b = 2
                if init:
                    self.c = 3

        def f(x):
            if False:
                print('Hello World!')
            x.c = 100
        for _ in range(REPETITION):
            x = C(True)
            f(x)
            self.assertEqual(x.c, 100)
        x = C(False)
        f(x)
        self.assertEqual(x.c, 100)

    def test_split_dict_start_tracking(self):
        if False:
            for i in range(10):
                print('nop')
        dels = 0

        class C:

            def __init__(self):
                if False:
                    print('Hello World!')
                self.a = 1

            def __del__(self):
                if False:
                    i = 10
                    return i + 15
                nonlocal dels
                dels += 1

        def f(x, v):
            if False:
                i = 10
                return i + 15
            x.b = v
        for _ in range(REPETITION):
            x = C()
            f(x, x)
            self.assertEqual(x.b, x)
            del x
        gc.collect()
        self.assertEqual(dels, REPETITION)

    def test_load_method_builtin(self):
        if False:
            i = 10
            return i + 15
        'INVOKE_METHOD on a builtin object w/o a dictionary'
        x = 'abc'

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return x.upper()
        for _ in range(REPETITION):
            self.assertEqual(f(x), 'ABC')

    def test_load_method_no_dict(self):
        if False:
            while True:
                i = 10
        'INVOKE_METHOD on a user defined object w/o a dictionary'

        class C:
            __slots__ = ()

            def f(self):
                if False:
                    print('Hello World!')
                return 42
        a = C()

        def f(x):
            if False:
                return 10
            return x.f()
        for _ in range(REPETITION):
            self.assertEqual(f(a), 42)

    def test_load_method_no_dict_invalidate(self):
        if False:
            for i in range(10):
                print('nop')

        class C:
            __slots__ = ()

            def f(self):
                if False:
                    return 10
                return 42
        a = C()

        def f(x):
            if False:
                while True:
                    i = 10
            return x.f()
        for _ in range(REPETITION):
            self.assertEqual(f(a), 42)
        C.f = lambda self: 100
        for _ in range(REPETITION):
            self.assertEqual(f(a), 100)

    def test_load_method_no_dict_invalidate_to_prop(self):
        if False:
            print('Hello World!')
        'switch from a method to a descriptor'

        class C:
            __slots__ = ()

            def f(self):
                if False:
                    for i in range(10):
                        print('nop')
                return 42
        a = C()

        def f(x):
            if False:
                while True:
                    i = 10
            return x.f()
        for _ in range(REPETITION):
            self.assertEqual(f(a), 42)
        C.f = property(lambda *args: lambda : 100)
        for _ in range(REPETITION):
            self.assertEqual(f(a), 100)

    def test_load_method_non_desc(self):
        if False:
            return 10
        "INVOKE_METHOD on a user defined object which isn't a descriptor"

        class callable:

            def __call__(self, *args):
                if False:
                    print('Hello World!')
                return 42

        class C:
            __slots__ = ()
            f = callable()
        a = C()

        def f(x):
            if False:
                while True:
                    i = 10
            return x.f()
        for _ in range(REPETITION):
            self.assertEqual(f(a), 42)

    def test_load_method_non_desc_invalidate(self):
        if False:
            i = 10
            return i + 15
        "INVOKE_METHOD on a user defined object which isn't a descriptor\n        and then modify the type"

        class callable:

            def __init__(self, value):
                if False:
                    print('Hello World!')
                self.value = value

            def __call__(self, *args):
                if False:
                    i = 10
                    return i + 15
                return self.value

        class C:
            __slots__ = ()
            f = callable(42)
        a = C()

        def f(x):
            if False:
                return 10
            return x.f()
        for _ in range(REPETITION):
            self.assertEqual(f(a), 42)
        C.f = callable(100)
        for _ in range(REPETITION):
            self.assertEqual(f(a), 100)

    def test_load_method_non_desc_invalidate_to_method(self):
        if False:
            while True:
                i = 10
        "INVOKE_METHOD on a user defined object which isn't a descriptor\n        and then modify the type"

        class callable:

            def __call__(self, *args):
                if False:
                    for i in range(10):
                        print('nop')
                return 42

        class C:
            __slots__ = ()
            f = callable()
        a = C()

        def f(x):
            if False:
                print('Hello World!')
            return x.f()
        for _ in range(REPETITION):
            self.assertEqual(f(a), 42)
        C.f = lambda self: 100
        for _ in range(REPETITION):
            self.assertEqual(f(a), 100)

    def test_load_method_with_dict(self):
        if False:
            print('Hello World!')

        class C:

            def f(self):
                if False:
                    print('Hello World!')
                return 42
        a = C()

        def f(x):
            if False:
                while True:
                    i = 10
            return x.f()
        for _ in range(REPETITION):
            self.assertEqual(f(a), 42)

    def test_load_method_with_dict_set_value(self):
        if False:
            while True:
                i = 10

        class C:

            def f(self):
                if False:
                    return 10
                return 42
        a = C()

        def f(x):
            if False:
                while True:
                    i = 10
            return x.f()
        for _ in range(REPETITION):
            self.assertEqual(f(a), 42)
        a.f = lambda : 100
        for _ in range(REPETITION):
            self.assertEqual(f(a), 100)

    def test_load_method_with_dict_value_set_initially(self):
        if False:
            while True:
                i = 10

        class C:

            def f(self):
                if False:
                    i = 10
                    return i + 15
                return 42
        a = C()

        def f(x):
            if False:
                print('Hello World!')
            return x.f()
        a.f = lambda : 100
        for _ in range(REPETITION):
            self.assertEqual(f(a), 100)

    def test_load_method_with_dict_desc_replace_value(self):
        if False:
            return 10
        hit_count = 0

        class desc:

            def __get__(self, inst, cls):
                if False:
                    print('Hello World!')
                nonlocal hit_count
                hit_count += 1
                return lambda : 42

        class C:
            f = desc()
        a = C()

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return x.f()
        for _ in range(REPETITION):
            self.assertEqual(f(a), 42)
        self.assertEqual(hit_count, REPETITION)
        a.__dict__['f'] = lambda : 100
        for _ in range(REPETITION):
            self.assertEqual(f(a), 100)
        self.assertEqual(hit_count, REPETITION)

    def test_load_method_with_dict_desc_initial_value(self):
        if False:
            i = 10
            return i + 15
        hit_count = 0

        class desc:

            def __get__(self, inst, cls):
                if False:
                    while True:
                        i = 10
                nonlocal hit_count
                hit_count += 1
                return lambda : 42

        class C:

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.f = lambda : 100
            f = desc()
        a = C()

        def f(x):
            if False:
                return 10
            return x.f()
        for _ in range(REPETITION):
            self.assertEqual(f(a), 100)
        self.assertEqual(hit_count, 0)

    def test_load_method_non_desc_with_dict(self):
        if False:
            print('Hello World!')

        class callable:

            def __call__(self, *args):
                if False:
                    print('Hello World!')
                return 42

        class C:
            f = callable()
        a = C()

        def f(x):
            if False:
                return 10
            return x.f()
        for _ in range(REPETITION):
            self.assertEqual(f(a), 42)

    def test_load_method_descr_builtin(self):
        if False:
            while True:
                i = 10
        'INVOKE_METHOD on a descriptor w/o a dictionary'
        x = 42

        def f(x):
            if False:
                print('Hello World!')
            try:
                return x.imag()
            except Exception as e:
                return type(e)
        for _ in range(REPETITION):
            self.assertEqual(f(x), TypeError)

    def test_descr_modifies_type(self):
        if False:
            print('Hello World!')
        for x in range(REPETITION):
            mutating = False

            class C:

                @property
                def f(self):
                    if False:
                        while True:
                            i = 10
                    if mutating:
                        C.f = lambda self: 42
                    return lambda : 100
            a = C()
            d = {}
            exec('def f(x): return x.f()', d)
            f = d['f']
            for i in range(REPETITION):
                if i == x:
                    mutating = True
                if i <= x:
                    self.assertEqual(f(a), 100)
                else:
                    self.assertEqual(f(a), 42)

    def test_polymorphic_method(self):
        if False:
            i = 10
            return i + 15
        outer = self

        class C:

            def f(self):
                if False:
                    print('Hello World!')
                outer.assertEqual(type(self).__name__, 'C')
                return 'C'

        class D:

            def f(self):
                if False:
                    while True:
                        i = 10
                outer.assertEqual(type(self).__name__, 'D')
                return 'D'

        def f(x):
            if False:
                while True:
                    i = 10
            return x.f()
        c = C()
        d = D()
        for i in range(REPETITION):
            self.assertEqual(f(c), 'C')
            self.assertEqual(f(d), 'D')

    def test_polymorphic_exhaust_cache(self):
        if False:
            return 10
        outer = self

        class C:

            def f(self):
                if False:
                    for i in range(10):
                        print('nop')
                outer.assertEqual(type(self).__name__, 'C')
                return 'C'

        def f(x):
            if False:
                while True:
                    i = 10
            return x.f()
        c = C()
        for i in range(REPETITION):
            self.assertEqual(f(c), 'C')
        l = []
        for i in range(500):

            class X:
                x = i

                def f(self):
                    if False:
                        return 10
                    return self.x
            self.assertEqual(f(X()), i)
        for i in range(REPETITION):
            self.assertEqual(f(c), 'C')

    def test_polymorphic_method_mutating(self):
        if False:
            return 10
        outer = self

        class C:
            name = 42

            def f(self):
                if False:
                    return 10
                outer.assertEqual(type(self).__name__, 'C')
                return C.name

        class D:

            def f(self):
                if False:
                    i = 10
                    return i + 15
                outer.assertEqual(type(self).__name__, 'D')
                return 'D'

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return x.f()
        c = C()
        d = D()
        for i in range(REPETITION):
            name = c.name
            self.assertEqual(f(c), name)
            C.name += 1
            self.assertEqual(f(d), 'D')

    def test_polymorphic_method_no_dict(self):
        if False:
            i = 10
            return i + 15
        outer = self

        class C:
            __slots__ = ()

            def f(self):
                if False:
                    print('Hello World!')
                outer.assertEqual(type(self).__name__, 'C')
                return 'C'

        class D:
            __slots__ = ()

            def f(self):
                if False:
                    i = 10
                    return i + 15
                outer.assertEqual(type(self).__name__, 'D')
                return 'D'

        def f(x):
            if False:
                print('Hello World!')
            return x.f()
        c = C()
        d = D()
        for i in range(REPETITION):
            self.assertEqual(f(c), 'C')
            self.assertEqual(f(d), 'D')

    def test_polymorphic_method_mutating_no_dict(self):
        if False:
            i = 10
            return i + 15
        outer = self

        class C:
            __slots__ = ()
            name = 42

            def f(self):
                if False:
                    while True:
                        i = 10
                outer.assertEqual(type(self).__name__, 'C')
                return C.name

        class D:
            __slots__ = ()

            def f(self):
                if False:
                    return 10
                outer.assertEqual(type(self).__name__, 'D')
                return 'D'

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return x.f()
        c = C()
        d = D()
        for i in range(REPETITION):
            name = c.name
            self.assertEqual(f(c), name)
            C.name += 1
            self.assertEqual(f(d), 'D')

    def test_polymorphic_method_mixed_dict(self):
        if False:
            while True:
                i = 10
        outer = self

        class C:
            __slots__ = ()

            def f(self):
                if False:
                    return 10
                outer.assertEqual(type(self).__name__, 'C')
                return 'C'

        class D:

            def f(self):
                if False:
                    while True:
                        i = 10
                outer.assertEqual(type(self).__name__, 'D')
                return 'D'

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return x.f()
        c = C()
        d = D()
        for i in range(REPETITION):
            self.assertEqual(f(c), 'C')
            self.assertEqual(f(d), 'D')

    def test_polymorphic_method_mutating_mixed_dict(self):
        if False:
            i = 10
            return i + 15
        outer = self

        class C:
            __slots__ = ()
            name = 42

            def f(self):
                if False:
                    i = 10
                    return i + 15
                outer.assertEqual(type(self).__name__, 'C')
                return C.name

        class D:

            def f(self):
                if False:
                    print('Hello World!')
                outer.assertEqual(type(self).__name__, 'D')
                return 'D'

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return x.f()
        c = C()
        d = D()
        for i in range(REPETITION):
            name = c.name
            self.assertEqual(f(c), name)
            C.name += 1
            self.assertEqual(f(d), 'D')

    def test_invoke_method_inst_only_split_dict(self):
        if False:
            print('Hello World!')

        class C:
            pass
        a = C()
        a.f = lambda : 42

        def f(x):
            if False:
                return 10
            return x.f()
        for i in range(REPETITION):
            self.assertEqual(f(a), 42)
        del a.f
        with self.assertRaises(AttributeError):
            f(a)

    def test_invoke_method_inst_only(self):
        if False:
            print('Hello World!')

        class C:
            pass
        a = C()
        a.foo = 42
        b = C()
        b.bar = 42
        b.f = lambda : 42

        def f(x):
            if False:
                while True:
                    i = 10
            return x.f()
        for i in range(REPETITION):
            self.assertEqual(f(b), 42)
        del b.f
        with self.assertRaises(AttributeError):
            f(b)

    def test_instance_dir_mutates_with_custom_hash(self):
        if False:
            while True:
                i = 10

        class C:

            def f(self):
                if False:
                    for i in range(10):
                        print('nop')
                return 42

        def f(x):
            if False:
                while True:
                    i = 10
            return x.f()
        x = C()
        for i in range(REPETITION):
            self.assertEqual(f(x), 42)

        class mystr(str):

            def __eq__(self, other):
                if False:
                    i = 10
                    return i + 15
                del C.f
                return super().__eq__(other)

            def __hash__(self):
                if False:
                    while True:
                        i = 10
                return str.__hash__(self)
        x.__dict__[mystr('f')] = lambda : 100
        self.assertEqual(f(x), 100)

    def test_instance_dir_mutates_with_custom_hash_different_attr(self):
        if False:
            print('Hello World!')

        class C:

            def f(self):
                if False:
                    print('Hello World!')
                return 42

        def f(x):
            if False:
                i = 10
                return i + 15
            return x.f()
        x = C()
        for i in range(REPETITION):
            self.assertEqual(f(x), 42)

        class mystr(str):

            def __eq__(self, other):
                if False:
                    for i in range(10):
                        print('nop')
                del C.f
                return super().__eq__(self, other)

            def __hash__(self):
                if False:
                    while True:
                        i = 10
                return hash('f')
        x.__dict__[mystr('g')] = lambda : 100
        self.assertEqual(f(x), 42)

    def test_instance_dir_mutates_with_custom_hash_descr(self):
        if False:
            i = 10
            return i + 15

        class descr:

            def __get__(self, inst, ctx):
                if False:
                    return 10
                return lambda : 42

        class C:
            f = descr()

        def f(x):
            if False:
                i = 10
                return i + 15
            return x.f()
        x = C()
        for i in range(REPETITION):
            self.assertEqual(f(x), 42)

        class mystr(str):

            def __eq__(self, other):
                if False:
                    while True:
                        i = 10
                del C.f
                return super().__eq__(other)

            def __hash__(self):
                if False:
                    print('Hello World!')
                return str.__hash__(self)
        x.__dict__[mystr('f')] = lambda : 100
        self.assertEqual(f(x), 100)

    def test_instance_dir_mutates_with_custom_hash_different_attr_descr(self):
        if False:
            while True:
                i = 10

        class descr:

            def __get__(self, inst, ctx):
                if False:
                    print('Hello World!')
                return lambda : 42

        class C:
            f = descr()

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return x.f()
        x = C()
        for i in range(REPETITION):
            self.assertEqual(f(x), 42)

        class mystr(str):

            def __eq__(self, other):
                if False:
                    return 10
                del C.f
                return super().__eq__(self, other)

            def __hash__(self):
                if False:
                    i = 10
                    return i + 15
                return hash('f')
        x.__dict__[mystr('g')] = lambda : 100
        self.assertEqual(f(x), 42)

    def test_loadmethod_cachelines(self):
        if False:
            for i in range(10):
                print('nop')

        class C:

            def f(self):
                if False:
                    while True:
                        i = 10
                return 42

        def f1(x):
            if False:
                return 10
            return x.f()

        def f2(x):
            if False:
                for i in range(10):
                    print('nop')
            return x.f()
        x = C()
        for i in range(REPETITION):
            self.assertEqual(f1(x), 42)

        class descr:

            def __get__(self, inst, ctx):
                if False:
                    i = 10
                    return i + 15
                return lambda : 100
        C.f = descr()
        for i in range(REPETITION):
            self.assertEqual(f2(x), 100)
        self.assertEqual(f1(x), 100)

    def test_exhaust_invalidation(self):
        if False:
            return 10

        class C:
            pass

        def f(x):
            if False:
                i = 10
                return i + 15
            return x.f()

        def g(x):
            if False:
                i = 10
                return i + 15
            return x.f()
        x = C()
        for i in range(2000):

            def maker(i):
                if False:
                    return 10
                return lambda self: i
            C.f = maker(i)
            self.assertEqual(f(x), i)
            self.assertEqual(g(x), i)

    def test_type_call(self):
        if False:
            return 10

        class C:

            def f(self):
                if False:
                    while True:
                        i = 10
                return 42
        a = C()

        def f(x, inst):
            if False:
                while True:
                    i = 10
            return x.f(inst)
        for _ in range(REPETITION):
            self.assertEqual(f(C, a), 42)

    def test_type_call_metatype(self):
        if False:
            while True:
                i = 10

        class MC(type):
            pass

        class C(metaclass=MC):

            def f(self):
                if False:
                    i = 10
                    return i + 15
                return 42
        a = C()

        def f(x, inst):
            if False:
                for i in range(10):
                    print('nop')
            return x.f(inst)
        for _ in range(REPETITION):
            self.assertEqual(f(C, a), 42)

    def test_type_call_metatype_add_getattr(self):
        if False:
            while True:
                i = 10

        class MC(type):
            pass

        class C(metaclass=MC):

            def f(self):
                if False:
                    i = 10
                    return i + 15
                return 42
        a = C()

        def f(x, inst):
            if False:
                print('Hello World!')
            return x.f(inst)
        for _ in range(REPETITION):
            self.assertEqual(f(C, a), 42)
        MC.__getattribute__ = lambda self, name: lambda self: 100
        self.assertEqual(f(C, a), 100)

    def test_metatype_getattr(self):
        if False:
            while True:
                i = 10

        class MC(type):

            def __getattribute__(self, name):
                if False:
                    while True:
                        i = 10
                return 100

        class C(metaclass=MC):
            x = 42

        def f(inst):
            if False:
                for i in range(10):
                    print('nop')
            return inst.x
        for _ in range(REPETITION):
            self.assertEqual(f(C), 100)

    def test_metatype_add_getattr(self):
        if False:
            print('Hello World!')

        class MC(type):
            pass

        class C(metaclass=MC):
            x = 42

        def f(inst):
            if False:
                for i in range(10):
                    print('nop')
            return inst.x
        for _ in range(REPETITION):
            self.assertEqual(f(C), 42)
        MC.__getattribute__ = lambda self, name: 100
        self.assertEqual(f(C), 100)

    def test_metatype_add_getattr_no_leak(self):
        if False:
            while True:
                i = 10

        class MC(type):
            pass

        class C(metaclass=MC):
            x = 42
        wr = weakref.ref(C)

        def f(inst):
            if False:
                while True:
                    i = 10
            return inst.x
        for _ in range(REPETITION):
            self.assertEqual(f(C), 42)
        import gc
        del C
        gc.collect()
        self.assertEqual(wr(), None)

    def test_metatype_change(self):
        if False:
            for i in range(10):
                print('nop')

        class MC(type):
            pass

        class MC2(type):

            def __getattribute__(self, name):
                if False:
                    while True:
                        i = 10
                return 100

        class C(metaclass=MC):
            x = 42

        def f(inst):
            if False:
                i = 10
                return i + 15
            return inst.x
        for _ in range(REPETITION):
            self.assertEqual(f(C), 42)
        C.__class__ = MC2
        self.assertEqual(f(C), 100)

    def test_type_call_invalidate(self):
        if False:
            return 10

        class C:

            def f(self):
                if False:
                    return 10
                return 42
        a = C()

        def f(x, inst):
            if False:
                i = 10
                return i + 15
            return x.f(inst)
        for _ in range(REPETITION):
            self.assertEqual(f(C, a), 42)
        C.f = lambda self: 100
        self.assertEqual(f(C, a), 100)

    def test_type_call_descr(self):
        if False:
            return 10
        test = self

        class descr:

            def __get__(self, inst, ctx):
                if False:
                    while True:
                        i = 10
                test.assertEqual(inst, None)
                test.assertEqual(ctx, C)
                return lambda : 42

        class C:
            f = descr()
        a = C()

        def f(x):
            if False:
                print('Hello World!')
            return x.f()
        for _ in range(REPETITION):
            self.assertEqual(f(C), 42)

    def test_type_call_non_descr(self):
        if False:
            print('Hello World!')
        test = self

        class descr:

            def __call__(self):
                if False:
                    while True:
                        i = 10
                return 42

        class C:
            f = descr()
        a = C()

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return x.f()
        for _ in range(REPETITION):
            self.assertEqual(f(C), 42)

    def test_load_slot(self):
        if False:
            print('Hello World!')

        class C:
            __slots__ = 'value'

            def __init__(self, value):
                if False:
                    for i in range(10):
                        print('nop')
                self.value = value

        def f(x):
            if False:
                return 10
            return x.value
        for i in range(REPETITION):
            x = C(i)
            self.assertEqual(f(x), i)

    def test_load_slot_cache_miss(self):
        if False:
            return 10

        class C:
            __slots__ = 'value'

            def __init__(self, value):
                if False:
                    i = 10
                    return i + 15
                self.value = value

        def f(x):
            if False:
                return 10
            return x.value
        for i in range(REPETITION):
            x = C(i)
            self.assertEqual(f(x), i)

        class D:
            value = 100
        self.assertEqual(f(D), 100)

    def test_load_slot_unset(self):
        if False:
            i = 10
            return i + 15

        class C:
            __slots__ = 'value'

        def f(x):
            if False:
                i = 10
                return i + 15
            return x.value
        for i in range(REPETITION):
            x = C()
            x.value = i
            self.assertEqual(f(x), i)
        with self.assertRaises(AttributeError):
            f(C())

    def test_store_slot(self):
        if False:
            for i in range(10):
                print('nop')

        class C:
            __slots__ = 'value'

        def f(x, i):
            if False:
                return 10
            x.value = i
        for i in range(REPETITION):
            x = C()
            f(x, i)
            self.assertEqual(x.value, i)

    def test_store_slot_cache_miss(self):
        if False:
            for i in range(10):
                print('nop')

        class C:
            __slots__ = 'value'

        def f(x, i):
            if False:
                return 10
            x.value = i
        for i in range(REPETITION):
            x = C()
            f(x, i)
            self.assertEqual(x.value, i)

        class D:
            pass
        x = D()
        f(x, 100)
        self.assertEqual(x.value, 100)

    @skipIf(cached_property is None, 'no cached_property')
    def test_cached_property(self):
        if False:
            print('Hello World!')

        class C:

            def __init__(self, value=42):
                if False:
                    for i in range(10):
                        print('nop')
                self.value = value
                self.calls = 0

            @cached_property
            def f(self):
                if False:
                    print('Hello World!')
                self.calls += 1
                return self.value

        def f(x):
            if False:
                print('Hello World!')
            return x.f
        for i in range(REPETITION):
            inst = C(i)
            self.assertEqual(f(inst), i)
        inst = C(42)
        v = inst.f
        for _ in range(REPETITION):
            self.assertEqual(f(inst), 42)
        x = C(42)
        f(x)
        f(x)
        self.assertEqual(x.calls, 1)

    @skipIf(cached_property is None, 'no cached_property')
    def test_cached_property_raises(self):
        if False:
            return 10

        class C:

            def __init__(self, raises=False):
                if False:
                    return 10
                self.raises = raises

            @cached_property
            def f(self):
                if False:
                    print('Hello World!')
                if self.raises:
                    raise ShadowError()
                return 42

        def f(x):
            if False:
                while True:
                    i = 10
            return x.f
        for _ in range(REPETITION):
            inst = C()
            self.assertEqual(f(inst), 42)
        inst = C(True)
        with self.assertRaises(ShadowError):
            f(inst)

    @skipIf(cached_property is None, 'no cached_property')
    def test_cached_property_slots(self):
        if False:
            i = 10
            return i + 15

        class C:
            __slots__ = ('f', 'value', 'calls')

            def __init__(self, value=42):
                if False:
                    return 10
                self.value = value
                self.calls = 0

        def f(self):
            if False:
                i = 10
                return i + 15
            self.calls += 1
            return self.value
        C.f = cached_property(f, C.f)

        def f(x):
            if False:
                i = 10
                return i + 15
            return x.f
        inst = C(42)
        v = inst.f
        for _ in range(REPETITION):
            self.assertEqual(f(inst), 42)
        x = C(42)
        f(x)
        f(x)
        self.assertEqual(x.calls, 1)

    @skipIf(cached_property is None, 'no cached_property')
    def test_cached_property_slots_raises(self):
        if False:
            while True:
                i = 10

        class C:
            __slots__ = ('raises', 'f')

            def __init__(self, raises=False):
                if False:
                    return 10
                self.raises = raises

        def f(self):
            if False:
                for i in range(10):
                    print('nop')
            if self.raises:
                raise ShadowError()
            return 42
        C.f = cached_property(f, C.f)

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return x.f
        for _ in range(REPETITION):
            inst = C()
            self.assertEqual(f(inst), 42)
        inst = C(True)
        with self.assertRaises(ShadowError):
            f(inst)

    def test_module_attr(self):
        if False:
            i = 10
            return i + 15
        mod = type(sys)('foo')
        mod.x = 42

        def f(x):
            if False:
                while True:
                    i = 10
            return x.x
        for _ in range(REPETITION):
            self.assertEqual(f(mod), 42)
        mod.x = 100
        for _ in range(REPETITION):
            self.assertEqual(f(mod), 100)

    def test_module_descr_conflict(self):
        if False:
            return 10
        mod = type(sys)('foo')
        func = mod.__dir__

        def f(x):
            if False:
                i = 10
                return i + 15
            return x.__dir__
        for _ in range(REPETITION):
            self.assertEqual(f(mod), func)
        mod.__dir__ = 100
        self.assertEqual(f(mod), 100)

    def test_multi_cache(self):
        if False:
            print('Hello World!')

        class C:
            x = 1

        class D:
            x = 2

        def f(a, c):
            if False:
                while True:
                    i = 10
            if c:
                return a.x
            else:
                return a.x
        c = C()
        d = D()
        for _i in range(REPETITION):
            self.assertEqual(f(c, True), 1)
            self.assertEqual(f(c, False), 1)
        C.x = 3
        self.assertEqual(f(d, True), 2)
        self.assertEqual(f(c, False), 3)

    def test_multi_cache_module(self):
        if False:
            i = 10
            return i + 15
        m1 = type(sys)('m1')
        m1.x = 1
        m1.y = 2
        m2 = type(sys)('m2')
        m2.x = 3
        m2.y = 4

        def f(a, c):
            if False:
                i = 10
                return i + 15
            if c == 1:
                return a.x
            elif c == 2:
                return a.y
            elif c == 3:
                return a.x
        for _i in range(REPETITION):
            self.assertEqual(f(m1, 1), 1)
            self.assertEqual(f(m1, 2), 2)
            self.assertEqual(f(m1, 3), 1)
        m1.x = 5
        self.assertEqual(f(m2, 2), 4)
        self.assertEqual(f(m1, 1), 5)

    def test_module_method(self):
        if False:
            return 10
        mymod = type(sys)('foo')

        def mod_meth():
            if False:
                i = 10
                return i + 15
            return 42
        mymod.mod_meth = mod_meth

        def f(x):
            if False:
                print('Hello World!')
            return x.mod_meth()
        for _i in range(REPETITION):
            self.assertEqual(f(mymod), 42)

    def test_module_method_invalidate(self):
        if False:
            for i in range(10):
                print('nop')
        mymod = type(sys)('foo')

        def mod_meth():
            if False:
                while True:
                    i = 10
            return 42
        mymod.mod_meth = mod_meth

        def f(x):
            if False:
                while True:
                    i = 10
            return x.mod_meth()
        for _i in range(REPETITION):
            self.assertEqual(f(mymod), 42)
        for _i in range(REPETITION):
            mymod.mod_meth = lambda : _i
            self.assertEqual(f(mymod), _i)

    def test_module_method_miss(self):
        if False:
            i = 10
            return i + 15
        mymod = type(sys)('foo')

        def mod_meth():
            if False:
                for i in range(10):
                    print('nop')
            return 42
        mymod.mod_meth = mod_meth

        def f(x):
            if False:
                return 10
            return x.mod_meth()

        class C:

            def mod_meth(self):
                if False:
                    return 10
                return 'abc'
        for _i in range(REPETITION):
            self.assertEqual(f(mymod), 42)
        self.assertEqual(f(C()), 'abc')

    def test_module_getattr(self):
        if False:
            for i in range(10):
                print('nop')
        mymod = type(sys)('foo')

        def mod_getattr(name):
            if False:
                i = 10
                return i + 15
            if name == 'attr':
                return 'abc'
            raise AttributeError(name)
        mymod.attr = 42
        mymod.__getattr__ = mod_getattr

        def f(x):
            if False:
                print('Hello World!')
            return x.attr
        for _i in range(REPETITION):
            self.assertEqual(f(mymod), 42)
        del mymod.attr
        self.assertEqual(f(mymod), 'abc')

    def test_module_method_getattr(self):
        if False:
            print('Hello World!')
        mymod = type(sys)('foo')

        def mod_meth():
            if False:
                while True:
                    i = 10
            return 42

        def mod_getattr(name):
            if False:
                i = 10
                return i + 15
            if name == 'mod_meth':
                return lambda : 'abc'
            raise AttributeError(name)
        mymod.mod_meth = mod_meth
        mymod.__getattr__ = mod_getattr

        def f(x):
            if False:
                i = 10
                return i + 15
            return x.mod_meth()
        for _i in range(REPETITION):
            self.assertEqual(f(mymod), 42)
        del mymod.mod_meth
        self.assertEqual(f(mymod), 'abc')

    def test_type_error_every_access(self):
        if False:
            return 10
        runcount = 0

        class Raises:

            def __get__(self, instance, owner):
                if False:
                    while True:
                        i = 10
                nonlocal runcount
                runcount += 1
                raise NotImplementedError

        class C:
            prop = Raises()

        def f(c):
            if False:
                return 10
            try:
                return c.prop
            except NotImplementedError:
                return 42
        for i in range(200):
            runcount = 0
            self.assertEqual(f(C), 42)
            self.assertEqual(runcount, 1)

    def test_module_error_every_access(self):
        if False:
            print('Hello World!')
        m = type(sys)('test')
        runcount = 0

        class mystr(str):

            def __eq__(self, other):
                if False:
                    while True:
                        i = 10
                nonlocal runcount
                runcount += 1
                raise NotImplementedError

            def __hash__(self):
                if False:
                    while True:
                        i = 10
                return str.__hash__(self)
        m.__dict__[mystr('foo')] = 42

        def f(c):
            if False:
                i = 10
                return i + 15
            try:
                return c.foo
            except AttributeError:
                return 42
        for i in range(200):
            runcount = 0
            self.assertRaises(NotImplementedError, f, m)
            self.assertEqual(runcount, 1)

    def test_module_error_getattr(self):
        if False:
            for i in range(10):
                print('nop')
        m = type(sys)('test')
        runcount = 0

        class mystr(str):

            def __eq__(self, other):
                if False:
                    return 10
                nonlocal runcount
                runcount += 1
                raise NotImplementedError

            def __hash__(self):
                if False:
                    return 10
                return str.__hash__(self)
        m.__dict__[mystr('foo')] = 100
        m.__getattr__ = lambda *args: 42

        def f(c):
            if False:
                for i in range(10):
                    print('nop')
            return c.foo
        for i in range(200):
            runcount = 0
            self.assertRaises(NotImplementedError, f, m)
            self.assertEqual(runcount, 1)

    def test_dict_subscr(self):
        if False:
            for i in range(10):
                print('nop')
        key = (1, 2)
        value = 1
        d = {key: value}

        def f():
            if False:
                while True:
                    i = 10
            return d[key]
        for __ in range(REPETITION):
            self.assertEqual(f(), value)

    def test_list_subscr(self):
        if False:
            for i in range(10):
                print('nop')
        d = list(range(5))

        def f(i):
            if False:
                while True:
                    i = 10
            return d[i]
        for __ in range(REPETITION):
            for i in range(5):
                self.assertEqual(f(i), i)

    def test_tuple_subscr(self):
        if False:
            for i in range(10):
                print('nop')
        t = (1, 2, 3, 4, 5)
        ans = (2, 3)

        def f():
            if False:
                return 10
            return t[1:3]
        for __ in range(REPETITION):
            self.assertEqual(f(), ans)

    def test_dict_str_key(self):
        if False:
            while True:
                i = 10
        key = 'mykey'
        value = 1
        d = {key: value}

        def f():
            if False:
                for i in range(10):
                    print('nop')
            return d[key]
        for __ in range(REPETITION):
            self.assertEqual(f(), value)

    def test_tuple_int_const_key(self):
        if False:
            print('Hello World!')
        t = (1, 2, 3)

        def f():
            if False:
                return 10
            return t[0]
        for __ in range(REPETITION):
            self.assertEqual(f(), 1)

    def test_dict_subscr_keyerror(self):
        if False:
            for i in range(10):
                print('nop')
        key = (1, 2)
        value = 1
        d = {key: value}
        wrong_key = (1, 3)

        def f(k):
            if False:
                print('Hello World!')
            return d[k]
        for __ in range(REPETITION):
            self.assertEqual(f(key), value)
        for __ in range(REPETITION):
            self.assertRaises(KeyError, f, wrong_key)

    def test_dict_subscr_to_non_dict(self):
        if False:
            print('Hello World!')
        key = 1
        value = 1
        d = {key: value}
        t = (1, 2, 3, 4)
        value2 = t[key]

        def f(d, k):
            if False:
                return 10
            return d[k]
        for __ in range(REPETITION):
            self.assertEqual(f(d, key), value)
        for __ in range(REPETITION):
            self.assertEqual(f(t, key), value2)

    def test_list_subscr_to_non_list(self):
        if False:
            i = 10
            return i + 15
        l = [1, 2, 3, 4]
        t = (5, 6, 7)

        def f(d, k):
            if False:
                return 10
            return d[k]
        for __ in range(REPETITION):
            self.assertEqual(f(l, 0), 1)
        for __ in range(REPETITION):
            self.assertEqual(f(t, 1), 6)

    def test_tuple_subscr_indexerror(self):
        if False:
            i = 10
            return i + 15
        t = (1, 2, 3, 4, 5)

        def f(i):
            if False:
                return 10
            return t[i]
        for __ in range(REPETITION):
            self.assertRaises(IndexError, f, 6)

    def test_tuple_subscr_to_non_tuple(self):
        if False:
            return 10
        l = [1, 2, 3, 4]
        t = (5, 6, 7)

        def f(d, k):
            if False:
                return 10
            return d[k]
        for __ in range(REPETITION):
            self.assertEqual(f(t, 0), 5)
        for __ in range(REPETITION):
            self.assertEqual(f(l, 1), 2)

    def test_dict_str_key_to_nonstr_key(self):
        if False:
            for i in range(10):
                print('nop')
        key = 'mykey'
        value = 1
        key2 = 3
        value2 = 4
        d = {key: value, key2: value2}

        def f(k):
            if False:
                i = 10
                return i + 15
            return d[k]
        for __ in range(REPETITION):
            self.assertEqual(f(key), value)
        for __ in range(REPETITION):
            self.assertEqual(f(key2), value2)

    def test_dict_str_key_to_non_dict(self):
        if False:
            for i in range(10):
                print('nop')
        key = 'mykey'
        value = 1
        d = {key: value}
        l = [1, 2, 3]

        def f(c, k):
            if False:
                i = 10
                return i + 15
            return c[k]
        for __ in range(REPETITION):
            self.assertEqual(f(d, key), value)
        for __ in range(REPETITION):
            self.assertEqual(f(l, 1), 2)

    def test_tuple_int_const_key_two_tuples(self):
        if False:
            i = 10
            return i + 15
        t = (1, 2, 3)
        t2 = (3, 4, 5)

        def f(t):
            if False:
                i = 10
                return i + 15
            return t[0]
        for __ in range(REPETITION):
            self.assertEqual(f(t), 1)
        for __ in range(REPETITION):
            self.assertEqual(f(t2), 3)

    def test_tuple_int_const_key_indexerror(self):
        if False:
            for i in range(10):
                print('nop')
        t = (0, 1, 2, 3, 4, 5, 6)
        t2 = (0, 1, 2)

        def g(t):
            if False:
                print('Hello World!')
            return t[6]
        for __ in range(REPETITION):
            self.assertEqual(g(t), 6)
        for __ in range(REPETITION):
            self.assertRaises(IndexError, g, t2)

    def test_tuple_int_const_key_too_long(self):
        if False:
            print('Hello World!')
        t = (1, 2, 3)

        def g():
            if False:
                while True:
                    i = 10
            return t[1267650600228229401496703205376]
        for __ in range(REPETITION):
            self.assertRaises(IndexError, g)

    def test_tuple_int_const_negative_key(self):
        if False:
            i = 10
            return i + 15
        t1 = (1, 2, 3)
        t2 = (-1, -2, -3, -4, -5)

        def f(t):
            if False:
                print('Hello World!')
            return t[-1]
        for __ in range(REPETITION):
            self.assertEqual(f(t1), 3)
        for __ in range(REPETITION):
            self.assertEqual(f(t2), -5)

    def test_tuple_const_int_not_tuple(self):
        if False:
            print('Hello World!')
        t = (1, 2, 3)
        d = {0: 'x'}

        def f(t):
            if False:
                return 10
            return t[0]
        for __ in range(REPETITION):
            self.assertEqual(f(t), 1)
        for __ in range(REPETITION):
            self.assertEqual(f(d), 'x')

    def test_polymorphic(self):
        if False:
            i = 10
            return i + 15

        class C:

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.value = 42

        class D:

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.value = 100

        def f(x):
            if False:
                i = 10
                return i + 15
            return x.value
        c = C()
        d = D()
        for _ in range(REPETITION):
            self.assertEqual(f(c), 42)
        self.assertEqual(f(d), 100)
        self.assertEqual(f(d), 100)
        self.assertEqual(f(d), 100)
        self.assertEqual(f(c), 42)

    def test_polymorphic_type_mutation(self):
        if False:
            while True:
                i = 10

        class C:
            value = 42

        class D:

            def __init__(self):
                if False:
                    return 10
                self.value = 100
        c = C()
        d = D()

        def f(x):
            if False:
                return 10
            return x.value

        def poly(x):
            if False:
                print('Hello World!')
            return x.value
        for _ in range(REPETITION):
            self.assertEqual(f(c), 42)
            self.assertEqual(poly(c), 42)
        poly(d)
        for i in range(2000):
            C.x = i
            f(c)
        self.assertEqual(poly(c), 42)
        C.value = 100
        self.assertEqual(poly(c), 100)

    def test_globals_remove_promote_to_builtin(self):
        if False:
            for i in range(10):
                print('nop')
        global filter
        orig_filter = filter
        filter = 42
        try:

            def f():
                if False:
                    return 10
                return filter
            for _ in range(REPETITION):
                self.assertEqual(f(), 42)
        finally:
            del filter
        self.assertIs(f(), orig_filter)
        try:
            builtins.filter = 43
            self.assertEqual(f(), 43)
        finally:
            builtins.filter = orig_filter

    def test_loadmethod_meta_getattr(self):
        if False:
            while True:
                i = 10

        class MC(type):

            def __getattribute__(self, name):
                if False:
                    print('Hello World!')
                return lambda x: x + 1

        class C(metaclass=MC):

            @staticmethod
            def f(x):
                if False:
                    print('Hello World!')
                return x

        def f(i):
            if False:
                return 10
            return C.f(i)
        for i in range(REPETITION):
            self.assertEqual(f(i), i + 1)

    def test_loadmethod_setattr(self):
        if False:
            return 10

        class C:
            pass

        def f(a, i):
            if False:
                i = 10
                return i + 15
            object.__setattr__(a, 'foo', i)
        a = C()
        for i in range(REPETITION):
            f(a, i)
            self.assertEqual(a.foo, i)

    def test_loadattr_setattr(self):
        if False:
            while True:
                i = 10

        class C:
            pass

        def f(a, i):
            if False:
                while True:
                    i = 10
            z = object.__setattr__
            z(a, 'foo', i)
        a = C()
        for i in range(REPETITION):
            f(a, i)
            self.assertEqual(a.foo, i)

    def test_module_invalidate(self):
        if False:
            print('Hello World!')
        mod = type(sys)('foo')
        mod.foo = 42

        def f1(x):
            if False:
                for i in range(10):
                    print('nop')
            return x.foo

        def f2(x):
            if False:
                return 10
            return x.foo
        for i in range(REPETITION):
            self.assertEqual(f1(mod), 42)
        mod.foo = 100
        for i in range(REPETITION):
            self.assertEqual(f2(mod), 100)
        del mod
        mod = type(sys)('foo')
        mod.foo = 300
        self.assertEqual(f1(mod), 300)

    def test_object_field(self):
        if False:
            i = 10
            return i + 15

        class C(OSError):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.filename = 'abc'
        for i in range(REPETITION):
            x = C()
            self.assertEqual(x.__dict__, {})
            self.assertEqual(x.filename, 'abc')

    def test_readonly_field(self):
        if False:
            print('Hello World!')

        class C:
            pass

        def f(x):
            if False:
                i = 10
                return i + 15
            x.start = 1
        for i in range(REPETITION):
            f(C())
        with self.assertRaises(AttributeError):
            f(range(5))

    @skipIf(StrictModule is None, 'no StrictModule')
    def test_strictmodule(self):
        if False:
            print('Hello World!')
        mod = type(sys)('foo')
        mod.x = 100
        d = mod.__dict__
        m = StrictModule(d, False)

        def f():
            if False:
                for i in range(10):
                    print('nop')
            return m.x
        for _i in range(REPETITION):
            self.assertEqual(f(), 100)
        d['x'] = 200
        for _i in range(REPETITION):
            self.assertEqual(f(), 200)

    @skipIf(StrictModule is None, 'no StrictModule')
    def test_strictmodule_descr_conflict(self):
        if False:
            i = 10
            return i + 15
        mod = type(sys)('foo')
        d = mod.__dict__
        m = StrictModule(d, False)
        func = m.__dir__

        def f(x):
            if False:
                return 10
            return x.__dir__
        for i in range(REPETITION):
            self.assertEqual(f(m), func)
        d['__dir__'] = 100
        self.assertEqual(f(m), 100)

    @skipIf(StrictModule is None, 'no StrictModule')
    def test_strictmodule_descr_conflict_with_patch(self):
        if False:
            i = 10
            return i + 15
        mod = type(sys)('foo')
        d = mod.__dict__
        m = StrictModule(d, True)
        func = m.__dir__

        def f(x):
            if False:
                while True:
                    i = 10
            return x.__dir__
        for i in range(REPETITION):
            self.assertEqual(f(m), func)
        strict_module_patch(m, '__dir__', 100)
        self.assertEqual(f(m), 100)

    @skipIf(StrictModule is None, 'no StrictModule')
    def test_strictmodule_method(self):
        if False:
            print('Hello World!')
        mod = type(sys)('foo')

        def mod_meth():
            if False:
                while True:
                    i = 10
            return 42
        mod.mod_meth = mod_meth
        d = mod.__dict__
        m = StrictModule(d, False)

        def f(x):
            if False:
                i = 10
                return i + 15
            return x.mod_meth()
        for _i in range(REPETITION):
            self.assertEqual(f(m), 42)

    @skipIf(StrictModule is None, 'no StrictModule')
    def test_strcitmodule_method_invalidate(self):
        if False:
            return 10
        mod = type(sys)('foo')

        def mod_meth():
            if False:
                return 10
            return 42
        mod.mod_meth = mod_meth
        d = mod.__dict__
        m = StrictModule(d, False)

        def f(x):
            if False:
                i = 10
                return i + 15
            return x.mod_meth()
        for _i in range(REPETITION):
            self.assertEqual(f(m), 42)
        for _i in range(REPETITION):
            d['mod_meth'] = lambda : _i
            self.assertEqual(f(m), _i)

    @skipIf(StrictModule is None, 'no StrictModule')
    def test_strictmodule_method_miss(self):
        if False:
            i = 10
            return i + 15
        mod = type(sys)('foo')

        def mod_meth():
            if False:
                for i in range(10):
                    print('nop')
            return 42
        mod.mod_meth = mod_meth
        d = mod.__dict__
        m = StrictModule(d, False)

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return x.mod_meth()

        class C:

            def mod_meth(self):
                if False:
                    print('Hello World!')
                return 'abc'
        for _i in range(REPETITION):
            self.assertEqual(f(m), 42)
        self.assertEqual(f(C()), 'abc')

    def test_loadattr_descr_changed_to_data_descr(self):
        if False:
            return 10

        class NonDataDescr:

            def __init__(self):
                if False:
                    return 10
                self.invoked_count = 0

            def __get__(self, obj, typ):
                if False:
                    while True:
                        i = 10
                self.invoked_count += 1
                obj.__dict__['foo'] = 'testing 123'
                return 'testing 123'
        descr = NonDataDescr()

        class TestObj:
            foo = descr

        def get_foo(obj):
            if False:
                while True:
                    i = 10
            return obj.foo
        obj = TestObj()
        self.assertEqual(get_foo(obj), 'testing 123')
        self.assertEqual(descr.invoked_count, 1)
        for _ in range(REPETITION):
            self.assertEqual(get_foo(obj), 'testing 123')
            self.assertEqual(descr.invoked_count, 1)

        def setter(self, obj, val):
            if False:
                return 10
            pass
        descr.__class__.__set__ = setter
        self.assertEqual(get_foo(obj), 'testing 123')
        self.assertEqual(descr.invoked_count, 2)

    def test_reassign_split_dict(self):
        if False:
            while True:
                i = 10

        class Foo:

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                self.attr = 100

        class Bar:

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.a0 = 0
                self.attr = 200

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return x.attr
        obj = Foo()
        for _ in range(REPETITION):
            self.assertEqual(f(obj), 100)
        obj2 = Bar()
        obj.__dict__ = obj2.__dict__
        self.assertEqual(f(obj), 200)

    def test_reassign_class_with_different_split_dict(self):
        if False:
            i = 10
            return i + 15

        class Foo:

            def __init__(self):
                if False:
                    print('Hello World!')
                self.attr = 100

        class Bar:

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                self.a0 = 0
                self.attr = 200

        def f(x):
            if False:
                while True:
                    i = 10
            return x.attr
        obj = Foo()
        for _ in range(REPETITION):
            self.assertEqual(f(obj), 100)
        obj2 = Bar()
        obj2.__class__ = Foo
        self.assertEqual(f(obj2), 200)

    @skipIf(not hasattr('gc', 'is_immortal'), 'no immortalization')
    def test_load_immortal_classmethod(self):
        if False:
            print('Hello World!')
        code = f'if 1:\n            class Foo:\n                @classmethod\n                def identity(cls, x):\n                    return x\n\n            import gc\n            gc.immortalize_heap()\n\n            def f(x):\n                return Foo.identity(x)\n\n            # Prime the cache\n            for _ in range({REPETITION}):\n                f(100)\n\n            print(f(100))\n            '
        (rc, out, err) = skip_ret_code_check_for_leaking_test_in_asan_mode('-c', code)
        self.assertEqual(out.strip(), b'100')

    @skipIf(not hasattr('gc', 'is_immortal'), 'no immortalization')
    def test_load_immortal_staticmethod(self):
        if False:
            for i in range(10):
                print('nop')
        code = f'if 1:\n            class Foo:\n                @staticmethod\n                def identity(x):\n                    return x\n\n            import gc\n            gc.immortalize_heap()\n\n            def f(x):\n                return Foo.identity(x)\n\n            # Prime the cache\n            for _ in range({REPETITION}):\n                f(100)\n\n            print(f(100))\n            '
        (rc, out, err) = skip_ret_code_check_for_leaking_test_in_asan_mode('-c', code)
        self.assertEqual(out.strip(), b'100')

    @skipIf(not hasattr('gc', 'is_immortal'), 'no immortalization')
    def test_load_immortal_wrapper_descr(self):
        if False:
            i = 10
            return i + 15
        code = f"if 1:\n            class Foo:\n                def __repr__(self):\n                    return 12345\n\n            import gc\n            gc.immortalize_heap()\n\n            def f():\n                return str.__repr__('hello')\n\n            # Prime the cache\n            for _ in range({REPETITION}):\n                f()\n\n            print(f())\n            "
        (rc, out, err) = skip_ret_code_check_for_leaking_test_in_asan_mode('-c', code)
        self.assertEqual(out.strip(), b"'hello'")

    @skipIf(not hasattr('gc', 'is_immortal'), 'no immortalization')
    def test_load_immortal_function(self):
        if False:
            return 10
        code = f'if 1:\n            class Oracle:\n                def speak():\n                    return 42\n\n            import gc\n            gc.immortalize_heap()\n\n            def f():\n                return Oracle.speak()\n\n            # Prime the cache\n            for _ in range({REPETITION}):\n                f()\n\n            print(f())\n            '
        (rc, out, err) = skip_ret_code_check_for_leaking_test_in_asan_mode('-c', code)
        self.assertEqual(out.strip(), b'42')

    @skipIf(not hasattr('gc', 'is_immortal'), 'no immortalization')
    def test_load_immortal_method_descriptor(self):
        if False:
            for i in range(10):
                print('nop')
        code = f'if 1:\n            import gc\n            gc.immortalize_heap()\n\n            def f(l):\n                return list.pop(l)\n\n            # Prime the cache\n            for _ in range({REPETITION}):\n                f([42])\n\n            print(f([42]))\n            '
        (rc, out, err) = skip_ret_code_check_for_leaking_test_in_asan_mode('-c', code)
        self.assertEqual(out.strip(), b'42')

    @skipIf(not hasattr('gc', 'is_immortal'), 'no immortalization')
    def test_load_immortal_builtin_function(self):
        if False:
            i = 10
            return i + 15
        code = f'if 1:\n            class Foo:\n                pass\n\n            import gc\n            gc.immortalize_heap()\n\n            def f():\n                return object.__new__(Foo)\n\n            # Prime the cache\n            for _ in range({REPETITION}):\n                f()\n\n            print(isinstance(f(), Foo))\n            '
        (rc, out, err) = skip_ret_code_check_for_leaking_test_in_asan_mode('-c', code)
        self.assertEqual(out.strip(), b'True')

    @skipIf(not hasattr('gc', 'is_immortal'), 'no immortalization')
    def test_load_unshadowed_immortal_method_split_dict(self):
        if False:
            for i in range(10):
                print('nop')
        code = f'if 1:\n            class Oracle:\n                def __init__(self):\n                    self.answer = 42\n\n                def speak(self):\n                    return self.answer\n\n            import gc\n            gc.immortalize_heap()\n\n            def f(x):\n                return x.speak()\n\n            # Prime the cache\n            for _ in range({REPETITION}):\n                f(Oracle())\n\n            print(f(Oracle()))\n            '
        (rc, out, err) = skip_ret_code_check_for_leaking_test_in_asan_mode('-c', code)
        self.assertEqual(out.strip(), b'42')

    @skipIf(not hasattr('gc', 'is_immortal'), 'no immortalization')
    def test_load_shadowed_immortal_method_split_dict(self):
        if False:
            for i in range(10):
                print('nop')
        code = f'if 1:\n            class Oracle:\n                def __init__(self):\n                    self.answer = 42\n\n                def speak(self):\n                    return self.answer\n\n            import gc\n            gc.immortalize_heap()\n\n            def f(x):\n                return x.speak()\n\n            # Prime the cache\n            for _ in range({REPETITION}):\n                f(Oracle())\n\n            # Shadow the method\n            obj = Oracle()\n            obj.speak = 12345\n\n            print(f(Oracle()))\n            '
        (rc, out, err) = skip_ret_code_check_for_leaking_test_in_asan_mode('-c', code)
        self.assertEqual(out.strip(), b'42')

    @skipIf(not hasattr('gc', 'is_immortal'), 'no immortalization')
    def test_load_unshadowed_immortal_method_combineddict(self):
        if False:
            return 10
        code = f'if 1:\n            class Oracle:\n                def __init__(self):\n                    self.answer = 42\n\n                def speak(self):\n                    return self.answer\n\n            import gc\n            gc.immortalize_heap()\n\n            obj = Oracle()\n            obj.foo = 1\n            # Force the class to use combined dictionaries\n            del obj.foo\n\n            def f(x):\n                return x.speak()\n\n            # Prime the cache\n            for _ in range({REPETITION}):\n                f(Oracle())\n\n            print(f(Oracle()))\n            '
        (rc, out, err) = assert_python_ok('-c', code)
        self.assertEqual(out.strip(), b'42')

    @skipIf(not hasattr('gc', 'is_immortal'), 'no immortalization')
    def test_load_shadowed_immortal_method_combineddict(self):
        if False:
            i = 10
            return i + 15
        code = f'if 1:\n            class Oracle:\n                def __init__(self):\n                    self.answer = 42\n\n                def speak(self):\n                    return self.answer\n\n            import gc\n            gc.immortalize_heap()\n\n            obj = Oracle()\n            obj.foo = 1\n            # Force the class to use combined dictionaries\n            del obj.foo\n\n            def f(x):\n                return x.speak()\n\n            # Prime the cache\n            for _ in range({REPETITION}):\n                f(Oracle())\n\n            # Shadow the method\n            obj = Oracle()\n            obj.speak = 12345\n\n            print(f(Oracle()))\n            '
        (rc, out, err) = skip_ret_code_check_for_leaking_test_in_asan_mode('-c', code)
        self.assertEqual(out.strip(), b'42')

    @skipIf(not hasattr('gc', 'is_immortal'), 'no immortalization')
    def test_load_unshadowed_immortal_method_no_dict(self):
        if False:
            i = 10
            return i + 15
        code = f'if 1:\n            import gc\n            gc.immortalize_heap()\n\n            def f(x):\n                return x.count(1)\n\n            l = [1, 2, 3, 1]\n            # Prime the cache\n            for _ in range({REPETITION}):\n                f(l)\n\n            print(f(l))\n            '
        (rc, out, err) = skip_ret_code_check_for_leaking_test_in_asan_mode('-c', code)
        self.assertEqual(out.strip(), b'2')

    def test_instance_to_type(self):
        if False:
            i = 10
            return i + 15

        def f(obj, use_type):
            if False:
                print('Hello World!')
            if use_type:
                return obj.foo
            else:
                return obj.foo

        class C:

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                self.foo = 42
        x = C()
        for i in range(REPETITION):
            f(x, False)
        with self.assertRaises(AttributeError):
            f(C, True)

    def test_load_method_function_no_attr(self):
        if False:
            return 10
        'Invalidating a cache and picking up a new cache from a type\n        needs to check that the type has the descriptor'

        class C:

            def getdoc(self):
                if False:
                    while True:
                        i = 10
                return 'doc'

        def f(x, z):
            if False:
                return 10
            if x:
                z.getdoc()
            else:
                z.getdoc()
        a = C()
        for i in range(REPETITION):
            f(True, a)
            f(False, a)
        try:
            f(False, FunctionType)
        except AttributeError:
            pass
        try:
            f(True, FunctionType)
        except AttributeError:
            pass
if __name__ == '__main__':
    unittest.main()