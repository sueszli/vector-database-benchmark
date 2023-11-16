import cython
import pickle

def test_optional_posonly_args1(a, b=10, /, c=100):
    if False:
        print('Hello World!')
    "\n    >>> test_optional_posonly_args1(1, 2, 3)\n    6\n    >>> test_optional_posonly_args1(1, 2, c=3)\n    6\n    >>> test_optional_posonly_args1(1, b=2, c=3)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: test_optional_posonly_args1() got ... keyword argument... 'b'\n    >>> test_optional_posonly_args1(1, 2)\n    103\n    >>> test_optional_posonly_args1(1, b=2)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: test_optional_posonly_args1() got ... keyword argument... 'b'\n    "
    return a + b + c

def test_optional_posonly_args2(a=1, b=10, /, c=100):
    if False:
        for i in range(10):
            print('nop')
    "\n    >>> test_optional_posonly_args2(1, 2, 3)\n    6\n    >>> test_optional_posonly_args2(1, 2, c=3)\n    6\n    >>> test_optional_posonly_args2(1, b=2, c=3)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: test_optional_posonly_args2() got ... keyword argument... 'b'\n    >>> test_optional_posonly_args2(1, 2)\n    103\n    >>> test_optional_posonly_args2(1, b=2)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: test_optional_posonly_args2() got ... keyword argument... 'b'\n    >>> test_optional_posonly_args2(1, c=2)\n    13\n    "
    return a + b + c

@cython.binding(True)
def func_introspection1(a, b, c, /, d, e=1, *, f, g=2):
    if False:
        i = 10
        return i + 15
    '\n    >>> assert func_introspection2.__code__.co_argcount == 5, func_introspection2.__code__.co_argcount\n    >>> func_introspection1.__defaults__\n    (1,)\n    '

@cython.binding(True)
def func_introspection2(a, b, c=1, /, d=2, e=3, *, f, g=4):
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> assert func_introspection2.__code__.co_argcount == 5, func_introspection2.__code__.co_argcount\n    >>> func_introspection2.__defaults__\n    (1, 2, 3)\n    '

def test_pos_only_call_via_unpacking(a, b, /):
    if False:
        i = 10
        return i + 15
    '\n    >>> test_pos_only_call_via_unpacking(*[1,2])\n    3\n    '
    return a + b

def test_use_positional_as_keyword1(a, /):
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> test_use_positional_as_keyword1(1)\n    >>> test_use_positional_as_keyword1(a=1)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: test_use_positional_as_keyword1() ... keyword argument...\n    '

def test_use_positional_as_keyword2(a, /, b):
    if False:
        return 10
    '\n    >>> test_use_positional_as_keyword2(1, 2)\n    >>> test_use_positional_as_keyword2(1, b=2)\n    >>> test_use_positional_as_keyword2(a=1, b=2)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: test_use_positional_as_keyword2() ... positional...argument...\n    '

def test_use_positional_as_keyword3(a, b, /):
    if False:
        print('Hello World!')
    '\n    >>> test_use_positional_as_keyword3(1, 2)\n    >>> test_use_positional_as_keyword3(a=1, b=2) # doctest:+ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: test_use_positional_as_keyword3() got ... keyword argument...\n    '

def test_positional_only_and_arg_invalid_calls(a, b, /, c):
    if False:
        print('Hello World!')
    '\n    >>> test_positional_only_and_arg_invalid_calls(1, 2, 3)\n    >>> test_positional_only_and_arg_invalid_calls(1, 2, c=3)\n    >>> test_positional_only_and_arg_invalid_calls(1, 2)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: test_positional_only_and_arg_invalid_calls() ... positional argument...\n    >>> test_positional_only_and_arg_invalid_calls(1)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: test_positional_only_and_arg_invalid_calls() ... positional arguments...\n    >>> test_positional_only_and_arg_invalid_calls(1,2,3,4)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: test_positional_only_and_arg_invalid_calls() takes ... positional arguments ...4 ...given...\n    '

def test_positional_only_and_optional_arg_invalid_calls(a, b, /, c=3):
    if False:
        while True:
            i = 10
    '\n    >>> test_positional_only_and_optional_arg_invalid_calls(1, 2)\n    >>> test_positional_only_and_optional_arg_invalid_calls(1)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: test_positional_only_and_optional_arg_invalid_calls() ... positional argument...\n    >>> test_positional_only_and_optional_arg_invalid_calls()  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: test_positional_only_and_optional_arg_invalid_calls() ... positional arguments...\n    >>> test_positional_only_and_optional_arg_invalid_calls(1, 2, 3, 4)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: test_positional_only_and_optional_arg_invalid_calls() takes ... positional arguments ...4 ...given...\n    '

def test_positional_only_and_kwonlyargs_invalid_calls(a, b, /, c, *, d, e):
    if False:
        return 10
    "\n    >>> test_positional_only_and_kwonlyargs_invalid_calls(1, 2, 3, d=1, e=2)\n    >>> test_positional_only_and_kwonlyargs_invalid_calls(1, 2, 3, e=2)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: test_positional_only_and_kwonlyargs_invalid_calls() ... keyword-only argument...d...\n    >>> test_positional_only_and_kwonlyargs_invalid_calls(1, 2, 3)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: test_positional_only_and_kwonlyargs_invalid_calls() ... keyword-only argument...d...\n    >>> test_positional_only_and_kwonlyargs_invalid_calls(1, 2)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: test_positional_only_and_kwonlyargs_invalid_calls() ... positional argument...\n    >>> test_positional_only_and_kwonlyargs_invalid_calls(1)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: test_positional_only_and_kwonlyargs_invalid_calls() ... positional arguments...\n    >>> test_positional_only_and_kwonlyargs_invalid_calls()  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: test_positional_only_and_kwonlyargs_invalid_calls() ... positional arguments...\n    >>> test_positional_only_and_kwonlyargs_invalid_calls(1, 2, 3, 4, 5, 6, d=7, e=8)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: test_positional_only_and_kwonlyargs_invalid_calls() takes ... positional arguments ...\n    >>> test_positional_only_and_kwonlyargs_invalid_calls(1, 2, 3, d=1, e=4, f=56)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: test_positional_only_and_kwonlyargs_invalid_calls() got an unexpected keyword argument 'f'\n    "

def test_positional_only_invalid_calls(a, b, /):
    if False:
        return 10
    '\n    >>> test_positional_only_invalid_calls(1, 2)\n    >>> test_positional_only_invalid_calls(1)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: test_positional_only_invalid_calls() ... positional argument...\n    >>> test_positional_only_invalid_calls()  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: test_positional_only_invalid_calls() ... positional arguments...\n    >>> test_positional_only_invalid_calls(1, 2, 3)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: test_positional_only_invalid_calls() takes ... positional arguments ...3 ...given...\n    '

def test_positional_only_with_optional_invalid_calls(a, b=2, /):
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> test_positional_only_with_optional_invalid_calls(1)\n    >>> test_positional_only_with_optional_invalid_calls()  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: test_positional_only_with_optional_invalid_calls() ... positional argument...\n    >>> test_positional_only_with_optional_invalid_calls(1, 2, 3)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: test_positional_only_with_optional_invalid_calls() takes ... positional arguments ...3 ...given...\n    '

def test_no_standard_args_usage(a, b, /, *, c):
    if False:
        while True:
            i = 10
    '\n    >>> test_no_standard_args_usage(1, 2, c=3)\n    >>> test_no_standard_args_usage(1, b=2, c=3)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: test_no_standard_args_usage() ... positional... argument...\n    '

def test_lambdas():
    if False:
        return 10
    '\n    >>> test_lambdas()\n    3\n    3\n    3\n    3\n    3\n    '
    x = lambda a, /, b: a + b
    print(x(1, 2))
    print(x(1, b=2))
    x = lambda a, /, b=2: a + b
    print(x(1))
    x = lambda a, b, /: a + b
    print(x(1, 2))
    x = lambda a, b, /: a + b
    print(x(1, 2))

class TestPosonlyMethods(object):
    """
    >>> TestPosonlyMethods().f(1,2)
    (1, 2)
    >>> TestPosonlyMethods.f(TestPosonlyMethods(), 1, 2)
    (1, 2)
    >>> try:
    ...     TestPosonlyMethods.f(1,2)
    ... except TypeError:
    ...    print("Got type error")
    Got type error
    >>> TestPosonlyMethods().f(1, b=2)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    TypeError: ...f() got ... keyword argument... 'b'
    """

    def f(self, a, b, /):
        if False:
            print('Hello World!')
        return (a, b)

class TestMangling(object):
    """
    >>> TestMangling().f()
    42
    >>> TestMangling().f2()
    42

    #>>> TestMangling().f3()
    #(42, 43)
    #>>> TestMangling().f4()
    #(42, 43, 44)

    >>> TestMangling().f2(1)
    1

    #>>> TestMangling().f3(1, _TestMangling__b=2)
    #(1, 2)
    #>>> TestMangling().f4(1, _TestMangling__b=2, _TestMangling__c=3)
    #(1, 2, 3)
    """

    def f(self, *, __a=42):
        if False:
            while True:
                i = 10
        return __a

    def f2(self, __a=42, /):
        if False:
            for i in range(10):
                print('nop')
        return __a

def test_module_function(a, b, /):
    if False:
        i = 10
        return i + 15
    '\n    >>> test_module_function(1, 2)\n    >>> test_module_function()  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: test_module_function() ... positional arguments...\n    '

def test_closures1(x, y):
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> test_closures1(1,2)(3,4)\n    10\n    >>> test_closures1(1,2)(3)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: ...g() ... positional argument...\n    >>> test_closures1(1,2)(3,4,5)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: ...g() ... positional argument...\n    '

    def g(x2, /, y2):
        if False:
            while True:
                i = 10
        return x + y + x2 + y2
    return g

def test_closures2(x, /, y):
    if False:
        while True:
            i = 10
    '\n    >>> test_closures2(1,2)(3,4)\n    10\n    '

    def g(x2, y2):
        if False:
            for i in range(10):
                print('nop')
        return x + y + x2 + y2
    return g

def test_closures3(x, /, y):
    if False:
        i = 10
        return i + 15
    '\n    >>> test_closures3(1,2)(3,4)\n    10\n    >>> test_closures3(1,2)(3)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: ...g() ... positional argument...\n    >>> test_closures3(1,2)(3,4,5)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: ...g() ... positional argument...\n    '

    def g(x2, /, y2):
        if False:
            return 10
        return x + y + x2 + y2
    return g

def test_same_keyword_as_positional_with_kwargs(something, /, **kwargs):
    if False:
        print('Hello World!')
    "\n    >>> test_same_keyword_as_positional_with_kwargs(42, something=42)\n    (42, {'something': 42})\n    >>> test_same_keyword_as_positional_with_kwargs(something=42)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: test_same_keyword_as_positional_with_kwargs() ... positional argument...\n    >>> test_same_keyword_as_positional_with_kwargs(42)\n    (42, {})\n    "
    return (something, kwargs)

def test_serialization1(a, b, /):
    if False:
        return 10
    '\n    >>> pickled_posonly = pickle.dumps(test_serialization1)\n    >>> unpickled_posonly = pickle.loads(pickled_posonly)\n    >>> unpickled_posonly(1, 2)\n    (1, 2)\n    >>> unpickled_posonly(a=1, b=2)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: test_serialization1() got ... keyword argument...\n    '
    return (a, b)

def test_serialization2(a, /, b):
    if False:
        return 10
    '\n    >>> pickled_optional = pickle.dumps(test_serialization2)\n    >>> unpickled_optional = pickle.loads(pickled_optional)\n    >>> unpickled_optional(1, 2)\n    (1, 2)\n    >>> unpickled_optional(a=1, b=2)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: test_serialization2() ... positional... argument...\n    '
    return (a, b)

def test_serialization3(a=1, /, b=2):
    if False:
        for i in range(10):
            print('nop')
    "\n    >>> pickled_defaults = pickle.dumps(test_serialization3)\n    >>> unpickled_defaults = pickle.loads(pickled_defaults)\n    >>> unpickled_defaults(1, 2)\n    (1, 2)\n    >>> unpickled_defaults(a=1, b=2)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: test_serialization3() got ... keyword argument... 'a'\n    "
    return (a, b)

async def test_async(a=1, /, b=2):
    """
    >>> test_async(a=1, b=2)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    TypeError: test_async() got ... keyword argument... 'a'
    """
    return (a, b)

def test_async_call(*args, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    >>> test_async_call(1, 2)\n    >>> test_async_call(1, b=2)\n    >>> test_async_call(1)\n    >>> test_async_call()\n    '
    try:
        coro = test_async(*args, **kwargs)
        coro.send(None)
    except StopIteration as e:
        result = e.value
    assert result == (1, 2), result

def test_generator(a=1, /, b=2):
    if False:
        i = 10
        return i + 15
    "\n    >>> test_generator(a=1, b=2)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: test_generator() got ... keyword argument... 'a'\n    >>> gen = test_generator(1, 2)\n    >>> next(gen)\n    (1, 2)\n    >>> gen = test_generator(1, b=2)\n    >>> next(gen)\n    (1, 2)\n    >>> gen = test_generator(1)\n    >>> next(gen)\n    (1, 2)\n    >>> gen = test_generator()\n    >>> next(gen)\n    (1, 2)\n    "
    yield (a, b)

def f_call_1_0_0(a, /):
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> f_call_1_0_0(1)\n    (1,)\n    '
    return (a,)

def f_call_1_1_0(a, /, b):
    if False:
        i = 10
        return i + 15
    '\n    >>> f_call_1_1_0(1,2)\n    (1, 2)\n    '
    return (a, b)

def f_call_1_1_1(a, /, b, *, c):
    if False:
        return 10
    '\n    >>> f_call_1_1_1(1,2,c=3)\n    (1, 2, 3)\n    '
    return (a, b, c)

def f_call_1_1_1_star(a, /, b, *args, c):
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> f_call_1_1_1_star(1,2,c=3)\n    (1, 2, (), 3)\n    >>> f_call_1_1_1_star(1,2,3,4,5,6,7,8,c=9)\n    (1, 2, (3, 4, 5, 6, 7, 8), 9)\n    '
    return (a, b, args, c)

def f_call_1_1_1_kwds(a, /, b, *, c, **kwds):
    if False:
        print('Hello World!')
    "\n    >>> f_call_1_1_1_kwds(1,2,c=3)\n    (1, 2, 3, {})\n    >>> f_call_1_1_1_kwds(1,2,c=3,d=4,e=5) == (1, 2, 3, {'d': 4, 'e': 5})\n    True\n    "
    return (a, b, c, kwds)

def f_call_1_1_1_star_kwds(a, /, b, *args, c, **kwds):
    if False:
        return 10
    "\n    >>> f_call_1_1_1_star_kwds(1,2,c=3,d=4,e=5) == (1, 2, (), 3, {'d': 4, 'e': 5})\n    True\n    >>> f_call_1_1_1_star_kwds(1,2,3,4,c=5,d=6,e=7) == (1, 2, (3, 4), 5, {'d': 6, 'e': 7})\n    True\n    "
    return (a, b, args, c, kwds)

def f_call_one_optional_kwd(a, /, *, b=2):
    if False:
        i = 10
        return i + 15
    '\n    >>> f_call_one_optional_kwd(1)\n    (1, 2)\n    >>> f_call_one_optional_kwd(1, b=3)\n    (1, 3)\n    '
    return (a, b)

def f_call_posonly_stararg(a, /, *args):
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> f_call_posonly_stararg(1)\n    (1, ())\n    >>> f_call_posonly_stararg(1, 2, 3, 4)\n    (1, (2, 3, 4))\n    '
    return (a, args)

def f_call_posonly_kwarg(a, /, **kw):
    if False:
        for i in range(10):
            print('nop')
    "\n    >>> f_call_posonly_kwarg(1)\n    (1, {})\n    >>> all_args = f_call_posonly_kwarg(1, b=2, c=3, d=4)\n    >>> all_args == (1, {'b': 2, 'c': 3, 'd': 4}) or all_args\n    True\n    "
    return (a, kw)

def f_call_posonly_stararg_kwarg(a, /, *args, **kw):
    if False:
        return 10
    "\n    >>> f_call_posonly_stararg_kwarg(1)\n    (1, (), {})\n    >>> f_call_posonly_stararg_kwarg(1, 2)\n    (1, (2,), {})\n    >>> all_args = f_call_posonly_stararg_kwarg(1, b=3, c=4)\n    >>> all_args == (1, (), {'b': 3, 'c': 4}) or all_args\n    True\n    >>> all_args = f_call_posonly_stararg_kwarg(1, 2, b=3, c=4)\n    >>> all_args == (1, (2,), {'b': 3, 'c': 4}) or all_args\n    True\n    "
    return (a, args, kw)

def test_empty_kwargs(a, b, /):
    if False:
        print('Hello World!')
    "\n    >>> test_empty_kwargs(1, 2)\n    (1, 2)\n    >>> test_empty_kwargs(1, 2, **{})\n    (1, 2)\n    >>> test_empty_kwargs(1, 2, **{'c': 3})\n    Traceback (most recent call last):\n    TypeError: test_empty_kwargs() got an unexpected keyword argument 'c'\n    "
    return (a, b)

@cython.cclass
class TestExtensionClass:
    """
    >>> t = TestExtensionClass()
    >>> t.f(1,2)
    (1, 2, 3)
    >>> t.f(1,2,4)
    (1, 2, 4)
    >>> t.f(1, 2, c=4)
    (1, 2, 4)
    >>> t.f(1, 2, 5, c=6)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    TypeError: ...f() got multiple values for ...argument 'c'
    """

    def f(self, a, b, /, c=3):
        if False:
            return 10
        return (a, b, c)