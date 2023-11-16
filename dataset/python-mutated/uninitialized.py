def conditional(cond):
    if False:
        return 10
    "\n    >>> conditional(True)\n    []\n    >>> conditional(False)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    ...\n    UnboundLocalError: ...local variable 'a'...\n    "
    if cond:
        a = []
    return a

def inside_loop(iter):
    if False:
        i = 10
        return i + 15
    "\n    >>> inside_loop([1,2,3])\n    3\n    >>> inside_loop([])  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    ...\n    UnboundLocalError: ...local variable 'i'...\n    "
    for i in iter:
        pass
    return i

def try_except(cond):
    if False:
        print('Hello World!')
    "\n    >>> try_except(True)\n    []\n    >>> try_except(False)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    ...\n    UnboundLocalError: ...local variable 'a'...\n    "
    try:
        if cond:
            a = []
        raise ValueError
    except ValueError:
        return a

def try_finally(cond):
    if False:
        i = 10
        return i + 15
    "\n    >>> try_finally(True)\n    []\n    >>> try_finally(False)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    ...\n    UnboundLocalError: ...local variable 'a'...\n    "
    try:
        if cond:
            a = []
        raise ValueError
    finally:
        return a

def deleted(cond):
    if False:
        while True:
            i = 10
    "\n    >>> deleted(False)\n    {}\n    >>> deleted(True)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    ...\n    UnboundLocalError: ...local variable 'a'...\n    "
    a = {}
    if cond:
        del a
    return a

def test_nested(cond):
    if False:
        return 10
    "\n    >>> test_nested(True)\n    >>> test_nested(False)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    ...\n    UnboundLocalError: ...local variable 'a'...\n    "
    if cond:

        def a():
            if False:
                return 10
            pass
    return a()

def test_outer(cond):
    if False:
        while True:
            i = 10
    "\n    >>> test_outer(True)\n    {}\n    >>> test_outer(False)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    ...\n    UnboundLocalError: ...local variable 'a'...\n    "
    if cond:
        a = {}

    def inner():
        if False:
            while True:
                i = 10
        return a
    return a

def test_inner(cond):
    if False:
        return 10
    "\n    >>> test_inner(True)\n    {}\n    >>> test_inner(False)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    ...\n    NameError: ...free variable 'a' ... in enclosing scope\n    "
    if cond:
        a = {}

    def inner():
        if False:
            while True:
                i = 10
        return a
    return inner()

def test_class(cond):
    if False:
        return 10
    "\n    >>> test_class(True)\n    1\n    >>> test_class(False)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    ...\n    UnboundLocalError: ...local variable 'A'...\n    "
    if cond:

        class A:
            x = 1
    return A.x

def test_try_except_regression(c):
    if False:
        i = 10
        return i + 15
    "\n    >>> test_try_except_regression(True)\n    (123,)\n    >>> test_try_except_regression(False)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    ...\n    UnboundLocalError: ...local variable 'a'...\n    "
    if c:
        a = (123,)
    try:
        return a
    except:
        return a

def test_try_finally_regression(c):
    if False:
        while True:
            i = 10
    "\n    >>> test_try_finally_regression(True)\n    (123,)\n    >>> test_try_finally_regression(False)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    ...\n    UnboundLocalError: ...local variable 'a'...\n    "
    if c:
        a = (123,)
    try:
        return a
    finally:
        return a

def test_expression_calculation_order_bug(a):
    if False:
        while True:
            i = 10
    "\n    >>> test_expression_calculation_order_bug(False)\n    []\n    >>> test_expression_calculation_order_bug(True)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    ...\n    UnboundLocalError: ...local variable 'b'...\n    "
    if not a:
        b = []
    return (a or b) and (b or a)