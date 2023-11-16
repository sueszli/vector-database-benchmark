import pytest

def test_alias_delay_initialization1(capture):
    if False:
        return 10
    '\n    A only initializes its trampoline class when we inherit from it; if we just\n    create and use an A instance directly, the trampoline initialization is\n    bypassed and we only initialize an A() instead (for performance reasons).\n    '
    from pybind11_tests import A, call_f

    class B(A):

        def __init__(self):
            if False:
                return 10
            super(B, self).__init__()

        def f(self):
            if False:
                return 10
            print('In python f()')
    with capture:
        a = A()
        call_f(a)
        del a
        pytest.gc_collect()
    assert capture == 'A.f()'
    with capture:
        b = B()
        call_f(b)
        del b
        pytest.gc_collect()
    assert capture == '\n        PyA.PyA()\n        PyA.f()\n        In python f()\n        PyA.~PyA()\n    '

def test_alias_delay_initialization2(capture):
    if False:
        while True:
            i = 10
    'A2, unlike the above, is configured to always initialize the alias; while\n    the extra initialization and extra class layer has small virtual dispatch\n    performance penalty, it also allows us to do more things with the trampoline\n    class such as defining local variables and performing construction/destruction.\n    '
    from pybind11_tests import A2, call_f

    class B2(A2):

        def __init__(self):
            if False:
                return 10
            super(B2, self).__init__()

        def f(self):
            if False:
                while True:
                    i = 10
            print('In python B2.f()')
    with capture:
        a2 = A2()
        call_f(a2)
        del a2
        pytest.gc_collect()
    assert capture == '\n        PyA2.PyA2()\n        PyA2.f()\n        A2.f()\n        PyA2.~PyA2()\n    '
    with capture:
        b2 = B2()
        call_f(b2)
        del b2
        pytest.gc_collect()
    assert capture == '\n        PyA2.PyA2()\n        PyA2.f()\n        In python B2.f()\n        PyA2.~PyA2()\n    '