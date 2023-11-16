import pytest
from pybind11_tests import ConstructorStats

def test_regressions():
    if False:
        i = 10
        return i + 15
    from pybind11_tests.issues import print_cchar, print_char
    assert print_cchar('const char *') == 'const char *'
    assert print_char('c') == 'c'

def test_dispatch_issue(msg):
    if False:
        return 10
    '#159: virtual function dispatch has problems with similar-named functions'
    from pybind11_tests.issues import DispatchIssue, dispatch_issue_go

    class PyClass1(DispatchIssue):

        def dispatch(self):
            if False:
                return 10
            return 'Yay..'

    class PyClass2(DispatchIssue):

        def dispatch(self):
            if False:
                print('Hello World!')
            with pytest.raises(RuntimeError) as excinfo:
                super(PyClass2, self).dispatch()
            assert msg(excinfo.value) == 'Tried to call pure virtual function "Base::dispatch"'
            p = PyClass1()
            return dispatch_issue_go(p)
    b = PyClass2()
    assert dispatch_issue_go(b) == 'Yay..'

def test_reference_wrapper():
    if False:
        for i in range(10):
            print('nop')
    "#171: Can't return reference wrappers (or STL data structures containing them)"
    from pybind11_tests.issues import Placeholder, return_vec_of_reference_wrapper
    assert str(return_vec_of_reference_wrapper(Placeholder(4))) == '[Placeholder[1], Placeholder[2], Placeholder[3], Placeholder[4]]'

def test_iterator_passthrough():
    if False:
        for i in range(10):
            print('nop')
    '#181: iterator passthrough did not compile'
    from pybind11_tests.issues import iterator_passthrough
    assert list(iterator_passthrough(iter([3, 5, 7, 9, 11, 13, 15]))) == [3, 5, 7, 9, 11, 13, 15]

def test_shared_ptr_gc():
    if False:
        return 10
    '// #187: issue involving std::shared_ptr<> return value policy & garbage collection'
    from pybind11_tests.issues import ElementList, ElementA
    el = ElementList()
    for i in range(10):
        el.add(ElementA(i))
    pytest.gc_collect()
    for (i, v) in enumerate(el.get()):
        assert i == v.value()

def test_no_id(msg):
    if False:
        i = 10
        return i + 15
    from pybind11_tests.issues import get_element, expect_float, expect_int
    with pytest.raises(TypeError) as excinfo:
        get_element(None)
    assert msg(excinfo.value) == '\n        get_element(): incompatible function arguments. The following argument types are supported:\n            1. (arg0: m.issues.ElementA) -> int\n\n        Invoked with: None\n    '
    with pytest.raises(TypeError) as excinfo:
        expect_int(5.2)
    assert msg(excinfo.value) == '\n        expect_int(): incompatible function arguments. The following argument types are supported:\n            1. (arg0: int) -> int\n\n        Invoked with: 5.2\n    '
    assert expect_float(12) == 12

def test_str_issue(msg):
    if False:
        for i in range(10):
            print('nop')
    'Issue #283: __str__ called on uninitialized instance when constructor arguments invalid'
    from pybind11_tests.issues import StrIssue
    assert str(StrIssue(3)) == 'StrIssue[3]'
    with pytest.raises(TypeError) as excinfo:
        str(StrIssue('no', 'such', 'constructor'))
    assert msg(excinfo.value) == "\n        __init__(): incompatible constructor arguments. The following argument types are supported:\n            1. m.issues.StrIssue(arg0: int)\n            2. m.issues.StrIssue()\n\n        Invoked with: 'no', 'such', 'constructor'\n    "

def test_nested():
    if False:
        for i in range(10):
            print('nop')
    " #328: first member in a class can't be used in operators"
    from pybind11_tests.issues import NestA, NestB, NestC, get_NestA, get_NestB, get_NestC
    a = NestA()
    b = NestB()
    c = NestC()
    a += 10
    assert get_NestA(a) == 13
    b.a += 100
    assert get_NestA(b.a) == 103
    c.b.a += 1000
    assert get_NestA(c.b.a) == 1003
    b -= 1
    assert get_NestB(b) == 3
    c.b -= 3
    assert get_NestB(c.b) == 1
    c *= 7
    assert get_NestC(c) == 35
    abase = a.as_base()
    assert abase.value == -2
    a.as_base().value += 44
    assert abase.value == 42
    assert c.b.a.as_base().value == -2
    c.b.a.as_base().value += 44
    assert c.b.a.as_base().value == 42
    del c
    pytest.gc_collect()
    del a
    pytest.gc_collect()
    assert abase.value == 42
    del abase, b
    pytest.gc_collect()

def test_move_fallback():
    if False:
        i = 10
        return i + 15
    from pybind11_tests.issues import get_moveissue1, get_moveissue2
    m2 = get_moveissue2(2)
    assert m2.value == 2
    m1 = get_moveissue1(1)
    assert m1.value == 1

def test_override_ref():
    if False:
        i = 10
        return i + 15
    from pybind11_tests.issues import OverrideTest
    o = OverrideTest('asdf')
    assert o.str_value() == 'asdf'
    assert o.A_value().value == 'hi'
    a = o.A_ref()
    assert a.value == 'hi'
    a.value = 'bye'
    assert a.value == 'bye'

def test_operators_notimplemented(capture):
    if False:
        for i in range(10):
            print('nop')
    from pybind11_tests.issues import OpTest1, OpTest2
    with capture:
        (c1, c2) = (OpTest1(), OpTest2())
        c1 + c1
        c2 + c2
        c2 + c1
        c1 + c2
    assert capture == '\n        Add OpTest1 with OpTest1\n        Add OpTest2 with OpTest2\n        Add OpTest2 with OpTest1\n        Add OpTest2 with OpTest1\n    '

def test_iterator_rvpolicy():
    if False:
        return 10
    " Issue 388: Can't make iterators via make_iterator() with different r/v policies "
    from pybind11_tests.issues import make_iterator_1
    from pybind11_tests.issues import make_iterator_2
    assert list(make_iterator_1()) == [1, 2, 3]
    assert list(make_iterator_2()) == [1, 2, 3]
    assert not isinstance(make_iterator_1(), type(make_iterator_2()))

def test_dupe_assignment():
    if False:
        for i in range(10):
            print('nop')
    ' Issue 461: overwriting a class with a function '
    from pybind11_tests.issues import dupe_exception_failures
    assert dupe_exception_failures() == []

def test_enable_shared_from_this_with_reference_rvp():
    if False:
        return 10
    ' Issue #471: shared pointer instance not dellocated '
    from pybind11_tests import SharedParent, SharedChild
    parent = SharedParent()
    child = parent.get_child()
    cstats = ConstructorStats.get(SharedChild)
    assert cstats.alive() == 1
    del child, parent
    assert cstats.alive() == 0

def test_non_destructed_holders():
    if False:
        for i in range(10):
            print('nop')
    ' Issue #478: unique ptrs constructed and freed without destruction '
    from pybind11_tests import SpecialHolderObj
    a = SpecialHolderObj(123)
    b = a.child()
    assert a.val == 123
    assert b.val == 124
    cstats = SpecialHolderObj.holder_cstats()
    assert cstats.alive() == 1
    del b
    assert cstats.alive() == 1
    del a
    assert cstats.alive() == 0

def test_complex_cast(capture):
    if False:
        for i in range(10):
            print('nop')
    ' Issue #484: number conversion generates unhandled exceptions '
    from pybind11_tests.issues import test_complex
    with capture:
        test_complex(1)
        test_complex(2j)
    assert capture == '\n        1.0\n        (0.0, 2.0)\n    '

def test_inheritance_override_def_static():
    if False:
        print('Hello World!')
    from pybind11_tests.issues import MyBase, MyDerived
    b = MyBase.make()
    d1 = MyDerived.make2()
    d2 = MyDerived.make()
    assert isinstance(b, MyBase)
    assert isinstance(d1, MyDerived)
    assert isinstance(d2, MyDerived)