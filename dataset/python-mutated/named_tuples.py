import collections
from builtins import _test_sink, _test_source
from typing import NamedTuple

class MyNamedTuple(NamedTuple):
    benign: int
    bad: str

def tainted_tuple() -> MyNamedTuple:
    if False:
        print('Hello World!')
    return MyNamedTuple(bad=_test_source(), benign=1)

def issue_with_bad():
    if False:
        while True:
            i = 10
    a = tainted_tuple()
    _test_sink(a.bad)

def no_issue_with_benign():
    if False:
        return 10
    a = tainted_tuple()
    _test_sink(a.benign)
OldSchoolNamedTuple = collections.namedtuple('OldSchoolNamedTuple', 'benign bad')

def tainted_old_tuple():
    if False:
        while True:
            i = 10
    return OldSchoolNamedTuple(bad=_test_source(), benign=1)

def issue_with_old_school_named_tuples():
    if False:
        return 10
    a = tainted_old_tuple()
    _test_sink(a.bad)

def no_issue_with_old_school_named_tuples():
    if False:
        i = 10
        return i + 15
    a = tainted_old_tuple()
    _test_sink(a.benign)

class InheritedNamedTuple(MyNamedTuple):
    pass

def inherited_tuple():
    if False:
        for i in range(10):
            print('nop')
    return InheritedNamedTuple(bad=_test_source(), benign=1)

def issue_with_inherited_named_tuple():
    if False:
        i = 10
        return i + 15
    a = inherited_tuple()
    _test_sink(a.bad)

def no_issue_with_benign_in_inherited_named_tuple():
    if False:
        print('Hello World!')
    a = inherited_tuple()
    _test_sink(a.benign)

def aliased_indicies_forward():
    if False:
        print('Hello World!')
    a = tainted_tuple()
    _test_sink(a[0])
    _test_sink(a[1])
    _test_sink(a[2])

def aliased_indicies_forward_unknown_attribute(i: int):
    if False:
        while True:
            i = 10
    a = tainted_tuple()
    return a[i]

def aliased_indicies_backward(a: MyNamedTuple):
    if False:
        print('Hello World!')
    _test_sink(a.benign)
    _test_sink(a[1])
    _test_sink(a[2])

def aliased_indicies_backward_unknown_attribute(a: MyNamedTuple, i: int):
    if False:
        for i in range(10):
            print('nop')
    _test_sink(a[i])

class NamedTupleWithTaintedAttribute(NamedTuple):
    benign: int
    bad: str

def issue_with_named_tuple_with_tainted_attribute():
    if False:
        return 10
    NamedTupleWithTaintedAttribute(bad=_test_source(), benign=1)