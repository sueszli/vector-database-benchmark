from builtins import _test_sink, _test_source
from typing import Optional, TypeVar
T = TypeVar('T')

class Builder:

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        self._saved: Optional[str] = None
        self._not_saved: Optional[str] = None

    def set_saved(self, saved: str) -> 'Builder':
        if False:
            for i in range(10):
                print('nop')
        self._saved = saved
        return self

    def set_not_saved(self, not_saved: str) -> 'Builder':
        if False:
            while True:
                i = 10
        self._not_saved = not_saved
        return self

    def async_save(self) -> None:
        if False:
            while True:
                i = 10
        _test_sink(self._saved)

    def set_saved_through_typevar(self: T, saved: str) -> T:
        if False:
            return 10
        self._saved = saved
        return self

    def set_not_saved_through_typevar(self: T, not_saved: str) -> T:
        if False:
            while True:
                i = 10
        self._not_saved = not_saved
        return self

    def return_self(self) -> 'Builder':
        if False:
            return 10
        return self

    def set_saved_no_return(self, saved: str) -> None:
        if False:
            print('Hello World!')
        self._saved = saved

def test_no_issue():
    if False:
        print('Hello World!')
    builder = Builder()
    builder.set_not_saved(_test_source()).set_saved('benign').async_save()

def test_issue():
    if False:
        for i in range(10):
            print('nop')
    builder = Builder()
    builder.set_not_saved('benign').set_saved(_test_source()).async_save()

def test_no_issue_with_type_var():
    if False:
        while True:
            i = 10
    builder = Builder()
    builder.set_not_saved_through_typevar(_test_source()).set_saved_through_typevar('benign').async_save()

def test_issue_with_type_var():
    if False:
        return 10
    builder = Builder()
    builder.set_not_saved_through_typevar('benign').set_saved_through_typevar(_test_source()).async_save()

def test_chained_class_setter():
    if False:
        print('Hello World!')
    builder = Builder()
    builder.return_self().set_saved_no_return(_test_source())
    _test_sink(builder)
    _test_sink(builder._saved)

def test_class_setter():
    if False:
        i = 10
        return i + 15
    builder = Builder()
    builder.set_saved_no_return(_test_source())
    _test_sink(builder)
    _test_sink(builder._saved)

def test_taint_update_receiver_declaration():
    if False:
        for i in range(10):
            print('nop')
    builder = Builder()
    builder.return_self().set_saved(_test_source())
    _test_sink(builder)
    _test_sink(builder._saved)
    _test_sink(builder.return_self())

class SubBuilder(Builder):
    pass

def test_no_issue_with_sub_builder():
    if False:
        return 10
    builder = SubBuilder()
    builder.set_not_saved_through_typevar(_test_source()).set_saved_through_typevar('benign').async_save()

def test_issue_with_sub_builder():
    if False:
        return 10
    builder = SubBuilder()
    builder.set_not_saved_through_typevar('benign').set_saved_through_typevar(_test_source()).async_save()