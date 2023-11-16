from builtins import _test_sink, _test_source
from contextlib import contextmanager
from typing import List, Optional, Type

class C:
    tainted_attribute: List[int] = []
    tainted_class_attribute: List[int] = []
    not_tainted = 2

class D(C):
    pass

def tainted_attribute_flow_issue(c: C) -> None:
    if False:
        return 10
    c.tainted_attribute = _test_source()

def untainted_flow_not_issue(c: C) -> None:
    if False:
        for i in range(10):
            print('nop')
    c.not_tainted = _test_source()

def tainted_attribute_for_class_not_issue(c: Type[C]) -> None:
    if False:
        return 10
    c.tainted_attribute = _test_source()

def tainted_attribute_through_inheritance_issue(d: D) -> None:
    if False:
        print('Hello World!')
    d.tainted_attribute = _test_source()

def tainted_class_attribute_through_instance_not_issue(c: C) -> None:
    if False:
        while True:
            i = 10
    c.tainted_class_attribute = _test_source()

def tainted_class_attribute_through_class_issue(class_object: Type[C]) -> None:
    if False:
        for i in range(10):
            print('nop')
    class_object.tainted_class_attribute = _test_source()

def tainted_class_attribute_through_double_underscore_class_issue(c: C) -> None:
    if False:
        while True:
            i = 10
    c.__class__.tainted_class_attribute = _test_source()

def tainted_class_attribute_through_optional_class_issue(class_object: Optional[Type[C]]) -> None:
    if False:
        while True:
            i = 10
    if class_object is not None:
        class_object.tainted_class_attribute = _test_source()

def global_class_attribute_issue() -> None:
    if False:
        return 10
    C.tainted_class_attribute = _test_source()

class HasClassmethods:

    @classmethod
    def _async_results_for_non_empty_query_from_db(cls, locale: str):
        if False:
            print('Hello World!')
        if not locale:
            emojis = cls._get_single_word_results(locale)
        else:
            emojis = cls._get_multi_word_results(locale)

    @classmethod
    def _get_multi_word_results(cls, locale: str):
        if False:
            while True:
                i = 10
        _test_sink(locale)
        return ''

    @classmethod
    def _get_single_word_results(cls, locale: str):
        if False:
            print('Hello World!')
        return ''

def test_classmethod():
    if False:
        while True:
            i = 10
    HasClassmethods._async_results_for_non_empty_query_from_db(_test_source())

class HasDecoratedClassmethod:

    @classmethod
    @contextmanager
    def to_sink(self, x):
        if False:
            while True:
                i = 10
        _test_sink(x)

def test_decorated_classmethod():
    if False:
        i = 10
        return i + 15
    HasDecoratedClassmethod.to_sink(_test_source())