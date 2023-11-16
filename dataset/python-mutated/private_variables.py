from builtins import _test_sink, _test_source
from typing import List

class Simple:

    def __init__(self, private: str='', public: str='') -> None:
        if False:
            print('Hello World!')
        self.__value: str = private
        self.value: str = public

    def private_into_sink(self) -> None:
        if False:
            while True:
                i = 10
        _test_sink(self.__value)

    def public_into_sink(self) -> None:
        if False:
            i = 10
            return i + 15
        _test_sink(self.value)

    @staticmethod
    def expand_subexpression(values: List[Simple]) -> None:
        if False:
            while True:
                i = 10
        _test_sink(values[0].__value)

    def getattr_public(self) -> str:
        if False:
            i = 10
            return i + 15
        return getattr(self, 'value')

    def getattr_private(self) -> str:
        if False:
            return 10
        return getattr(self, '_Simple__value')

    def getattr_invalid(self) -> str:
        if False:
            while True:
                i = 10
        return getattr(self, '__value')

def test_simple() -> None:
    if False:
        print('Hello World!')
    Simple(private=_test_source()).private_into_sink()

def test_private_public_different() -> None:
    if False:
        for i in range(10):
            print('nop')
    Simple(private=_test_source()).private_into_sink()
    Simple(private=_test_source()).public_into_sink()
    Simple(public=_test_source()).private_into_sink()
    Simple(public=_test_source()).public_into_sink()

def test_expand_subexpression() -> None:
    if False:
        for i in range(10):
            print('nop')
    Simple.expand_subexpression([Simple(private=_test_source())])
    Simple.expand_subexpression([Simple(), Simple(private=_test_source())])

def test_getattr() -> None:
    if False:
        print('Hello World!')
    _test_sink(Simple(private=_test_source()).getattr_public())
    _test_sink(Simple(private=_test_source()).getattr_private())
    _test_sink(Simple(private=_test_source()).getattr_invalid())
    _test_sink(Simple(public=_test_source()).getattr_public())
    _test_sink(Simple(public=_test_source()).getattr_private())
    _test_sink(Simple(public=_test_source()).getattr_invalid())

def test_bypass_private() -> None:
    if False:
        i = 10
        return i + 15
    _test_sink(Simple(private=_test_source())._Simple__value)
    _test_sink(Simple(public=_test_source())._Simple__value)
    _test_sink(Simple(private=_test_source()).__value)
    _test_sink(Simple(public=_test_source()).__value)

class Other:

    @staticmethod
    def private_into_sink(s: Simple) -> None:
        if False:
            print('Hello World!')
        _test_sink(s.__value)

def test_access_from_other_class() -> None:
    if False:
        while True:
            i = 10
    Other.private_into_sink(Simple(private=_test_source()))

class PrivateAttributeSourceModels:

    def __init__(self):
        if False:
            return 10
        self.__model_mangled: str = ''
        self.__model_unmangled: str = ''
        self.__model_query: str = ''

    def get_model_mangled(self) -> str:
        if False:
            return 10
        return self.__model_mangled

    def get_model_unmangled(self) -> str:
        if False:
            while True:
                i = 10
        return self.__model_unmangled

    def get_model_query(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.__model_query

def test_private_attribute_source_models() -> None:
    if False:
        print('Hello World!')
    _test_sink(PrivateAttributeSourceModels().get_model_mangled())
    _test_sink(PrivateAttributeSourceModels().get_model_unmangled())
    _test_sink(PrivateAttributeSourceModels().get_model_query())

class PrivateAttributeSinkModels:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.__model_mangled: str = ''
        self.__model_unmangled: str = ''
        self.__model_query: str = ''

    def set_model_mangled(self, value: str) -> None:
        if False:
            return 10
        self.__model_mangled = value

    def set_model_unmangled(self, value: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.__model_unmangled = value

    def set_model_query(self, value: str) -> None:
        if False:
            while True:
                i = 10
        self.__model_query = value

def test_private_attribute_sink_models() -> None:
    if False:
        print('Hello World!')
    PrivateAttributeSinkModels().set_model_mangled(_test_source())
    PrivateAttributeSinkModels().set_model_unmangled(_test_source())
    PrivateAttributeSinkModels().set_model_query(_test_source())