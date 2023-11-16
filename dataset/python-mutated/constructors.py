from builtins import _test_sink, _test_source
from typing import Any, Dict

class SomeAPI:
    HOST = 'api.some.com/1.1'
    AUTHENTICATE_URL = f'https://{HOST}/some.json'

    def __init__(self, oauth_token: str, oauth_token_secret: str) -> None:
        if False:
            while True:
                i = 10
        self.oauth_token = oauth_token
        self.oauth_token_secret = oauth_token_secret

    @classmethod
    def from_default_keys(cls, oauth_token: str, oauth_token_secret: str) -> 'SomeAPI':
        if False:
            i = 10
            return i + 15
        return cls(oauth_token, oauth_token_secret)

    def async_get_authenticated_user(self):
        if False:
            print('Hello World!')
        eval(self.AUTHENTICATE_URL)

class HttpRequest:
    POST: Dict[str, Any] = {}

def test_construction(request: HttpRequest):
    if False:
        while True:
            i = 10
    data = request.POST
    instance = SomeAPI.from_default_keys(data['1'], data['2'])
    instance.async_get_authenticated_user()
    return instance

class SourceInConstructor:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.x = _test_source()
        self.y = 0

def test_source_in_constructor():
    if False:
        print('Hello World!')
    c = SourceInConstructor()
    _test_sink(c.x)
    _test_sink(c.y)

class ParentWithInit:

    def __init__(self):
        if False:
            print('Hello World!')
        pass

class ChildWithNew(ParentWithInit):

    def __new__(cls, input):
        if False:
            return 10
        _test_sink(input)
        return object.__new__(cls)

def test_new_thing():
    if False:
        while True:
            i = 10
    c = ChildWithNew(_test_source())

class BothNewAndInit:

    def __new__(cls):
        if False:
            while True:
                i = 10
        obj = super(BothNewAndInit, cls).__new__()
        obj.foo = _test_source()
        return obj

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        _test_sink(self.foo)

def test_both_new_and_init_callgraph():
    if False:
        while True:
            i = 10
    BothNewAndInit()

class BaseConstructor:

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self.x = _test_source()

class DerivedConstructor(BaseConstructor):

    def __init__(self, y: int) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self.y = y

class InitWithModel:

    def __init__(self, tito=None, not_tito=None):
        if False:
            return 10
        ...

def test_init_model():
    if False:
        return 10
    _test_sink(InitWithModel(tito=_test_source()))
    _test_sink(InitWithModel(not_tito=_test_source()))

class NewWithModel:

    def __new__(cls, tito=None, not_tito=None):
        if False:
            while True:
                i = 10
        ...

def test_new_model():
    if False:
        i = 10
        return i + 15
    _test_sink(NewWithModel(tito=_test_source()))
    _test_sink(NewWithModel(not_tito=_test_source()))

class ClassStub:
    ...

def test_class_stub():
    if False:
        return 10
    _test_sink(ClassStub(_test_source()))