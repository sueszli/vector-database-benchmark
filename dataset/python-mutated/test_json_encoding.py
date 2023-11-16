import sys
from dataclasses import asdict, dataclass
from functools import partial
from json import dumps as sdumps
from string import ascii_lowercase
from typing import Dict
import pytest
try:
    import ujson
    from ujson import dumps as udumps
    ujson_version = tuple(map(int, ujson.__version__.strip(ascii_lowercase).split('.')))
    NO_UJSON = False
    DEFAULT_DUMPS = udumps
except ModuleNotFoundError:
    NO_UJSON = True
    DEFAULT_DUMPS = partial(sdumps, separators=(',', ':'))
    ujson_version = None
from sanic import Sanic
from sanic.response import BaseHTTPResponse, json

@dataclass
class Foo:
    bar: str

    def __json__(self):
        if False:
            for i in range(10):
                print('nop')
        return udumps(asdict(self))

@pytest.fixture
def foo():
    if False:
        for i in range(10):
            print('nop')
    return Foo(bar='bar')

@pytest.fixture
def payload(foo: Foo):
    if False:
        for i in range(10):
            print('nop')
    return {'foo': foo}

@pytest.fixture(autouse=True)
def default_back_to_ujson():
    if False:
        print('Hello World!')
    yield
    BaseHTTPResponse._dumps = DEFAULT_DUMPS

def test_change_encoder():
    if False:
        while True:
            i = 10
    Sanic('Test', dumps=sdumps)
    assert BaseHTTPResponse._dumps == sdumps

def test_change_encoder_to_some_custom():
    if False:
        print('Hello World!')

    def my_custom_encoder():
        if False:
            for i in range(10):
                print('nop')
        return 'foo'
    Sanic('Test', dumps=my_custom_encoder)
    assert BaseHTTPResponse._dumps == my_custom_encoder

@pytest.mark.skipif(NO_UJSON is True, reason='ujson not installed')
def test_json_response_ujson(payload: Dict[str, Foo]):
    if False:
        for i in range(10):
            print('nop')
    'ujson will look at __json__'
    response = json(payload)
    assert response.body == b'{"foo":{"bar":"bar"}}'
    with pytest.raises(TypeError, match='Object of type Foo is not JSON serializable'):
        json(payload, dumps=sdumps)
    Sanic('Test', dumps=sdumps)
    with pytest.raises(TypeError, match='Object of type Foo is not JSON serializable'):
        json(payload)

@pytest.mark.skipif(NO_UJSON is True or ujson_version >= (5, 4, 0), reason='ujson not installed or version is 5.4.0 or newer, which can handle arbitrary size integers')
def test_json_response_json():
    if False:
        return 10
    'One of the easiest ways to tell the difference is that ujson cannot\n    serialize over 64 bits'
    too_big_for_ujson = 111111111111111111111
    with pytest.raises(OverflowError, match='int too big to convert'):
        json(too_big_for_ujson)
    response = json(too_big_for_ujson, dumps=sdumps)
    assert sys.getsizeof(response.body) == 54
    Sanic('Test', dumps=sdumps)
    response = json(too_big_for_ujson)
    assert sys.getsizeof(response.body) == 54