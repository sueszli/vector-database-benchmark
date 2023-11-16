import orjson
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

class TestTypedDict:

    def test_typeddict(self):
        if False:
            print('Hello World!')
        '\n        dumps() TypedDict\n        '

        class TypedDict1(TypedDict):
            a: str
            b: int
        obj = TypedDict1(a='a', b=1)
        assert orjson.dumps(obj) == b'{"a":"a","b":1}'