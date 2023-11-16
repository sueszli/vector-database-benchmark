import pytest
import orjson

@pytest.mark.parametrize('input', [b'"\xc8\x93', b'"\xc8'])
def test_invalid(input):
    if False:
        print('Hello World!')
    with pytest.raises(orjson.JSONDecodeError):
        orjson.loads(input)