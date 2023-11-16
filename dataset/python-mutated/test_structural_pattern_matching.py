import sys
import pytest

@pytest.mark.skipif(sys.version_info < (3, 10), reason='requires python 3.10 or higher')
def test_match_kwargs(create_module):
    if False:
        return 10
    module = create_module("\nfrom pydantic import BaseModel\n\nclass Model(BaseModel):\n    a: str\n    b: str\n\ndef main(model):\n    match model:\n        case Model(a='a', b=b):\n            return b\n        case Model(a='a2'):\n            return 'b2'\n        case _:\n            return None\n")
    assert module.main(module.Model(a='a', b='b')) == 'b'
    assert module.main(module.Model(a='a2', b='b')) == 'b2'
    assert module.main(module.Model(a='x', b='b')) is None