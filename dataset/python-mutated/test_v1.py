from pydantic import VERSION
from pydantic.v1 import VERSION as V1_VERSION
from pydantic.v1 import BaseModel as V1BaseModel
from pydantic.v1 import root_validator as v1_root_validator

def test_version():
    if False:
        i = 10
        return i + 15
    assert V1_VERSION.startswith('1.')
    assert V1_VERSION != VERSION

def test_root_validator():
    if False:
        while True:
            i = 10

    class Model(V1BaseModel):
        v: str

        @v1_root_validator(pre=True)
        @classmethod
        def root_validator(cls, values):
            if False:
                return 10
            values['v'] += '-v1'
            return values
    model = Model(v='value')
    assert model.v == 'value-v1'