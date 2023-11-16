import pytest
from fastapi.exceptions import FastAPIError

def test_invalid_response_model():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(FastAPIError):
        from docs_src.response_model.tutorial003_04 import app
        assert app