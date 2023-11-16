import pytest
from fastapi.exceptions import FastAPIError
from ...utils import needs_py310

@needs_py310
def test_invalid_response_model():
    if False:
        i = 10
        return i + 15
    with pytest.raises(FastAPIError):
        from docs_src.response_model.tutorial003_04_py310 import app
        assert app