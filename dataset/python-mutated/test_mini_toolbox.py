import pytest
from . import mini_toolbox

def test_output_capturer_doesnt_swallow_exceptions():
    if False:
        return 10
    with pytest.raises(ZeroDivisionError):
        with mini_toolbox.OutputCapturer():
            1 / 0