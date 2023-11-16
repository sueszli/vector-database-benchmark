from vispy.util.dpi import get_dpi
from vispy.testing import run_tests_if_main

def test_dpi():
    if False:
        i = 10
        return i + 15
    'Test dpi support'
    dpi = get_dpi()
    assert dpi > 0.0
    assert isinstance(dpi, float)
run_tests_if_main()