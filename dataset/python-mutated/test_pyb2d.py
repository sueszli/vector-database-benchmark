import pytest

@pytest.mark.driver_timeout(40)
def test_pyb2d(selenium_standalone, request):
    if False:
        for i in range(10):
            print('nop')
    selenium = selenium_standalone
    selenium.load_package('pyb2d')
    selenium.run('\n        import numpy as np\n        import b2d\n        w = b2d.world(gravity=(0,-10))\n        ')