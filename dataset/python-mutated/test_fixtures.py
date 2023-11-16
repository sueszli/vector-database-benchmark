import time
from tests.common import TIME_INCREMENT

def test_time_consistently_increments_in_tests():
    if False:
        print('Hello World!')
    x = time.time()
    y = time.time()
    z = time.time()
    assert y == x + TIME_INCREMENT
    assert z == y + TIME_INCREMENT