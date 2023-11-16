import math

def test_exceptions():
    if False:
        for i in range(10):
            print('nop')
    try:
        x = math.exp(-1000000000)
    except:
        raise RuntimeError
    x = 1
    try:
        x = math.sqrt(-1.0)
    except ValueError:
        return x
    else:
        raise RuntimeError
test_exceptions()