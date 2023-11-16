import math

def test_exceptions():
    if False:
        return 10
    try:
        x = math.exp(-1000000000)
    except:
        raise RuntimeError
    try:
        x = math.sqrt(-1.0)
    except ValueError:
        return x
    else:
        raise RuntimeError
test_exceptions()