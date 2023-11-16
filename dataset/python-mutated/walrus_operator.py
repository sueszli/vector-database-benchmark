from builtins import _test_source

def walrus_operator_forward():
    if False:
        while True:
            i = 10
    (x := _test_source())
    (x := 1)
    (y := _test_source())
    return (x, y)

def walrus_operator_backward(x):
    if False:
        print('Hello World!')
    (y := (y := x))
    (x := 1)
    return (x, y)