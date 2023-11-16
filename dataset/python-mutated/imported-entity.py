from randomlib.errors import RandomError

def test_test():
    if False:
        for i in range(10):
            print('nop')
    raise RandomError()