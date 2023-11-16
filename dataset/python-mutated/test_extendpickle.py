import dill as pickle
try:
    from StringIO import StringIO
except ImportError:
    from io import BytesIO as StringIO

def my_fn(x):
    if False:
        for i in range(10):
            print('nop')
    return x * 17

def test_extend():
    if False:
        for i in range(10):
            print('nop')
    obj = lambda : my_fn(34)
    assert obj() == 578
    obj_io = StringIO()
    pickler = pickle.Pickler(obj_io)
    pickler.dump(obj)
    obj_str = obj_io.getvalue()
    obj2_io = StringIO(obj_str)
    unpickler = pickle.Unpickler(obj2_io)
    obj2 = unpickler.load()
    assert obj2() == 578
if __name__ == '__main__':
    test_extend()