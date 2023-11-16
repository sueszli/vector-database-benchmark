from functools import wraps
import paddle

def deco1(fun):
    if False:
        while True:
            i = 10

    @wraps(fun)
    def inner(*args, **kwargs):
        if False:
            return 10
        print('in decos.deco1, added 1')
        _t = paddle.to_tensor([1])
        _tt = fun(*args, **kwargs)
        return paddle.add(_t, _tt)
    return inner

def deco2(x=0):
    if False:
        while True:
            i = 10

    def inner_deco(func):
        if False:
            return 10

        @wraps(func)
        def inner(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            print(f'in decos.deco2, added {x}')
            _t = paddle.to_tensor(x)
            _tt = func(*args, **kwargs)
            return paddle.add(_t, _tt)
        return inner
    return inner_deco