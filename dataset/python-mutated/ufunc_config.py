"""Typing tests for `numpy._core._ufunc_config`."""
import numpy as np

def func1(a: str, b: int) -> None:
    if False:
        i = 10
        return i + 15
    return None

def func2(a: str, b: int, c: float=1.0) -> None:
    if False:
        return 10
    return None

def func3(a: str, b: int) -> int:
    if False:
        return 10
    return 0

class Write1:

    def write(self, a: str) -> None:
        if False:
            print('Hello World!')
        return None

class Write2:

    def write(self, a: str, b: int=1) -> None:
        if False:
            i = 10
            return i + 15
        return None

class Write3:

    def write(self, a: str) -> int:
        if False:
            for i in range(10):
                print('nop')
        return 0
_err_default = np.geterr()
_bufsize_default = np.getbufsize()
_errcall_default = np.geterrcall()
try:
    np.seterr(all=None)
    np.seterr(divide='ignore')
    np.seterr(over='warn')
    np.seterr(under='call')
    np.seterr(invalid='raise')
    np.geterr()
    np.setbufsize(4096)
    np.getbufsize()
    np.seterrcall(func1)
    np.seterrcall(func2)
    np.seterrcall(func3)
    np.seterrcall(Write1())
    np.seterrcall(Write2())
    np.seterrcall(Write3())
    np.geterrcall()
    with np.errstate(call=func1, all='call'):
        pass
    with np.errstate(call=Write1(), divide='log', over='log'):
        pass
finally:
    np.seterr(**_err_default)
    np.setbufsize(_bufsize_default)
    np.seterrcall(_errcall_default)