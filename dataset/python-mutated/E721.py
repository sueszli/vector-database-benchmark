if type(res) == type(42):
    pass
if type(res) != type(''):
    pass
if type(res) == memoryview:
    pass
import types
if res == types.IntType:
    pass
import types
if type(res) is not types.ListType:
    pass
assert type(res) == type(False) or type(res) == type(None)
assert type(res) == type([])
assert type(res) == type(())
assert type(res) == type((0,))
assert type(res) == type(0)
assert type(res) != type((1,))
assert type(res) is type((1,))
assert type(res) is not type((1,))
assert type(res) == type([2])
assert type(res) == type(())
assert type(res) == type((0,))
import types
if isinstance(res, int):
    pass
if isinstance(res, str):
    pass
if isinstance(res, types.MethodType):
    pass
if isinstance(res, memoryview):
    pass
if type(res) is type:
    pass
if type(res) == type:
    pass

def func_histype(a, b, c):
    if False:
        while True:
            i = 10
    pass
try:
    pass
except:
    pass
try:
    pass
except Exception:
    pass
except:
    pass
try:
    pass
except:
    pass
fake_code = '"\ntry:\n    do_something()\nexcept:\n    pass\n'
try:
    pass
except Exception:
    pass
from . import custom_types as types
red = types.ColorTypeRED
red is types.ColorType.RED
from . import compute_type
if compute_type(foo) == 5:
    pass

class Foo:

    def asdf(self, value: str | None):
        if False:
            return 10
        if type(value) is str:
            ...

class Foo:

    def type(self):
        if False:
            i = 10
            return i + 15
        pass

    def asdf(self, value: str | None):
        if False:
            for i in range(10):
                print('nop')
        if type(value) is str:
            ...

class Foo:

    def asdf(self, value: str | None):
        if False:
            return 10

        def type():
            if False:
                while True:
                    i = 10
            pass
        if type(value) is str:
            ...