from returns.context import RequiresContextIOResultE
from returns.io import IOSuccess

def test_regression394():
    if False:
        for i in range(10):
            print('nop')
    '\n    It used to raise ``ImmutableStateError`` for type aliases.\n\n    Here we use the minimal reproduction sample.\n\n    .. code:: python\n\n      Traceback (most recent call last):\n        File "ex.py", line 18, in <module>\n            get_ip_addr("https://google.com")\n        File "ex.py", line 13, in get_ip_addr\n            return RequiresContextIOResultE(lambda _: IOSuccess(1))\n        File "../3.7.7/lib/python3.7/typing.py", line 677, in __call__\n            result.__orig_class__ = self\n        File "../returns/returns/primitives/types.py", line 42, in __setattr__\n            raise ImmutableStateError()\n        returns.primitives.exceptions.ImmutableStateError\n\n    See: https://github.com/dry-python/returns/issues/394\n\n    '
    RequiresContextIOResultE(lambda _: IOSuccess(1))