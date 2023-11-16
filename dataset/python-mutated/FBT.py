def function(posonly_nohint, posonly_nonboolhint: int, posonly_boolhint: bool, posonly_boolstrhint: 'bool', /, offset, posorkw_nonvalued_nohint, posorkw_nonvalued_nonboolhint: int, posorkw_nonvalued_boolhint: bool, posorkw_nonvalued_boolstrhint: 'bool', posorkw_boolvalued_nohint=True, posorkw_boolvalued_nonboolhint: int=True, posorkw_boolvalued_boolhint: bool=True, posorkw_boolvalued_boolstrhint: 'bool'=True, posorkw_nonboolvalued_nohint=1, posorkw_nonboolvalued_nonboolhint: int=2, posorkw_nonboolvalued_boolhint: bool=3, posorkw_nonboolvalued_boolstrhint: 'bool'=4, *, kwonly_nonvalued_nohint, kwonly_nonvalued_nonboolhint: int, kwonly_nonvalued_boolhint: bool, kwonly_nonvalued_boolstrhint: 'bool', kwonly_boolvalued_nohint=True, kwonly_boolvalued_nonboolhint: int=False, kwonly_boolvalued_boolhint: bool=True, kwonly_boolvalued_boolstrhint: 'bool'=True, kwonly_nonboolvalued_nohint=5, kwonly_nonboolvalued_nonboolhint: int=1, kwonly_nonboolvalued_boolhint: bool=1, kwonly_nonboolvalued_boolstrhint: 'bool'=1, **kw):
    if False:
        i = 10
        return i + 15
    ...

def used(do):
    if False:
        while True:
            i = 10
    return do
used('a', True)
used(do=True)
'\nFBT003 Boolean positional value on dict\n'
a = {'a': 'b'}
a.get('hello', False)
{}.get('hello', False)
{}.setdefault('hello', True)
{}.pop('hello', False)
{}.pop(True, False)
dict.fromkeys(('world',), True)
{}.deploy(True, False)
getattr(someobj, attrname, False)
mylist.index(True)
bool(False)
int(True)
str(int(False))
cfg.get('hello', True)
cfg.getint('hello', True)
cfg.getfloat('hello', True)
cfg.getboolean('hello', True)
os.set_blocking(0, False)
g_action.set_enabled(True)
settings.set_enable_developer_extras(True)
foo.is_(True)
bar.is_not(False)
next(iter([]), False)
sa.func.coalesce(tbl.c.valid, False)

class Registry:

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        self._switches = [False] * len(Switch)

    def __setitem__(self, switch: Switch, value: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._switches[switch.value] = value

    @foo.setter
    def foo(self, value: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def foo(self, value: bool) -> None:
        if False:
            print('Hello World!')
        pass

    def foo(self) -> None:
        if False:
            print('Hello World!')
        object.__setattr__(self, 'flag', True)
from typing import Optional, Union

def func(x: Union[list, Optional[int | str | float | bool]]):
    if False:
        return 10
    pass

def func(x: bool | str):
    if False:
        for i in range(10):
            print('nop')
    pass

def func(x: int | str):
    if False:
        print('Hello World!')
    pass