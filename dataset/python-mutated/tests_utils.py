from ast import literal_eval
from collections import defaultdict
from typing import Union
from tqdm.utils import envwrap

def test_envwrap(monkeypatch):
    if False:
        while True:
            i = 10
    'Test @envwrap (basic)'
    monkeypatch.setenv('FUNC_A', '42')
    monkeypatch.setenv('FUNC_TyPe_HiNt', '1337')
    monkeypatch.setenv('FUNC_Unused', 'x')

    @envwrap('FUNC_')
    def func(a=1, b=2, type_hint: int=None):
        if False:
            for i in range(10):
                print('nop')
        return (a, b, type_hint)
    assert (42, 2, 1337) == func()
    assert (99, 2, 1337) == func(a=99)

def test_envwrap_types(monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    'Test @envwrap(types)'
    monkeypatch.setenv('FUNC_notype', '3.14159')

    @envwrap('FUNC_', types=defaultdict(lambda : literal_eval))
    def func(notype=None):
        if False:
            while True:
                i = 10
        return notype
    assert 3.14159 == func()
    monkeypatch.setenv('FUNC_number', '1')
    monkeypatch.setenv('FUNC_string', '1')

    @envwrap('FUNC_', types={'number': int})
    def nofallback(number=None, string=None):
        if False:
            i = 10
            return i + 15
        return (number, string)
    assert 1, '1' == nofallback()

def test_envwrap_annotations(monkeypatch):
    if False:
        i = 10
        return i + 15
    'Test @envwrap with typehints'
    monkeypatch.setenv('FUNC_number', '1.1')
    monkeypatch.setenv('FUNC_string', '1.1')

    @envwrap('FUNC_')
    def annotated(number: Union[int, float]=None, string: int=None):
        if False:
            while True:
                i = 10
        return (number, string)
    assert 1.1, '1.1' == annotated()