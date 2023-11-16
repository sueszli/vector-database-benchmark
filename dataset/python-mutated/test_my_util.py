from __future__ import annotations
from .....plugins.module_utils.my_util import hello

def test_hello():
    if False:
        for i in range(10):
            print('nop')
    assert hello('Ansibull') == 'Hello Ansibull'