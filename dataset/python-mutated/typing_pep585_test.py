from __future__ import annotations
import pytest
from pyupgrade._data import Settings
from pyupgrade._main import _fix_plugins

@pytest.mark.parametrize(('s', 'version'), (pytest.param('x = lambda foo: None', (3, 9), id='lambdas do not have type annotations'), pytest.param('from typing import List\nx: List[int]\n', (3, 8), id='not python 3.9+'), pytest.param('from __future__ import annotations\nfrom typing import List\nSomeAlias = List[int]\n', (3, 8), id='not in a type annotation context'), pytest.param('from typing import Union\nx: Union[int, str]\n', (3, 9), id='not a PEP 585 type')))
def test_fix_generic_types_noop(s, version):
    if False:
        while True:
            i = 10
    assert _fix_plugins(s, settings=Settings(min_version=version)) == s

def test_noop_keep_runtime_typing():
    if False:
        for i in range(10):
            print('nop')
    s = 'from __future__ import annotations\nfrom typing import List\ndef f(x: List[str]) -> None: ...\n'
    assert _fix_plugins(s, settings=Settings(keep_runtime_typing=True)) == s

def test_keep_runtime_typing_ignored_in_py39():
    if False:
        print('Hello World!')
    s = 'from __future__ import annotations\nfrom typing import List\ndef f(x: List[str]) -> None: ...\n'
    expected = 'from __future__ import annotations\nfrom typing import List\ndef f(x: list[str]) -> None: ...\n'
    settings = Settings(min_version=(3, 9), keep_runtime_typing=True)
    assert _fix_plugins(s, settings=settings) == expected

@pytest.mark.parametrize(('s', 'expected'), (pytest.param('from typing import List\nx: List[int]\n', 'from typing import List\nx: list[int]\n', id='from import of List'), pytest.param('import typing\nx: typing.List[int]\n', 'import typing\nx: list[int]\n', id='import of typing + typing.List'), pytest.param('from typing import List\nSomeAlias = List[int]\n', 'from typing import List\nSomeAlias = list[int]\n', id='not in a type annotation context')))
def test_fix_generic_types(s, expected):
    if False:
        return 10
    ret = _fix_plugins(s, settings=Settings(min_version=(3, 9)))
    assert ret == expected

@pytest.mark.parametrize(('s', 'expected'), (pytest.param('from __future__ import annotations\nfrom typing import List\nx: List[int]\n', 'from __future__ import annotations\nfrom typing import List\nx: list[int]\n', id='variable annotations'), pytest.param('from __future__ import annotations\nfrom typing import List\ndef f(x: List[int]) -> None: ...\n', 'from __future__ import annotations\nfrom typing import List\ndef f(x: list[int]) -> None: ...\n', id='argument annotations'), pytest.param('from __future__ import annotations\nfrom typing import List\ndef f() -> List[int]: ...\n', 'from __future__ import annotations\nfrom typing import List\ndef f() -> list[int]: ...\n', id='return annotations')))
def test_fix_generic_types_future_annotations(s, expected):
    if False:
        for i in range(10):
            print('nop')
    ret = _fix_plugins(s, settings=Settings())
    assert ret == expected