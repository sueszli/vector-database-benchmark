from __future__ import annotations
import pytest
from pyupgrade._data import Settings
from pyupgrade._main import _fix_plugins

@pytest.mark.parametrize('s', (pytest.param('class Text: ...\ntext = Text()\n', id='not a type annotation'),))
def test_fix_typing_text_noop(s):
    if False:
        print('Hello World!')
    assert _fix_plugins(s, settings=Settings()) == s

@pytest.mark.parametrize(('s', 'expected'), (pytest.param('from typing import Text\nx: Text\n', 'from typing import Text\nx: str\n', id='from import of Text'), pytest.param('import typing\nx: typing.Text\n', 'import typing\nx: str\n', id='import of typing + typing.Text'), pytest.param('from typing import Text\nSomeAlias = Text\n', 'from typing import Text\nSomeAlias = str\n', id='not in a type annotation context')))
def test_fix_typing_text(s, expected):
    if False:
        while True:
            i = 10
    ret = _fix_plugins(s, settings=Settings())
    assert ret == expected