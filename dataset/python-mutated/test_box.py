import pytest
from rich.console import ConsoleOptions, ConsoleDimensions
from rich.box import ASCII, DOUBLE, ROUNDED, HEAVY, SQUARE

def test_str():
    if False:
        while True:
            i = 10
    assert str(ASCII) == '+--+\n| ||\n|-+|\n| ||\n|-+|\n|-+|\n| ||\n+--+\n'

def test_repr():
    if False:
        for i in range(10):
            print('nop')
    assert repr(ASCII) == 'Box(...)'

def test_get_top():
    if False:
        i = 10
        return i + 15
    top = HEAVY.get_top(widths=[1, 2])
    assert top == '┏━┳━━┓'

def test_get_row():
    if False:
        return 10
    head_row = DOUBLE.get_row(widths=[3, 2, 1], level='head')
    assert head_row == '╠═══╬══╬═╣'
    row = ASCII.get_row(widths=[1, 2, 3], level='row')
    assert row == '|-+--+---|'
    foot_row = ROUNDED.get_row(widths=[2, 1, 3], level='foot')
    assert foot_row == '├──┼─┼───┤'
    with pytest.raises(ValueError):
        ROUNDED.get_row(widths=[1, 2, 3], level='FOO')

def test_get_bottom():
    if False:
        while True:
            i = 10
    bottom = HEAVY.get_bottom(widths=[1, 2, 3])
    assert bottom == '┗━┻━━┻━━━┛'

def test_box_substitute():
    if False:
        while True:
            i = 10
    options = ConsoleOptions(ConsoleDimensions(80, 25), legacy_windows=True, min_width=1, max_width=100, is_terminal=True, encoding='utf-8', max_height=25)
    assert HEAVY.substitute(options) == SQUARE
    options.legacy_windows = False
    assert HEAVY.substitute(options) == HEAVY
    options.encoding = 'ascii'
    assert HEAVY.substitute(options) == ASCII