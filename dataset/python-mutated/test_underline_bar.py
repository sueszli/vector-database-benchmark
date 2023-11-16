from unittest.mock import create_autospec
from rich.console import Console, ConsoleOptions
from rich.text import Text
from tests.utilities.render import render
from textual.renderables.bar import Bar
MAGENTA = '\x1b[35m'
GREY = '\x1b[38;5;59m'
STOP = '\x1b[0m'
GREEN = '\x1b[32m'
RED = '\x1b[31m'

def test_no_highlight():
    if False:
        print('Hello World!')
    bar = Bar(width=6)
    assert render(bar) == f'{GREY}━━━━━━{STOP}'

def test_highlight_from_zero():
    if False:
        while True:
            i = 10
    bar = Bar(highlight_range=(0, 2.5), width=6)
    assert render(bar) == f'{MAGENTA}━━{STOP}{MAGENTA}╸{STOP}{GREY}━━━{STOP}'

def test_highlight_from_zero_point_five():
    if False:
        for i in range(10):
            print('nop')
    bar = Bar(highlight_range=(0.5, 2), width=6)
    assert render(bar) == f'{MAGENTA}╺━{STOP}{GREY}╺{STOP}{GREY}━━━{STOP}'

def test_highlight_middle():
    if False:
        print('Hello World!')
    bar = Bar(highlight_range=(2, 4), width=6)
    assert render(bar) == f'{GREY}━{STOP}{GREY}╸{STOP}{MAGENTA}━━{STOP}{GREY}╺{STOP}{GREY}━{STOP}'

def test_highlight_half_start():
    if False:
        i = 10
        return i + 15
    bar = Bar(highlight_range=(2.5, 4), width=6)
    assert render(bar) == f'{GREY}━━{STOP}{MAGENTA}╺━{STOP}{GREY}╺{STOP}{GREY}━{STOP}'

def test_highlight_half_end():
    if False:
        for i in range(10):
            print('nop')
    bar = Bar(highlight_range=(2, 4.5), width=6)
    assert render(bar) == f'{GREY}━{STOP}{GREY}╸{STOP}{MAGENTA}━━{STOP}{MAGENTA}╸{STOP}{GREY}━{STOP}'

def test_highlight_half_start_and_half_end():
    if False:
        while True:
            i = 10
    bar = Bar(highlight_range=(2.5, 4.5), width=6)
    assert render(bar) == f'{GREY}━━{STOP}{MAGENTA}╺━{STOP}{MAGENTA}╸{STOP}{GREY}━{STOP}'

def test_highlight_to_near_end():
    if False:
        for i in range(10):
            print('nop')
    bar = Bar(highlight_range=(3, 5.5), width=6)
    assert render(bar) == f'{GREY}━━{STOP}{GREY}╸{STOP}{MAGENTA}━━{STOP}{MAGENTA}╸{STOP}'

def test_highlight_to_end():
    if False:
        for i in range(10):
            print('nop')
    bar = Bar(highlight_range=(3, 6), width=6)
    assert render(bar) == f'{GREY}━━{STOP}{GREY}╸{STOP}{MAGENTA}━━━{STOP}'

def test_highlight_out_of_bounds_start():
    if False:
        i = 10
        return i + 15
    bar = Bar(highlight_range=(-2, 3), width=6)
    assert render(bar) == f'{MAGENTA}━━━{STOP}{GREY}╺{STOP}{GREY}━━{STOP}'

def test_highlight_out_of_bounds_end():
    if False:
        while True:
            i = 10
    bar = Bar(highlight_range=(3, 9), width=6)
    assert render(bar) == f'{GREY}━━{STOP}{GREY}╸{STOP}{MAGENTA}━━━{STOP}'

def test_highlight_full_range_out_of_bounds_end():
    if False:
        i = 10
        return i + 15
    bar = Bar(highlight_range=(9, 10), width=6)
    assert render(bar) == f'{GREY}━━━━━━{STOP}'

def test_highlight_full_range_out_of_bounds_start():
    if False:
        print('Hello World!')
    bar = Bar(highlight_range=(-5, -2), width=6)
    assert render(bar) == f'{GREY}━━━━━━{STOP}'

def test_custom_styles():
    if False:
        return 10
    bar = Bar(highlight_range=(2, 4), highlight_style='red', background_style='green', width=6)
    assert render(bar) == f'{GREEN}━{STOP}{GREEN}╸{STOP}{RED}━━{STOP}{GREEN}╺{STOP}{GREEN}━{STOP}'

def test_clickable_ranges():
    if False:
        print('Hello World!')
    bar = Bar(highlight_range=(0, 1), width=6, clickable_ranges={'foo': (0, 2), 'bar': (4, 5)})
    console = create_autospec(Console)
    options = create_autospec(ConsoleOptions)
    text: Text = list(bar.__rich_console__(console, options))[0]
    (start, end, style) = text.spans[-2]
    assert (start, end) == (0, 2)
    assert style.meta == {'@click': "range_clicked('foo')"}
    (start, end, style) = text.spans[-1]
    assert (start, end) == (4, 5)
    assert style.meta == {'@click': "range_clicked('bar')"}