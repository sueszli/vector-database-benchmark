import io
from textwrap import dedent
import pytest
from rich import box
from rich.console import Console
from rich.rule import Rule
from rich.table import Table

@pytest.mark.parametrize('expand_kwarg', ({}, {'expand': False}))
def test_rule_in_unexpanded_table(expand_kwarg):
    if False:
        while True:
            i = 10
    console = Console(width=32, file=io.StringIO(), legacy_windows=False, _environ={})
    table = Table(box=box.ASCII, show_header=False, **expand_kwarg)
    table.add_column()
    table.add_column()
    table.add_row('COL1', 'COL2')
    table.add_row('COL1', Rule())
    table.add_row('COL1', 'COL2')
    console.print(table)
    expected = dedent('        +-------------+\n        | COL1 | COL2 |\n        | COL1 | ──── |\n        | COL1 | COL2 |\n        +-------------+\n        ')
    result = console.file.getvalue()
    assert result == expected

def test_rule_in_expanded_table():
    if False:
        while True:
            i = 10
    console = Console(width=32, file=io.StringIO(), legacy_windows=False, _environ={})
    table = Table(box=box.ASCII, expand=True, show_header=False)
    table.add_column()
    table.add_column()
    table.add_row('COL1', 'COL2')
    table.add_row('COL1', Rule(style=None))
    table.add_row('COL1', 'COL2')
    console.print(table)
    expected = dedent('        +------------------------------+\n        | COL1          | COL2         |\n        | COL1          | ──────────── |\n        | COL1          | COL2         |\n        +------------------------------+\n        ')
    result = console.file.getvalue()
    assert result == expected

def test_rule_in_ratio_table():
    if False:
        for i in range(10):
            print('nop')
    console = Console(width=32, file=io.StringIO(), legacy_windows=False, _environ={})
    table = Table(box=box.ASCII, expand=True, show_header=False)
    table.add_column(ratio=1)
    table.add_column()
    table.add_row('COL1', 'COL2')
    table.add_row('COL1', Rule(style=None))
    table.add_row('COL1', 'COL2')
    console.print(table)
    expected = dedent('        +------------------------------+\n        | COL1                  | COL2 |\n        | COL1                  | ──── |\n        | COL1                  | COL2 |\n        +------------------------------+\n        ')
    result = console.file.getvalue()
    assert result == expected