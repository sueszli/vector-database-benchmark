import pytest
from dagster._check import CheckError
from dagster._utils.indenting_printer import IndentingPrinter, IndentingStringIoPrinter
LOREM_IPSUM = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum'
FORMATTED_LOREM = '# Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut\n# labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris\n# nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit\n# esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt\n# in culpa qui officia deserunt mollit anim id est laborum\n    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut\n    labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco\n    laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in\n    voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat\n    non proident, sunt in culpa qui officia deserunt mollit anim id est laborum\n        # Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt\n        # ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation\n        # ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in\n        # reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur\n        # sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id\n        # est laborum\n'

class CollectingIndentingPrinter(IndentingPrinter):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.lines = []

        def _add_line(text):
            if False:
                for i in range(10):
                    print('nop')
            if str is not None:
                self.lines.append(text)
        super(CollectingIndentingPrinter, self).__init__(*args, printer=_add_line, **kwargs)

    def result(self):
        if False:
            for i in range(10):
                print('nop')
        return '\n'.join(self.lines)

def create_printer(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    return CollectingIndentingPrinter(*args, **kwargs)

def test_basic_printer():
    if False:
        while True:
            i = 10
    printer = create_printer()
    printer.line('test')
    assert printer.result() == 'test'

def test_indent_printer():
    if False:
        return 10
    printer = create_printer()
    printer.line('test')
    with printer.with_indent():
        printer.line('test indent')
    with printer.with_indent('bop'):
        printer.line('another')
        printer.line('yet')
    assert printer.result() == 'test\n  test indent\nbop\n  another\n  yet'

def test_parameterized_indent():
    if False:
        return 10
    printer = create_printer(indent_level=4)
    printer.line('test')
    with printer.with_indent():
        printer.line('test indent')
    assert printer.result() == 'test\n    test indent'

def test_bad_decrease_indent():
    if False:
        while True:
            i = 10
    printer = create_printer(indent_level=4)
    with pytest.raises(Exception):
        printer.decrease_indent()

def test_indent_printer_blank_line():
    if False:
        print('Hello World!')
    printer = create_printer()
    printer.line('test')
    printer.blank_line()
    with printer.with_indent():
        printer.line('test indent')
    assert printer.result() == 'test\n\n  test indent'

def test_double_indent():
    if False:
        while True:
            i = 10
    printer = create_printer()
    printer.line('test')
    with printer.with_indent():
        printer.line('test indent')
        with printer.with_indent():
            printer.line('test double indent')
    assert printer.result() == 'test\n  test indent\n    test double indent'

def test_append():
    if False:
        while True:
            i = 10
    printer = create_printer()
    printer.append('a')
    printer.line('')
    assert printer.result() == 'a'

def test_double_append():
    if False:
        i = 10
        return i + 15
    printer = create_printer()
    printer.append('a')
    printer.append('a')
    printer.line('')
    assert printer.result() == 'aa'

def test_append_plus_line():
    if False:
        for i in range(10):
            print('nop')
    printer = create_printer()
    printer.append('a')
    printer.line('b')
    assert printer.result() == 'ab'

def test_blank_line_error():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(CheckError):
        printer = create_printer()
        printer.append('a')
        printer.blank_line()

def test_indenting_block_printer_context_management():
    if False:
        return 10
    with IndentingStringIoPrinter() as printer:
        printer.line('Hello, world!')
        assert printer.read() == 'Hello, world!\n'

def test_indenting_block_printer_block_printing():
    if False:
        while True:
            i = 10
    with IndentingStringIoPrinter(indent_level=4) as printer:
        printer.comment(LOREM_IPSUM)
        with printer.with_indent():
            printer.block(LOREM_IPSUM)
            with printer.with_indent():
                printer.comment(LOREM_IPSUM)
        assert printer.read() == FORMATTED_LOREM