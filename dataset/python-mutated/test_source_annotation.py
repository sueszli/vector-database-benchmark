import pytest
from vyper.utils import annotate_source_code, indent
TEST_TEXT = '\ntest\nlines\nto\nindent\n'[1:-1]

def test_indent_indents_text():
    if False:
        i = 10
        return i + 15
    assert indent(TEST_TEXT, indent_chars='-', level=1) == '\n-test\n-lines\n-to\n-indent\n'[1:-1]
    assert indent(TEST_TEXT, indent_chars=' ', level=4) == '\n    test\n    lines\n    to\n    indent\n'[1:-1]
    assert indent(TEST_TEXT, indent_chars=[' ', '*', '-', '='], level=4) == '\n    test\n****lines\n----to\n====indent\n'[1:-1]

def test_indent_raises_value_errors():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError, match='Must provide indentation chars for each line'):
        indent(TEST_TEXT, indent_chars=[' '], level=1)
    with pytest.raises(ValueError, match='Unrecognized indentation characters value'):
        indent(TEST_TEXT, indent_chars=None, level=1)
TEST_SOURCE_CODE = "\n# Attempts to display the line and column of violating code.\nclass ParserException(Exception):\n    def __init__(self, message='Error Message not found.', item=None):\n        self.message = message\n        self.lineno = None\n        self.col_offset = None\n\n        if isinstance(item, tuple):  # is a position.\n            self.lineno, self.col_offset = item\n        elif item and hasattr(item, 'lineno'):\n            self.set_err_pos(item.lineno, item.col_offset)\n            if hasattr(item, 'source_code'):\n                self.source_code = item.source_code.splitlines()\n\n    def set_err_pos(self, lineno, col_offset):\n        if not self.lineno:\n            self.lineno = lineno\n\n            if not self.col_offset:\n                self.col_offset = col_offset\n\n    def __str__(self):\n        output = self.message\n\n        if self.lineno and hasattr(self, 'source_code'):\n\n            output = f'line {self.lineno}: {output}\\n{self.source_code[self.lineno -1]}'\n\n            if self.col_offset:\n                col = '-' * self.col_offset + '^'\n                output += '\\n' + col\n\n        elif self.lineno is not None and self.col_offset is not None:\n            output = f'line {self.lineno}:{self.col_offset} {output}'\n\n        return output\n"[1:-1]

def test_annotate_source_code_marks_positions_in_source_code():
    if False:
        while True:
            i = 10
    annotation = annotate_source_code(TEST_SOURCE_CODE, 22, col_offset=16, context_lines=0, line_numbers=False)
    assert annotation == '\n    def __str__(self):\n----------------^\n'[1:-1]
    annotation = annotate_source_code(TEST_SOURCE_CODE, 22, col_offset=15, context_lines=1, line_numbers=False)
    assert annotation == '\n\n    def __str__(self):\n---------------^\n        output = self.message\n'[1:-1]
    annotation = annotate_source_code(TEST_SOURCE_CODE, 22, col_offset=20, context_lines=2, line_numbers=False)
    assert annotation == '\n                self.col_offset = col_offset\n\n    def __str__(self):\n--------------------^\n        output = self.message\n\n'[1:-1]
    annotation = annotate_source_code(TEST_SOURCE_CODE, 1, col_offset=5, context_lines=3, line_numbers=True)
    assert annotation == "\n---> 1 # Attempts to display the line and column of violating code.\n------------^\n     2 class ParserException(Exception):\n     3     def __init__(self, message='Error Message not found.', item=None):\n     4         self.message = message\n"[1:-1]
    annotation = annotate_source_code(TEST_SOURCE_CODE, 36, col_offset=8, context_lines=4, line_numbers=True)
    assert annotation == "\n     32\n     33         elif self.lineno is not None and self.col_offset is not None:\n     34             output = f'line {self.lineno}:{self.col_offset} {output}'\n     35\n---> 36         return output\n----------------^\n"[1:-1]
    annotation = annotate_source_code(TEST_SOURCE_CODE, 15, col_offset=8, context_lines=11, line_numbers=True)
    assert annotation == "\n      4         self.message = message\n      5         self.lineno = None\n      6         self.col_offset = None\n      7\n      8         if isinstance(item, tuple):  # is a position.\n      9             self.lineno, self.col_offset = item\n     10         elif item and hasattr(item, 'lineno'):\n     11             self.set_err_pos(item.lineno, item.col_offset)\n     12             if hasattr(item, 'source_code'):\n     13                 self.source_code = item.source_code.splitlines()\n     14\n---> 15     def set_err_pos(self, lineno, col_offset):\n----------------^\n     16         if not self.lineno:\n     17             self.lineno = lineno\n     18\n     19             if not self.col_offset:\n     20                 self.col_offset = col_offset\n     21\n     22     def __str__(self):\n     23         output = self.message\n     24\n     25         if self.lineno and hasattr(self, 'source_code'):\n     26\n"[1:-1]
    annotation = annotate_source_code(TEST_SOURCE_CODE, 15, col_offset=None, context_lines=3, line_numbers=True)
    assert annotation == "\n     12             if hasattr(item, 'source_code'):\n     13                 self.source_code = item.source_code.splitlines()\n     14\n---> 15     def set_err_pos(self, lineno, col_offset):\n     16         if not self.lineno:\n     17             self.lineno = lineno\n     18\n"[1:-1]
    annotation = annotate_source_code(TEST_SOURCE_CODE, 15, col_offset=None, context_lines=2, line_numbers=False)
    assert annotation == '\n                self.source_code = item.source_code.splitlines()\n\n    def set_err_pos(self, lineno, col_offset):\n        if not self.lineno:\n            self.lineno = lineno\n'[1:-1]

@pytest.mark.parametrize('bad_lineno', (-100, -1, 0, 45, 1000))
def test_annotate_source_code_raises_value_errors(bad_lineno):
    if False:
        return 10
    with pytest.raises(ValueError, match='Line number is out of range'):
        annotate_source_code(TEST_SOURCE_CODE, bad_lineno)