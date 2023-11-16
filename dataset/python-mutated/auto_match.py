"""
Utilities function for keybinding with prompt toolkit.

This will be bound to specific key press and filter modes,
like whether we are in edit mode, and whether the completer is open.
"""
import re
from prompt_toolkit.key_binding import KeyPressEvent

def parenthesis(event: KeyPressEvent):
    if False:
        for i in range(10):
            print('nop')
    'Auto-close parenthesis'
    event.current_buffer.insert_text('()')
    event.current_buffer.cursor_left()

def brackets(event: KeyPressEvent):
    if False:
        while True:
            i = 10
    'Auto-close brackets'
    event.current_buffer.insert_text('[]')
    event.current_buffer.cursor_left()

def braces(event: KeyPressEvent):
    if False:
        for i in range(10):
            print('nop')
    'Auto-close braces'
    event.current_buffer.insert_text('{}')
    event.current_buffer.cursor_left()

def double_quote(event: KeyPressEvent):
    if False:
        i = 10
        return i + 15
    'Auto-close double quotes'
    event.current_buffer.insert_text('""')
    event.current_buffer.cursor_left()

def single_quote(event: KeyPressEvent):
    if False:
        for i in range(10):
            print('nop')
    'Auto-close single quotes'
    event.current_buffer.insert_text("''")
    event.current_buffer.cursor_left()

def docstring_double_quotes(event: KeyPressEvent):
    if False:
        while True:
            i = 10
    'Auto-close docstring (double quotes)'
    event.current_buffer.insert_text('""""')
    event.current_buffer.cursor_left(3)

def docstring_single_quotes(event: KeyPressEvent):
    if False:
        while True:
            i = 10
    'Auto-close docstring (single quotes)'
    event.current_buffer.insert_text("''''")
    event.current_buffer.cursor_left(3)

def raw_string_parenthesis(event: KeyPressEvent):
    if False:
        while True:
            i = 10
    'Auto-close parenthesis in raw strings'
    matches = re.match('.*(r|R)[\\"\'](-*)', event.current_buffer.document.current_line_before_cursor)
    dashes = matches.group(2) if matches else ''
    event.current_buffer.insert_text('()' + dashes)
    event.current_buffer.cursor_left(len(dashes) + 1)

def raw_string_bracket(event: KeyPressEvent):
    if False:
        print('Hello World!')
    'Auto-close bracker in raw strings'
    matches = re.match('.*(r|R)[\\"\'](-*)', event.current_buffer.document.current_line_before_cursor)
    dashes = matches.group(2) if matches else ''
    event.current_buffer.insert_text('[]' + dashes)
    event.current_buffer.cursor_left(len(dashes) + 1)

def raw_string_braces(event: KeyPressEvent):
    if False:
        print('Hello World!')
    'Auto-close braces in raw strings'
    matches = re.match('.*(r|R)[\\"\'](-*)', event.current_buffer.document.current_line_before_cursor)
    dashes = matches.group(2) if matches else ''
    event.current_buffer.insert_text('{}' + dashes)
    event.current_buffer.cursor_left(len(dashes) + 1)

def skip_over(event: KeyPressEvent):
    if False:
        i = 10
        return i + 15
    'Skip over automatically added parenthesis/quote.\n\n    (rather than adding another parenthesis/quote)'
    event.current_buffer.cursor_right()

def delete_pair(event: KeyPressEvent):
    if False:
        print('Hello World!')
    'Delete auto-closed parenthesis'
    event.current_buffer.delete()
    event.current_buffer.delete_before_cursor()
auto_match_parens = {'(': parenthesis, '[': brackets, '{': braces}
auto_match_parens_raw_string = {'(': raw_string_parenthesis, '[': raw_string_bracket, '{': raw_string_braces}