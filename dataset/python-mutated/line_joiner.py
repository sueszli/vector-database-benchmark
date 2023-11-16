"""Join logical lines together.

Determine how many lines can be joined into one line. For instance, we could
join these statements into one line:

  if a == 42:
    continue

like this:

  if a == 42: continue

There are a few restrictions:

  1. The lines should have been joined in the original source.
  2. The joined lines must not go over the column boundary if placed on the same
     line.
  3. They need to be very simple statements.

Note: Because we don't allow the use of a semicolon to separate statements, it
follows that there can only be at most two lines to join.
"""
from yapf.yapflib import style
_CLASS_OR_FUNC = frozenset({'def', 'class'})

def CanMergeMultipleLines(lines, last_was_merged=False):
    if False:
        print('Hello World!')
    'Determine if multiple lines can be joined into one.\n\n  Arguments:\n    lines: (list of LogicalLine) This is a splice of LogicalLines from the full\n      code base.\n    last_was_merged: (bool) The last line was merged.\n\n  Returns:\n    True if two consecutive lines can be joined together. In reality, this will\n    only happen if two consecutive lines can be joined, due to the style guide.\n  '
    indent_amt = lines[0].depth * style.Get('INDENT_WIDTH')
    if len(lines) == 1 or indent_amt > style.Get('COLUMN_LIMIT'):
        return False
    if len(lines) >= 3 and lines[2].depth >= lines[1].depth and (lines[0].depth != lines[2].depth):
        return False
    if lines[0].first.value in _CLASS_OR_FUNC:
        return False
    limit = style.Get('COLUMN_LIMIT') - indent_amt
    if lines[0].last.total_length < limit:
        limit -= lines[0].last.total_length
        if lines[0].first.value == 'if':
            return _CanMergeLineIntoIfStatement(lines, limit)
        if last_was_merged and lines[0].first.value in {'elif', 'else'}:
            return _CanMergeLineIntoIfStatement(lines, limit)
    return False

def _CanMergeLineIntoIfStatement(lines, limit):
    if False:
        i = 10
        return i + 15
    'Determine if we can merge a short if-then statement into one line.\n\n  Two lines of an if-then statement can be merged if they were that way in the\n  original source, fit on the line without going over the column limit, and are\n  considered "simple" statements --- typically statements like \'pass\',\n  \'continue\', and \'break\'.\n\n  Arguments:\n    lines: (list of LogicalLine) The lines we are wanting to merge.\n    limit: (int) The amount of space remaining on the line.\n\n  Returns:\n    True if the lines can be merged, False otherwise.\n  '
    if len(lines[1].tokens) == 1 and lines[1].last.is_multiline_string:
        return True
    if lines[0].lineno != lines[1].lineno:
        return False
    if lines[1].last.total_length >= limit:
        return False
    return style.Get('JOIN_MULTIPLE_LINES')