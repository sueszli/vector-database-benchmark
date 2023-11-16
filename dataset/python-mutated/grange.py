"""DNS GENERATE range conversion."""
from typing import Tuple
import dns

def from_text(text: str) -> Tuple[int, int, int]:
    if False:
        for i in range(10):
            print('nop')
    'Convert the text form of a range in a ``$GENERATE`` statement to an\n    integer.\n\n    *text*, a ``str``, the textual range in ``$GENERATE`` form.\n\n    Returns a tuple of three ``int`` values ``(start, stop, step)``.\n    '
    start = -1
    stop = -1
    step = 1
    cur = ''
    state = 0
    if text and text[0] == '-':
        raise dns.exception.SyntaxError('Start cannot be a negative number')
    for c in text:
        if c == '-' and state == 0:
            start = int(cur)
            cur = ''
            state = 1
        elif c == '/':
            stop = int(cur)
            cur = ''
            state = 2
        elif c.isdigit():
            cur += c
        else:
            raise dns.exception.SyntaxError('Could not parse %s' % c)
    if state == 0:
        raise dns.exception.SyntaxError('no stop value specified')
    elif state == 1:
        stop = int(cur)
    else:
        assert state == 2
        step = int(cur)
    assert step >= 1
    assert start >= 0
    if start > stop:
        raise dns.exception.SyntaxError('start must be <= stop')
    return (start, stop, step)