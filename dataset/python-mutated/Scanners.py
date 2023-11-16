"""
Python Lexical Analyser

Scanning an input stream
"""
from __future__ import absolute_import
import cython
cython.declare(BOL=object, EOL=object, EOF=object, NOT_FOUND=object)
from . import Errors
from .Regexps import BOL, EOL, EOF
NOT_FOUND = object()

class Scanner(object):
    """
    A Scanner is used to read tokens from a stream of characters
    using the token set specified by a Plex.Lexicon.

    Constructor:

      Scanner(lexicon, stream, name = '')

        See the docstring of the __init__ method for details.

    Methods:

      See the docstrings of the individual methods for more
      information.

      read() --> (value, text)
        Reads the next lexical token from the stream.

      position() --> (name, line, col)
        Returns the position of the last token read using the
        read() method.

      begin(state_name)
        Causes scanner to change state.

      produce(value [, text])
        Causes return of a token value to the caller of the
        Scanner.

    """

    def __init__(self, lexicon, stream, name='', initial_pos=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Scanner(lexicon, stream, name = '')\n\n          |lexicon| is a Plex.Lexicon instance specifying the lexical tokens\n          to be recognised.\n\n          |stream| can be a file object or anything which implements a\n          compatible read() method.\n\n          |name| is optional, and may be the name of the file being\n          scanned or any other identifying string.\n        "
        self.trace = 0
        self.buffer = u''
        self.buf_start_pos = 0
        self.next_pos = 0
        self.cur_pos = 0
        self.cur_line = 1
        self.start_pos = 0
        self.current_scanner_position_tuple = ('', 0, 0)
        self.last_token_position_tuple = ('', 0, 0)
        self.text = None
        self.state_name = None
        self.lexicon = lexicon
        self.stream = stream
        self.name = name
        self.queue = []
        self.initial_state = None
        self.begin('')
        self.next_pos = 0
        self.cur_pos = 0
        self.cur_line_start = 0
        self.cur_char = BOL
        self.input_state = 1
        if initial_pos is not None:
            (self.cur_line, self.cur_line_start) = (initial_pos[1], -initial_pos[2])

    def read(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Read the next lexical token from the stream and return a\n        tuple (value, text), where |value| is the value associated with\n        the token as specified by the Lexicon, and |text| is the actual\n        string read from the stream. Returns (None, '') on end of file.\n        "
        queue = self.queue
        while not queue:
            (self.text, action) = self.scan_a_token()
            if action is None:
                self.produce(None)
                self.eof()
            else:
                value = action.perform(self, self.text)
                if value is not None:
                    self.produce(value)
        (result, self.last_token_position_tuple) = queue[0]
        del queue[0]
        return result

    def unread(self, token, value, position):
        if False:
            for i in range(10):
                print('nop')
        self.queue.insert(0, ((token, value), position))

    def get_current_scan_pos(self):
        if False:
            i = 10
            return i + 15
        return self.current_scanner_position_tuple

    def scan_a_token(self):
        if False:
            return 10
        "\n        Read the next input sequence recognised by the machine\n        and return (text, action). Returns ('', None) on end of\n        file.\n        "
        self.start_pos = self.cur_pos
        self.current_scanner_position_tuple = (self.name, self.cur_line, self.cur_pos - self.cur_line_start)
        action = self.run_machine_inlined()
        if action is not None:
            if self.trace:
                print('Scanner: read: Performing %s %d:%d' % (action, self.start_pos, self.cur_pos))
            text = self.buffer[self.start_pos - self.buf_start_pos:self.cur_pos - self.buf_start_pos]
            return (text, action)
        else:
            if self.cur_pos == self.start_pos:
                if self.cur_char is EOL:
                    self.next_char()
                if self.cur_char is None or self.cur_char is EOF:
                    return (u'', None)
            raise Errors.UnrecognizedInput(self, self.state_name)

    def run_machine_inlined(self):
        if False:
            while True:
                i = 10
        '\n        Inlined version of run_machine for speed.\n        '
        state = self.initial_state
        cur_pos = self.cur_pos
        cur_line = self.cur_line
        cur_line_start = self.cur_line_start
        cur_char = self.cur_char
        input_state = self.input_state
        next_pos = self.next_pos
        buffer = self.buffer
        buf_start_pos = self.buf_start_pos
        buf_len = len(buffer)
        (b_action, b_cur_pos, b_cur_line, b_cur_line_start, b_cur_char, b_input_state, b_next_pos) = (None, 0, 0, 0, u'', 0, 0)
        trace = self.trace
        while 1:
            if trace:
                print('State %d, %d/%d:%s -->' % (state['number'], input_state, cur_pos, repr(cur_char)))
            action = state['action']
            if action is not None:
                (b_action, b_cur_pos, b_cur_line, b_cur_line_start, b_cur_char, b_input_state, b_next_pos) = (action, cur_pos, cur_line, cur_line_start, cur_char, input_state, next_pos)
            c = cur_char
            new_state = state.get(c, NOT_FOUND)
            if new_state is NOT_FOUND:
                new_state = c and state.get('else')
            if new_state:
                if trace:
                    print('State %d' % new_state['number'])
                state = new_state
                if input_state == 1:
                    cur_pos = next_pos
                    buf_index = next_pos - buf_start_pos
                    if buf_index < buf_len:
                        c = buffer[buf_index]
                        next_pos += 1
                    else:
                        discard = self.start_pos - buf_start_pos
                        data = self.stream.read(4096)
                        buffer = self.buffer[discard:] + data
                        self.buffer = buffer
                        buf_start_pos += discard
                        self.buf_start_pos = buf_start_pos
                        buf_len = len(buffer)
                        buf_index -= discard
                        if data:
                            c = buffer[buf_index]
                            next_pos += 1
                        else:
                            c = u''
                    if c == u'\n':
                        cur_char = EOL
                        input_state = 2
                    elif not c:
                        cur_char = EOL
                        input_state = 4
                    else:
                        cur_char = c
                elif input_state == 2:
                    cur_char = u'\n'
                    input_state = 3
                elif input_state == 3:
                    cur_line += 1
                    cur_line_start = cur_pos = next_pos
                    cur_char = BOL
                    input_state = 1
                elif input_state == 4:
                    cur_char = EOF
                    input_state = 5
                else:
                    cur_char = u''
            else:
                if trace:
                    print('blocked')
                if b_action is not None:
                    (action, cur_pos, cur_line, cur_line_start, cur_char, input_state, next_pos) = (b_action, b_cur_pos, b_cur_line, b_cur_line_start, b_cur_char, b_input_state, b_next_pos)
                else:
                    action = None
                break
        self.cur_pos = cur_pos
        self.cur_line = cur_line
        self.cur_line_start = cur_line_start
        self.cur_char = cur_char
        self.input_state = input_state
        self.next_pos = next_pos
        if trace:
            if action is not None:
                print('Doing %s' % action)
        return action

    def next_char(self):
        if False:
            while True:
                i = 10
        input_state = self.input_state
        if self.trace:
            print('Scanner: next: %s [%d] %d' % (' ' * 20, input_state, self.cur_pos))
        if input_state == 1:
            self.cur_pos = self.next_pos
            c = self.read_char()
            if c == u'\n':
                self.cur_char = EOL
                self.input_state = 2
            elif not c:
                self.cur_char = EOL
                self.input_state = 4
            else:
                self.cur_char = c
        elif input_state == 2:
            self.cur_char = u'\n'
            self.input_state = 3
        elif input_state == 3:
            self.cur_line += 1
            self.cur_line_start = self.cur_pos = self.next_pos
            self.cur_char = BOL
            self.input_state = 1
        elif input_state == 4:
            self.cur_char = EOF
            self.input_state = 5
        else:
            self.cur_char = u''
        if self.trace:
            print('--> [%d] %d %r' % (input_state, self.cur_pos, self.cur_char))

    def position(self):
        if False:
            return 10
        '\n        Return a tuple (name, line, col) representing the location of\n        the last token read using the read() method. |name| is the\n        name that was provided to the Scanner constructor; |line|\n        is the line number in the stream (1-based); |col| is the\n        position within the line of the first character of the token\n        (0-based).\n        '
        return self.last_token_position_tuple

    def get_position(self):
        if False:
            print('Hello World!')
        '\n        Python accessible wrapper around position(), only for error reporting.\n        '
        return self.position()

    def begin(self, state_name):
        if False:
            i = 10
            return i + 15
        'Set the current state of the scanner to the named state.'
        self.initial_state = self.lexicon.get_initial_state(state_name)
        self.state_name = state_name

    def produce(self, value, text=None):
        if False:
            print('Hello World!')
        '\n        Called from an action procedure, causes |value| to be returned\n        as the token value from read(). If |text| is supplied, it is\n        returned in place of the scanned text.\n\n        produce() can be called more than once during a single call to an action\n        procedure, in which case the tokens are queued up and returned one\n        at a time by subsequent calls to read(), until the queue is empty,\n        whereupon scanning resumes.\n        '
        if text is None:
            text = self.text
        self.queue.append(((value, text), self.current_scanner_position_tuple))

    def eof(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Override this method if you want something to be done at\n        end of file.\n        '
        pass

    @property
    def start_line(self):
        if False:
            print('Hello World!')
        return self.last_token_position_tuple[1]