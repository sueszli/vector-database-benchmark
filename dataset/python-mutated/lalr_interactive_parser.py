from typing import Iterator, List
from copy import copy
import warnings
from lark.exceptions import UnexpectedToken
from lark.lexer import Token, LexerThread

class InteractiveParser:
    """InteractiveParser gives you advanced control over parsing and error handling when parsing with LALR.

    For a simpler interface, see the ``on_error`` argument to ``Lark.parse()``.
    """

    def __init__(self, parser, parser_state, lexer_thread: LexerThread):
        if False:
            print('Hello World!')
        self.parser = parser
        self.parser_state = parser_state
        self.lexer_thread = lexer_thread
        self.result = None

    @property
    def lexer_state(self) -> LexerThread:
        if False:
            while True:
                i = 10
        warnings.warn('lexer_state will be removed in subsequent releases. Use lexer_thread instead.', DeprecationWarning)
        return self.lexer_thread

    def feed_token(self, token: Token):
        if False:
            while True:
                i = 10
        'Feed the parser with a token, and advance it to the next state, as if it received it from the lexer.\n\n        Note that ``token`` has to be an instance of ``Token``.\n        '
        return self.parser_state.feed_token(token, token.type == '$END')

    def iter_parse(self) -> Iterator[Token]:
        if False:
            while True:
                i = 10
        'Step through the different stages of the parse, by reading tokens from the lexer\n        and feeding them to the parser, one per iteration.\n\n        Returns an iterator of the tokens it encounters.\n\n        When the parse is over, the resulting tree can be found in ``InteractiveParser.result``.\n        '
        for token in self.lexer_thread.lex(self.parser_state):
            yield token
            self.result = self.feed_token(token)

    def exhaust_lexer(self) -> List[Token]:
        if False:
            while True:
                i = 10
        "Try to feed the rest of the lexer state into the interactive parser.\n\n        Note that this modifies the instance in place and does not feed an '$END' Token\n        "
        return list(self.iter_parse())

    def feed_eof(self, last_token=None):
        if False:
            return 10
        "Feed a '$END' Token. Borrows from 'last_token' if given."
        eof = Token.new_borrow_pos('$END', '', last_token) if last_token is not None else self.lexer_thread._Token('$END', '', 0, 1, 1)
        return self.feed_token(eof)

    def __copy__(self):
        if False:
            while True:
                i = 10
        "Create a new interactive parser with a separate state.\n\n        Calls to feed_token() won't affect the old instance, and vice-versa.\n        "
        return type(self)(self.parser, copy(self.parser_state), copy(self.lexer_thread))

    def copy(self):
        if False:
            while True:
                i = 10
        return copy(self)

    def __eq__(self, other):
        if False:
            return 10
        if not isinstance(other, InteractiveParser):
            return False
        return self.parser_state == other.parser_state and self.lexer_thread == other.lexer_thread

    def as_immutable(self):
        if False:
            for i in range(10):
                print('nop')
        'Convert to an ``ImmutableInteractiveParser``.'
        p = copy(self)
        return ImmutableInteractiveParser(p.parser, p.parser_state, p.lexer_thread)

    def pretty(self):
        if False:
            while True:
                i = 10
        "Print the output of ``choices()`` in a way that's easier to read."
        out = ['Parser choices:']
        for (k, v) in self.choices().items():
            out.append('\t- %s -> %r' % (k, v))
        out.append('stack size: %s' % len(self.parser_state.state_stack))
        return '\n'.join(out)

    def choices(self):
        if False:
            i = 10
            return i + 15
        'Returns a dictionary of token types, matched to their action in the parser.\n\n        Only returns token types that are accepted by the current state.\n\n        Updated by ``feed_token()``.\n        '
        return self.parser_state.parse_conf.parse_table.states[self.parser_state.position]

    def accepts(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the set of possible tokens that will advance the parser into a new valid state.'
        accepts = set()
        conf_no_callbacks = copy(self.parser_state.parse_conf)
        conf_no_callbacks.callbacks = {}
        for t in self.choices():
            if t.isupper():
                new_cursor = copy(self)
                new_cursor.parser_state.parse_conf = conf_no_callbacks
                try:
                    new_cursor.feed_token(self.lexer_thread._Token(t, ''))
                except UnexpectedToken:
                    pass
                else:
                    accepts.add(t)
        return accepts

    def resume_parse(self):
        if False:
            print('Hello World!')
        'Resume automated parsing from the current state.\n        '
        return self.parser.parse_from_state(self.parser_state, last_token=self.lexer_thread.state.last_token)

class ImmutableInteractiveParser(InteractiveParser):
    """Same as ``InteractiveParser``, but operations create a new instance instead
    of changing it in-place.
    """
    result = None

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash((self.parser_state, self.lexer_thread))

    def feed_token(self, token):
        if False:
            for i in range(10):
                print('nop')
        c = copy(self)
        c.result = InteractiveParser.feed_token(c, token)
        return c

    def exhaust_lexer(self):
        if False:
            for i in range(10):
                print('nop')
        "Try to feed the rest of the lexer state into the parser.\n\n        Note that this returns a new ImmutableInteractiveParser and does not feed an '$END' Token"
        cursor = self.as_mutable()
        cursor.exhaust_lexer()
        return cursor.as_immutable()

    def as_mutable(self):
        if False:
            print('Hello World!')
        'Convert to an ``InteractiveParser``.'
        p = copy(self)
        return InteractiveParser(p.parser, p.parser_state, p.lexer_thread)