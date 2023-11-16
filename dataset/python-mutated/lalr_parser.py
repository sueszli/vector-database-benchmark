"""This module implements a LALR(1) Parser
"""
from typing import Dict, Any, Optional
from ..lexer import Token, LexerThread
from ..utils import Serialize
from ..common import ParserConf, ParserCallbacks
from .lalr_analysis import LALR_Analyzer, IntParseTable, ParseTableBase
from .lalr_interactive_parser import InteractiveParser
from lark.exceptions import UnexpectedCharacters, UnexpectedInput, UnexpectedToken
from .lalr_parser_state import ParserState, ParseConf

class LALR_Parser(Serialize):

    def __init__(self, parser_conf: ParserConf, debug: bool=False, strict: bool=False):
        if False:
            print('Hello World!')
        analysis = LALR_Analyzer(parser_conf, debug=debug, strict=strict)
        analysis.compute_lalr()
        callbacks = parser_conf.callbacks
        self._parse_table = analysis.parse_table
        self.parser_conf = parser_conf
        self.parser = _Parser(analysis.parse_table, callbacks, debug)

    @classmethod
    def deserialize(cls, data, memo, callbacks, debug=False):
        if False:
            for i in range(10):
                print('nop')
        inst = cls.__new__(cls)
        inst._parse_table = IntParseTable.deserialize(data, memo)
        inst.parser = _Parser(inst._parse_table, callbacks, debug)
        return inst

    def serialize(self, memo: Any=None) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        return self._parse_table.serialize(memo)

    def parse_interactive(self, lexer: LexerThread, start: str):
        if False:
            return 10
        return self.parser.parse(lexer, start, start_interactive=True)

    def parse(self, lexer, start, on_error=None):
        if False:
            return 10
        try:
            return self.parser.parse(lexer, start)
        except UnexpectedInput as e:
            if on_error is None:
                raise
            while True:
                if isinstance(e, UnexpectedCharacters):
                    s = e.interactive_parser.lexer_thread.state
                    p = s.line_ctr.char_pos
                if not on_error(e):
                    raise e
                if isinstance(e, UnexpectedCharacters):
                    if p == s.line_ctr.char_pos:
                        s.line_ctr.feed(s.text[p:p + 1])
                try:
                    return e.interactive_parser.resume_parse()
                except UnexpectedToken as e2:
                    if isinstance(e, UnexpectedToken) and e.token.type == e2.token.type == '$END' and (e.interactive_parser == e2.interactive_parser):
                        raise e2
                    e = e2
                except UnexpectedCharacters as e2:
                    e = e2

class _Parser:
    parse_table: ParseTableBase
    callbacks: ParserCallbacks
    debug: bool

    def __init__(self, parse_table: ParseTableBase, callbacks: ParserCallbacks, debug: bool=False):
        if False:
            print('Hello World!')
        self.parse_table = parse_table
        self.callbacks = callbacks
        self.debug = debug

    def parse(self, lexer: LexerThread, start: str, value_stack=None, state_stack=None, start_interactive=False):
        if False:
            i = 10
            return i + 15
        parse_conf = ParseConf(self.parse_table, self.callbacks, start)
        parser_state = ParserState(parse_conf, lexer, state_stack, value_stack)
        if start_interactive:
            return InteractiveParser(self, parser_state, parser_state.lexer)
        return self.parse_from_state(parser_state)

    def parse_from_state(self, state: ParserState, last_token: Optional[Token]=None):
        if False:
            i = 10
            return i + 15
        'Run the main LALR parser loop\n\n        Parameters:\n            state - the initial state. Changed in-place.\n            last_token - Used only for line information in case of an empty lexer.\n        '
        try:
            token = last_token
            for token in state.lexer.lex(state):
                assert token is not None
                state.feed_token(token)
            end_token = Token.new_borrow_pos('$END', '', token) if token else Token('$END', '', 0, 1, 1)
            return state.feed_token(end_token, True)
        except UnexpectedInput as e:
            try:
                e.interactive_parser = InteractiveParser(self, state, state.lexer)
            except NameError:
                pass
            raise e
        except Exception as e:
            if self.debug:
                print('')
                print('STATE STACK DUMP')
                print('----------------')
                for (i, s) in enumerate(state.state_stack):
                    print('%d)' % i, s)
                print('')
            raise