"""Utility functions common to the C and C++ domains."""
from __future__ import annotations
import re
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable
from docutils import nodes
from sphinx import addnodes
from sphinx.util import logging
if TYPE_CHECKING:
    from docutils.nodes import TextElement
    from sphinx.config import Config
logger = logging.getLogger(__name__)
StringifyTransform = Callable[[Any], str]
_whitespace_re = re.compile('\\s+')
anon_identifier_re = re.compile('(@[a-zA-Z0-9_])[a-zA-Z0-9_]*\\b')
identifier_re = re.compile("\n    (   # This 'extends' _anon_identifier_re with the ordinary identifiers,\n        # make sure they are in sync.\n        (~?\\b[a-zA-Z_])  # ordinary identifiers\n    |   (@[a-zA-Z0-9_])  # our extension for names of anonymous entities\n    )\n    [a-zA-Z0-9_]*\\b\n", flags=re.VERBOSE)
integer_literal_re = re.compile("[1-9][0-9]*(\\'[0-9]+)*")
octal_literal_re = re.compile("0[0-7]*(\\'[0-7]+)*")
hex_literal_re = re.compile("0[xX][0-9a-fA-F]+(\\'[0-9a-fA-F]+)*")
binary_literal_re = re.compile("0[bB][01]+(\\'[01]+)*")
integers_literal_suffix_re = re.compile('\n    # unsigned and/or (long) long, in any order, but at least one of them\n    (\n        ([uU]    ([lL]  |  (ll)  |  (LL))?)\n        |\n        (([lL]  |  (ll)  |  (LL))    [uU]?)\n    )\\b\n    # the ending word boundary is important for distinguishing\n    # between suffixes and UDLs in C++\n', flags=re.VERBOSE)
float_literal_re = re.compile("\n    [+-]?(\n    # decimal\n      ([0-9]+(\\'[0-9]+)*[eE][+-]?[0-9]+(\\'[0-9]+)*)\n    | (([0-9]+(\\'[0-9]+)*)?\\.[0-9]+(\\'[0-9]+)*([eE][+-]?[0-9]+(\\'[0-9]+)*)?)\n    | ([0-9]+(\\'[0-9]+)*\\.([eE][+-]?[0-9]+(\\'[0-9]+)*)?)\n    # hex\n    | (0[xX][0-9a-fA-F]+(\\'[0-9a-fA-F]+)*[pP][+-]?[0-9a-fA-F]+(\\'[0-9a-fA-F]+)*)\n    | (0[xX]([0-9a-fA-F]+(\\'[0-9a-fA-F]+)*)?\\.\n        [0-9a-fA-F]+(\\'[0-9a-fA-F]+)*([pP][+-]?[0-9a-fA-F]+(\\'[0-9a-fA-F]+)*)?)\n    | (0[xX][0-9a-fA-F]+(\\'[0-9a-fA-F]+)*\\.([pP][+-]?[0-9a-fA-F]+(\\'[0-9a-fA-F]+)*)?)\n    )\n", flags=re.VERBOSE)
float_literal_suffix_re = re.compile('[fFlL]\\b')
char_literal_re = re.compile('\n    ((?:u8)|u|U|L)?\n    \'(\n      (?:[^\\\\\'])\n    | (\\\\(\n        (?:[\'"?\\\\abfnrtv])\n      | (?:[0-7]{1,3})\n      | (?:x[0-9a-fA-F]{2})\n      | (?:u[0-9a-fA-F]{4})\n      | (?:U[0-9a-fA-F]{8})\n      ))\n    )\'\n', flags=re.VERBOSE)

def verify_description_mode(mode: str) -> None:
    if False:
        print('Hello World!')
    if mode not in ('lastIsName', 'noneIsName', 'markType', 'markName', 'param', 'udl'):
        raise Exception("Description mode '%s' is invalid." % mode)

class NoOldIdError(Exception):
    pass

class ASTBaseBase:

    def __eq__(self, other: Any) -> bool:
        if False:
            return 10
        if type(self) is not type(other):
            return False
        try:
            for (key, value) in self.__dict__.items():
                if value != getattr(other, key):
                    return False
        except AttributeError:
            return False
        return True
    __hash__ = None

    def clone(self) -> Any:
        if False:
            print('Hello World!')
        return deepcopy(self)

    def _stringify(self, transform: StringifyTransform) -> str:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError(repr(self))

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        return self._stringify(lambda ast: str(ast))

    def get_display_string(self) -> str:
        if False:
            return 10
        return self._stringify(lambda ast: ast.get_display_string())

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return '<%s>' % self.__class__.__name__

class ASTAttribute(ASTBaseBase):

    def describe_signature(self, signode: TextElement) -> None:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError(repr(self))

class ASTCPPAttribute(ASTAttribute):

    def __init__(self, arg: str) -> None:
        if False:
            return 10
        self.arg = arg

    def _stringify(self, transform: StringifyTransform) -> str:
        if False:
            return 10
        return '[[' + self.arg + ']]'

    def describe_signature(self, signode: TextElement) -> None:
        if False:
            print('Hello World!')
        signode.append(addnodes.desc_sig_punctuation('[[', '[['))
        signode.append(nodes.Text(self.arg))
        signode.append(addnodes.desc_sig_punctuation(']]', ']]'))

class ASTGnuAttribute(ASTBaseBase):

    def __init__(self, name: str, args: ASTBaseParenExprList | None) -> None:
        if False:
            return 10
        self.name = name
        self.args = args

    def _stringify(self, transform: StringifyTransform) -> str:
        if False:
            print('Hello World!')
        res = [self.name]
        if self.args:
            res.append(transform(self.args))
        return ''.join(res)

class ASTGnuAttributeList(ASTAttribute):

    def __init__(self, attrs: list[ASTGnuAttribute]) -> None:
        if False:
            print('Hello World!')
        self.attrs = attrs

    def _stringify(self, transform: StringifyTransform) -> str:
        if False:
            print('Hello World!')
        res = ['__attribute__((']
        first = True
        for attr in self.attrs:
            if not first:
                res.append(', ')
            first = False
            res.append(transform(attr))
        res.append('))')
        return ''.join(res)

    def describe_signature(self, signode: TextElement) -> None:
        if False:
            print('Hello World!')
        txt = str(self)
        signode.append(nodes.Text(txt))

class ASTIdAttribute(ASTAttribute):
    """For simple attributes defined by the user."""

    def __init__(self, id: str) -> None:
        if False:
            i = 10
            return i + 15
        self.id = id

    def _stringify(self, transform: StringifyTransform) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.id

    def describe_signature(self, signode: TextElement) -> None:
        if False:
            return 10
        signode.append(nodes.Text(self.id))

class ASTParenAttribute(ASTAttribute):
    """For paren attributes defined by the user."""

    def __init__(self, id: str, arg: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.id = id
        self.arg = arg

    def _stringify(self, transform: StringifyTransform) -> str:
        if False:
            return 10
        return self.id + '(' + self.arg + ')'

    def describe_signature(self, signode: TextElement) -> None:
        if False:
            while True:
                i = 10
        txt = str(self)
        signode.append(nodes.Text(txt))

class ASTAttributeList(ASTBaseBase):

    def __init__(self, attrs: list[ASTAttribute]) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.attrs = attrs

    def __len__(self) -> int:
        if False:
            return 10
        return len(self.attrs)

    def __add__(self, other: ASTAttributeList) -> ASTAttributeList:
        if False:
            print('Hello World!')
        return ASTAttributeList(self.attrs + other.attrs)

    def _stringify(self, transform: StringifyTransform) -> str:
        if False:
            while True:
                i = 10
        return ' '.join((transform(attr) for attr in self.attrs))

    def describe_signature(self, signode: TextElement) -> None:
        if False:
            i = 10
            return i + 15
        if len(self.attrs) == 0:
            return
        self.attrs[0].describe_signature(signode)
        if len(self.attrs) == 1:
            return
        for attr in self.attrs[1:]:
            signode.append(addnodes.desc_sig_space())
            attr.describe_signature(signode)

class ASTBaseParenExprList(ASTBaseBase):
    pass

class UnsupportedMultiCharacterCharLiteral(Exception):
    pass

class DefinitionError(Exception):
    pass

class BaseParser:

    def __init__(self, definition: str, *, location: nodes.Node | tuple[str, int] | str, config: Config) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.definition = definition.strip()
        self.location = location
        self.config = config
        self.pos = 0
        self.end = len(self.definition)
        self.last_match: re.Match[str] | None = None
        self._previous_state: tuple[int, re.Match[str] | None] = (0, None)
        self.otherErrors: list[DefinitionError] = []
        self.allowFallbackExpressionParsing = True

    def _make_multi_error(self, errors: list[Any], header: str) -> DefinitionError:
        if False:
            for i in range(10):
                print('nop')
        if len(errors) == 1:
            if len(header) > 0:
                return DefinitionError(header + '\n' + str(errors[0][0]))
            else:
                return DefinitionError(str(errors[0][0]))
        result = [header, '\n']
        for e in errors:
            if len(e[1]) > 0:
                indent = '  '
                result.append(e[1])
                result.append(':\n')
                for line in str(e[0]).split('\n'):
                    if len(line) == 0:
                        continue
                    result.append(indent)
                    result.append(line)
                    result.append('\n')
            else:
                result.append(str(e[0]))
        return DefinitionError(''.join(result))

    @property
    def language(self) -> str:
        if False:
            return 10
        raise NotImplementedError

    def status(self, msg: str) -> None:
        if False:
            i = 10
            return i + 15
        indicator = '-' * self.pos + '^'
        logger.debug(f'{msg}\n{self.definition}\n{indicator}')

    def fail(self, msg: str) -> None:
        if False:
            i = 10
            return i + 15
        errors = []
        indicator = '-' * self.pos + '^'
        exMain = DefinitionError('Invalid %s declaration: %s [error at %d]\n  %s\n  %s' % (self.language, msg, self.pos, self.definition, indicator))
        errors.append((exMain, 'Main error'))
        for err in self.otherErrors:
            errors.append((err, 'Potential other error'))
        self.otherErrors = []
        raise self._make_multi_error(errors, '')

    def warn(self, msg: str) -> None:
        if False:
            print('Hello World!')
        logger.warning(msg, location=self.location)

    def match(self, regex: re.Pattern[str]) -> bool:
        if False:
            i = 10
            return i + 15
        match = regex.match(self.definition, self.pos)
        if match is not None:
            self._previous_state = (self.pos, self.last_match)
            self.pos = match.end()
            self.last_match = match
            return True
        return False

    def skip_string(self, string: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        strlen = len(string)
        if self.definition[self.pos:self.pos + strlen] == string:
            self.pos += strlen
            return True
        return False

    def skip_word(self, word: str) -> bool:
        if False:
            print('Hello World!')
        return self.match(re.compile('\\b%s\\b' % re.escape(word)))

    def skip_ws(self) -> bool:
        if False:
            return 10
        return self.match(_whitespace_re)

    def skip_word_and_ws(self, word: str) -> bool:
        if False:
            while True:
                i = 10
        if self.skip_word(word):
            self.skip_ws()
            return True
        return False

    def skip_string_and_ws(self, string: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if self.skip_string(string):
            self.skip_ws()
            return True
        return False

    @property
    def eof(self) -> bool:
        if False:
            print('Hello World!')
        return self.pos >= self.end

    @property
    def current_char(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        try:
            return self.definition[self.pos]
        except IndexError:
            return 'EOF'

    @property
    def matched_text(self) -> str:
        if False:
            while True:
                i = 10
        if self.last_match is not None:
            return self.last_match.group()
        return ''

    def read_rest(self) -> str:
        if False:
            return 10
        rv = self.definition[self.pos:]
        self.pos = self.end
        return rv

    def assert_end(self, *, allowSemicolon: bool=False) -> None:
        if False:
            while True:
                i = 10
        self.skip_ws()
        if allowSemicolon:
            if not self.eof and self.definition[self.pos:] != ';':
                self.fail('Expected end of definition or ;.')
        elif not self.eof:
            self.fail('Expected end of definition.')

    @property
    def id_attributes(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    @property
    def paren_attributes(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def _parse_balanced_token_seq(self, end: list[str]) -> str:
        if False:
            while True:
                i = 10
        brackets = {'(': ')', '[': ']', '{': '}'}
        startPos = self.pos
        symbols: list[str] = []
        while not self.eof:
            if len(symbols) == 0 and self.current_char in end:
                break
            if self.current_char in brackets:
                symbols.append(brackets[self.current_char])
            elif len(symbols) > 0 and self.current_char == symbols[-1]:
                symbols.pop()
            elif self.current_char in ')]}':
                self.fail("Unexpected '%s' in balanced-token-seq." % self.current_char)
            self.pos += 1
        if self.eof:
            self.fail('Could not find end of balanced-token-seq starting at %d.' % startPos)
        return self.definition[startPos:self.pos]

    def _parse_attribute(self) -> ASTAttribute | None:
        if False:
            return 10
        self.skip_ws()
        startPos = self.pos
        if self.skip_string_and_ws('['):
            if not self.skip_string('['):
                self.pos = startPos
            else:
                arg = self._parse_balanced_token_seq(end=[']'])
                if not self.skip_string_and_ws(']'):
                    self.fail("Expected ']' in end of attribute.")
                if not self.skip_string_and_ws(']'):
                    self.fail("Expected ']' in end of attribute after [[...]")
                return ASTCPPAttribute(arg)
        if self.skip_word_and_ws('__attribute__'):
            if not self.skip_string_and_ws('('):
                self.fail("Expected '(' after '__attribute__'.")
            if not self.skip_string_and_ws('('):
                self.fail("Expected '(' after '__attribute__('.")
            attrs = []
            while 1:
                if self.match(identifier_re):
                    name = self.matched_text
                    exprs = self._parse_paren_expression_list()
                    attrs.append(ASTGnuAttribute(name, exprs))
                if self.skip_string_and_ws(','):
                    continue
                if self.skip_string_and_ws(')'):
                    break
                self.fail("Expected identifier, ')', or ',' in __attribute__.")
            if not self.skip_string_and_ws(')'):
                self.fail("Expected ')' after '__attribute__((...)'")
            return ASTGnuAttributeList(attrs)
        for id in self.id_attributes:
            if self.skip_word_and_ws(id):
                return ASTIdAttribute(id)
        for id in self.paren_attributes:
            if not self.skip_string_and_ws(id):
                continue
            if not self.skip_string('('):
                self.fail("Expected '(' after user-defined paren-attribute.")
            arg = self._parse_balanced_token_seq(end=[')'])
            if not self.skip_string(')'):
                self.fail("Expected ')' to end user-defined paren-attribute.")
            return ASTParenAttribute(id, arg)
        return None

    def _parse_attribute_list(self) -> ASTAttributeList:
        if False:
            while True:
                i = 10
        res = []
        while True:
            attr = self._parse_attribute()
            if attr is None:
                break
            res.append(attr)
        return ASTAttributeList(res)

    def _parse_paren_expression_list(self) -> ASTBaseParenExprList | None:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError