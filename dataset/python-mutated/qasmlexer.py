"""
OPENQASM Lexer.

This is a wrapper around the PLY lexer to support the "include" statement
by creating a stack of lexers.
"""
import os
import numpy as np
from ply import lex
from . import node
from .exceptions import QasmError
CORE_LIBS_PATH = os.path.join(os.path.dirname(__file__), 'libs')
CORE_LIBS = os.listdir(CORE_LIBS_PATH)

class QasmLexer:
    """OPENQASM Lexer.

    This is a wrapper around the PLY lexer to support the "include" statement
    by creating a stack of lexers.
    """

    def __mklexer__(self, filename):
        if False:
            i = 10
            return i + 15
        'Create a PLY lexer.'
        self.lexer = lex.lex(module=self, debug=False)
        self.filename = filename
        self.lineno = 1
        if filename:
            with open(filename) as ifile:
                self.data = ifile.read()
            self.lexer.input(self.data)

    def __init__(self, filename):
        if False:
            i = 10
            return i + 15
        'Create the OPENQASM lexer.'
        self.__mklexer__(filename)
        self.stack = []

    def input(self, data):
        if False:
            for i in range(10):
                print('nop')
        'Set the input text data.'
        self.data = data
        self.lexer.input(data)

    def token(self):
        if False:
            return 10
        'Return the next token.'
        ret = self.lexer.token()
        return ret

    def pop(self):
        if False:
            return 10
        'Pop a PLY lexer off the stack.'
        self.lexer = self.stack.pop()
        self.filename = self.lexer.qasm_file
        self.lineno = self.lexer.qasm_line

    def push(self, filename):
        if False:
            print('Hello World!')
        'Push a PLY lexer on the stack to parse filename.'
        self.lexer.qasm_file = self.filename
        self.lexer.qasm_line = self.lineno
        self.stack.append(self.lexer)
        self.__mklexer__(filename)
    literals = '=()[]{};<>,.+-/*^"'
    reserved = {'barrier': 'BARRIER', 'creg': 'CREG', 'gate': 'GATE', 'if': 'IF', 'measure': 'MEASURE', 'opaque': 'OPAQUE', 'qreg': 'QREG', 'pi': 'PI', 'reset': 'RESET'}
    tokens = ['NNINTEGER', 'REAL', 'CX', 'U', 'FORMAT', 'ASSIGN', 'MATCHES', 'ID', 'STRING'] + list(reserved.values())

    def t_REAL(self, t):
        if False:
            i = 10
            return i + 15
        '(([0-9]+|([0-9]+)?\\.[0-9]+|[0-9]+\\.)[eE][+-]?[0-9]+)|(([0-9]+)?\\.[0-9]+|[0-9]+\\.)'
        if np.iscomplex(t):
            return t.real
        else:
            return t

    def t_NNINTEGER(self, t):
        if False:
            return 10
        '[1-9]+[0-9]*|0'
        t.value = int(t.value)
        return t

    def t_ASSIGN(self, t):
        if False:
            i = 10
            return i + 15
        '->'
        return t

    def t_MATCHES(self, t):
        if False:
            i = 10
            return i + 15
        '=='
        return t

    def t_STRING(self, t):
        if False:
            for i in range(10):
                print('nop')
        '\\"([^\\\\\\"]|\\\\.)*\\"'
        return t

    def t_INCLUDE(self, _):
        if False:
            return 10
        'include'
        next_token = self.lexer.token()
        lineno = next_token.lineno
        if isinstance(next_token.value, str):
            incfile = next_token.value.strip('"')
        else:
            raise QasmError('Invalid include: must be a quoted string.')
        if incfile in CORE_LIBS:
            incfile = os.path.join(CORE_LIBS_PATH, incfile)
        next_token = self.lexer.token()
        if next_token is None or next_token.value != ';':
            raise QasmError('Invalid syntax, missing ";" at line', str(lineno))
        if not os.path.exists(incfile):
            raise QasmError('Include file %s cannot be found, line %s, file %s' % (incfile, str(next_token.lineno), self.filename))
        self.push(incfile)
        return self.lexer.token()

    def t_FORMAT(self, t):
        if False:
            while True:
                i = 10
        'OPENQASM\\s+[0-9]+(\\.[0-9]+)?'
        return t

    def t_COMMENT(self, _):
        if False:
            for i in range(10):
                print('nop')
        '//.*'
        pass

    def t_CX(self, t):
        if False:
            while True:
                i = 10
        'CX'
        return t

    def t_U(self, t):
        if False:
            print('Hello World!')
        'U'
        return t

    def t_ID(self, t):
        if False:
            while True:
                i = 10
        '[a-z][a-zA-Z0-9_]*'
        t.type = self.reserved.get(t.value, 'ID')
        if t.type == 'ID':
            t.value = node.Id(t.value, self.lineno, self.filename)
        return t

    def t_newline(self, t):
        if False:
            while True:
                i = 10
        '\\n+'
        self.lineno += len(t.value)
        t.lexer.lineno = self.lineno

    def t_eof(self, _):
        if False:
            print('Hello World!')
        if self.stack:
            self.pop()
            return self.lexer.token()
        return None
    t_ignore = ' \t\r'

    def t_error(self, t):
        if False:
            return 10
        raise QasmError('Unable to match any token rule, got -->%s<-- Check your OPENQASM source and any include statements.' % t.value[0])