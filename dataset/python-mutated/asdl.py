from collections import namedtuple
import re
__all__ = ['builtin_types', 'parse', 'AST', 'Module', 'Type', 'Constructor', 'Field', 'Sum', 'Product', 'VisitorBase', 'Check', 'check']
builtin_types = {'identifier', 'string', 'int', 'constant'}

class AST:

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

class Module(AST):

    def __init__(self, name, dfns):
        if False:
            for i in range(10):
                print('nop')
        self.name = name
        self.dfns = dfns
        self.types = {type.name: type.value for type in dfns}

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'Module({0.name}, {0.dfns})'.format(self)

class Type(AST):

    def __init__(self, name, value):
        if False:
            print('Hello World!')
        self.name = name
        self.value = value

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'Type({0.name}, {0.value})'.format(self)

class Constructor(AST):

    def __init__(self, name, fields=None):
        if False:
            i = 10
            return i + 15
        self.name = name
        self.fields = fields or []

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'Constructor({0.name}, {0.fields})'.format(self)

class Field(AST):

    def __init__(self, type, name=None, seq=False, opt=False):
        if False:
            while True:
                i = 10
        self.type = type
        self.name = name
        self.seq = seq
        self.opt = opt

    def __str__(self):
        if False:
            return 10
        if self.seq:
            extra = '*'
        elif self.opt:
            extra = '?'
        else:
            extra = ''
        return '{}{} {}'.format(self.type, extra, self.name)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        if self.seq:
            extra = ', seq=True'
        elif self.opt:
            extra = ', opt=True'
        else:
            extra = ''
        if self.name is None:
            return 'Field({0.type}{1})'.format(self, extra)
        else:
            return 'Field({0.type}, {0.name}{1})'.format(self, extra)

class Sum(AST):

    def __init__(self, types, attributes=None):
        if False:
            i = 10
            return i + 15
        self.types = types
        self.attributes = attributes or []

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        if self.attributes:
            return 'Sum({0.types}, {0.attributes})'.format(self)
        else:
            return 'Sum({0.types})'.format(self)

class Product(AST):

    def __init__(self, fields, attributes=None):
        if False:
            print('Hello World!')
        self.fields = fields
        self.attributes = attributes or []

    def __repr__(self):
        if False:
            while True:
                i = 10
        if self.attributes:
            return 'Product({0.fields}, {0.attributes})'.format(self)
        else:
            return 'Product({0.fields})'.format(self)

class VisitorBase(object):
    """Generic tree visitor for ASTs."""

    def __init__(self):
        if False:
            print('Hello World!')
        self.cache = {}

    def visit(self, obj, *args):
        if False:
            print('Hello World!')
        klass = obj.__class__
        meth = self.cache.get(klass)
        if meth is None:
            methname = 'visit' + klass.__name__
            meth = getattr(self, methname, None)
            self.cache[klass] = meth
        if meth:
            try:
                meth(obj, *args)
            except Exception as e:
                print('Error visiting %r: %s' % (obj, e))
                raise

class Check(VisitorBase):
    """A visitor that checks a parsed ASDL tree for correctness.

    Errors are printed and accumulated.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(Check, self).__init__()
        self.cons = {}
        self.errors = 0
        self.types = {}

    def visitModule(self, mod):
        if False:
            while True:
                i = 10
        for dfn in mod.dfns:
            self.visit(dfn)

    def visitType(self, type):
        if False:
            while True:
                i = 10
        self.visit(type.value, str(type.name))

    def visitSum(self, sum, name):
        if False:
            return 10
        for t in sum.types:
            self.visit(t, name)

    def visitConstructor(self, cons, name):
        if False:
            while True:
                i = 10
        key = str(cons.name)
        conflict = self.cons.get(key)
        if conflict is None:
            self.cons[key] = name
        else:
            print('Redefinition of constructor {}'.format(key))
            print('Defined in {} and {}'.format(conflict, name))
            self.errors += 1
        for f in cons.fields:
            self.visit(f, key)

    def visitField(self, field, name):
        if False:
            for i in range(10):
                print('nop')
        key = str(field.type)
        l = self.types.setdefault(key, [])
        l.append(name)

    def visitProduct(self, prod, name):
        if False:
            for i in range(10):
                print('nop')
        for f in prod.fields:
            self.visit(f, name)

def check(mod):
    if False:
        for i in range(10):
            print('nop')
    'Check the parsed ASDL tree for correctness.\n\n    Return True if success. For failure, the errors are printed out and False\n    is returned.\n    '
    v = Check()
    v.visit(mod)
    for t in v.types:
        if t not in mod.types and (not t in builtin_types):
            v.errors += 1
            uses = ', '.join(v.types[t])
            print('Undefined type {}, used in {}'.format(t, uses))
    return not v.errors

def parse(filename):
    if False:
        while True:
            i = 10
    'Parse ASDL from the given file and return a Module node describing it.'
    with open(filename, encoding='utf-8') as f:
        parser = ASDLParser()
        return parser.parse(f.read())

class TokenKind:
    """TokenKind is provides a scope for enumerated token kinds."""
    (ConstructorId, TypeId, Equals, Comma, Question, Pipe, Asterisk, LParen, RParen, LBrace, RBrace) = range(11)
    operator_table = {'=': Equals, ',': Comma, '?': Question, '|': Pipe, '(': LParen, ')': RParen, '*': Asterisk, '{': LBrace, '}': RBrace}
Token = namedtuple('Token', 'kind value lineno')

class ASDLSyntaxError(Exception):

    def __init__(self, msg, lineno=None):
        if False:
            print('Hello World!')
        self.msg = msg
        self.lineno = lineno or '<unknown>'

    def __str__(self):
        if False:
            return 10
        return 'Syntax error on line {0.lineno}: {0.msg}'.format(self)

def tokenize_asdl(buf):
    if False:
        return 10
    'Tokenize the given buffer. Yield Token objects.'
    for (lineno, line) in enumerate(buf.splitlines(), 1):
        for m in re.finditer('\\s*(\\w+|--.*|.)', line.strip()):
            c = m.group(1)
            if c[0].isalpha():
                if c[0].isupper():
                    yield Token(TokenKind.ConstructorId, c, lineno)
                else:
                    yield Token(TokenKind.TypeId, c, lineno)
            elif c[:2] == '--':
                break
            else:
                try:
                    op_kind = TokenKind.operator_table[c]
                except KeyError:
                    raise ASDLSyntaxError('Invalid operator %s' % c, lineno)
                yield Token(op_kind, c, lineno)

class ASDLParser:
    """Parser for ASDL files.

    Create, then call the parse method on a buffer containing ASDL.
    This is a simple recursive descent parser that uses tokenize_asdl for the
    lexing.
    """

    def __init__(self):
        if False:
            return 10
        self._tokenizer = None
        self.cur_token = None

    def parse(self, buf):
        if False:
            print('Hello World!')
        'Parse the ASDL in the buffer and return an AST with a Module root.\n        '
        self._tokenizer = tokenize_asdl(buf)
        self._advance()
        return self._parse_module()

    def _parse_module(self):
        if False:
            for i in range(10):
                print('nop')
        if self._at_keyword('module'):
            self._advance()
        else:
            raise ASDLSyntaxError('Expected "module" (found {})'.format(self.cur_token.value), self.cur_token.lineno)
        name = self._match(self._id_kinds)
        self._match(TokenKind.LBrace)
        defs = self._parse_definitions()
        self._match(TokenKind.RBrace)
        return Module(name, defs)

    def _parse_definitions(self):
        if False:
            while True:
                i = 10
        defs = []
        while self.cur_token.kind == TokenKind.TypeId:
            typename = self._advance()
            self._match(TokenKind.Equals)
            type = self._parse_type()
            defs.append(Type(typename, type))
        return defs

    def _parse_type(self):
        if False:
            i = 10
            return i + 15
        if self.cur_token.kind == TokenKind.LParen:
            return self._parse_product()
        else:
            sumlist = [Constructor(self._match(TokenKind.ConstructorId), self._parse_optional_fields())]
            while self.cur_token.kind == TokenKind.Pipe:
                self._advance()
                sumlist.append(Constructor(self._match(TokenKind.ConstructorId), self._parse_optional_fields()))
            return Sum(sumlist, self._parse_optional_attributes())

    def _parse_product(self):
        if False:
            while True:
                i = 10
        return Product(self._parse_fields(), self._parse_optional_attributes())

    def _parse_fields(self):
        if False:
            return 10
        fields = []
        self._match(TokenKind.LParen)
        while self.cur_token.kind == TokenKind.TypeId:
            typename = self._advance()
            (is_seq, is_opt) = self._parse_optional_field_quantifier()
            id = self._advance() if self.cur_token.kind in self._id_kinds else None
            fields.append(Field(typename, id, seq=is_seq, opt=is_opt))
            if self.cur_token.kind == TokenKind.RParen:
                break
            elif self.cur_token.kind == TokenKind.Comma:
                self._advance()
        self._match(TokenKind.RParen)
        return fields

    def _parse_optional_fields(self):
        if False:
            while True:
                i = 10
        if self.cur_token.kind == TokenKind.LParen:
            return self._parse_fields()
        else:
            return None

    def _parse_optional_attributes(self):
        if False:
            return 10
        if self._at_keyword('attributes'):
            self._advance()
            return self._parse_fields()
        else:
            return None

    def _parse_optional_field_quantifier(self):
        if False:
            print('Hello World!')
        (is_seq, is_opt) = (False, False)
        if self.cur_token.kind == TokenKind.Asterisk:
            is_seq = True
            self._advance()
        elif self.cur_token.kind == TokenKind.Question:
            is_opt = True
            self._advance()
        return (is_seq, is_opt)

    def _advance(self):
        if False:
            i = 10
            return i + 15
        ' Return the value of the current token and read the next one into\n            self.cur_token.\n        '
        cur_val = None if self.cur_token is None else self.cur_token.value
        try:
            self.cur_token = next(self._tokenizer)
        except StopIteration:
            self.cur_token = None
        return cur_val
    _id_kinds = (TokenKind.ConstructorId, TokenKind.TypeId)

    def _match(self, kind):
        if False:
            return 10
        "The 'match' primitive of RD parsers.\n\n        * Verifies that the current token is of the given kind (kind can\n          be a tuple, in which the kind must match one of its members).\n        * Returns the value of the current token\n        * Reads in the next token\n        "
        if isinstance(kind, tuple) and self.cur_token.kind in kind or self.cur_token.kind == kind:
            value = self.cur_token.value
            self._advance()
            return value
        else:
            raise ASDLSyntaxError('Unmatched {} (found {})'.format(kind, self.cur_token.kind), self.cur_token.lineno)

    def _at_keyword(self, keyword):
        if False:
            for i in range(10):
                print('nop')
        return self.cur_token.kind == TokenKind.TypeId and self.cur_token.value == keyword