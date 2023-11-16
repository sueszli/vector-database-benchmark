"""
Parser and utilities for the smart 'if' tag
"""

class TokenBase:
    """
    Base class for operators and literals, mainly for debugging and for throwing
    syntax errors.
    """
    id = None
    value = None
    first = second = None

    def nud(self, parser):
        if False:
            i = 10
            return i + 15
        raise parser.error_class("Not expecting '%s' in this position in if tag." % self.id)

    def led(self, left, parser):
        if False:
            while True:
                i = 10
        raise parser.error_class("Not expecting '%s' as infix operator in if tag." % self.id)

    def display(self):
        if False:
            while True:
                i = 10
        '\n        Return what to display in error messages for this node\n        '
        return self.id

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        out = [str(x) for x in [self.id, self.first, self.second] if x is not None]
        return '(' + ' '.join(out) + ')'

def infix(bp, func):
    if False:
        print('Hello World!')
    '\n    Create an infix operator, given a binding power and a function that\n    evaluates the node.\n    '

    class Operator(TokenBase):
        lbp = bp

        def led(self, left, parser):
            if False:
                return 10
            self.first = left
            self.second = parser.expression(bp)
            return self

        def eval(self, context):
            if False:
                while True:
                    i = 10
            try:
                return func(context, self.first, self.second)
            except Exception:
                return False
    return Operator

def prefix(bp, func):
    if False:
        print('Hello World!')
    '\n    Create a prefix operator, given a binding power and a function that\n    evaluates the node.\n    '

    class Operator(TokenBase):
        lbp = bp

        def nud(self, parser):
            if False:
                while True:
                    i = 10
            self.first = parser.expression(bp)
            self.second = None
            return self

        def eval(self, context):
            if False:
                i = 10
                return i + 15
            try:
                return func(context, self.first)
            except Exception:
                return False
    return Operator
OPERATORS = {'or': infix(6, lambda context, x, y: x.eval(context) or y.eval(context)), 'and': infix(7, lambda context, x, y: x.eval(context) and y.eval(context)), 'not': prefix(8, lambda context, x: not x.eval(context)), 'in': infix(9, lambda context, x, y: x.eval(context) in y.eval(context)), 'not in': infix(9, lambda context, x, y: x.eval(context) not in y.eval(context)), 'is': infix(10, lambda context, x, y: x.eval(context) is y.eval(context)), 'is not': infix(10, lambda context, x, y: x.eval(context) is not y.eval(context)), '==': infix(10, lambda context, x, y: x.eval(context) == y.eval(context)), '!=': infix(10, lambda context, x, y: x.eval(context) != y.eval(context)), '>': infix(10, lambda context, x, y: x.eval(context) > y.eval(context)), '>=': infix(10, lambda context, x, y: x.eval(context) >= y.eval(context)), '<': infix(10, lambda context, x, y: x.eval(context) < y.eval(context)), '<=': infix(10, lambda context, x, y: x.eval(context) <= y.eval(context))}
for (key, op) in OPERATORS.items():
    op.id = key

class Literal(TokenBase):
    """
    A basic self-resolvable object similar to a Django template variable.
    """
    id = 'literal'
    lbp = 0

    def __init__(self, value):
        if False:
            i = 10
            return i + 15
        self.value = value

    def display(self):
        if False:
            return 10
        return repr(self.value)

    def nud(self, parser):
        if False:
            return 10
        return self

    def eval(self, context):
        if False:
            i = 10
            return i + 15
        return self.value

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '(%s %r)' % (self.id, self.value)

class EndToken(TokenBase):
    lbp = 0

    def nud(self, parser):
        if False:
            return 10
        raise parser.error_class('Unexpected end of expression in if tag.')
EndToken = EndToken()

class IfParser:
    error_class = ValueError

    def __init__(self, tokens):
        if False:
            i = 10
            return i + 15
        num_tokens = len(tokens)
        mapped_tokens = []
        i = 0
        while i < num_tokens:
            token = tokens[i]
            if token == 'is' and i + 1 < num_tokens and (tokens[i + 1] == 'not'):
                token = 'is not'
                i += 1
            elif token == 'not' and i + 1 < num_tokens and (tokens[i + 1] == 'in'):
                token = 'not in'
                i += 1
            mapped_tokens.append(self.translate_token(token))
            i += 1
        self.tokens = mapped_tokens
        self.pos = 0
        self.current_token = self.next_token()

    def translate_token(self, token):
        if False:
            print('Hello World!')
        try:
            op = OPERATORS[token]
        except (KeyError, TypeError):
            return self.create_var(token)
        else:
            return op()

    def next_token(self):
        if False:
            for i in range(10):
                print('nop')
        if self.pos >= len(self.tokens):
            return EndToken
        else:
            retval = self.tokens[self.pos]
            self.pos += 1
            return retval

    def parse(self):
        if False:
            while True:
                i = 10
        retval = self.expression()
        if self.current_token is not EndToken:
            raise self.error_class("Unused '%s' at end of if expression." % self.current_token.display())
        return retval

    def expression(self, rbp=0):
        if False:
            while True:
                i = 10
        t = self.current_token
        self.current_token = self.next_token()
        left = t.nud(self)
        while rbp < self.current_token.lbp:
            t = self.current_token
            self.current_token = self.next_token()
            left = t.led(left, self)
        return left

    def create_var(self, value):
        if False:
            i = 10
            return i + 15
        return Literal(value)