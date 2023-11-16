"""
Topic: 下降解析器
Desc : 
"""
import re
import collections
NUM = '(?P<NUM>\\d+)'
PLUS = '(?P<PLUS>\\+)'
MINUS = '(?P<MINUS>-)'
TIMES = '(?P<TIMES>\\*)'
DIVIDE = '(?P<DIVIDE>/)'
LPAREN = '(?P<LPAREN>\\()'
RPAREN = '(?P<RPAREN>\\))'
WS = '(?P<WS>\\s+)'
master_pat = re.compile('|'.join([NUM, PLUS, MINUS, TIMES, DIVIDE, LPAREN, RPAREN, WS]))
Token = collections.namedtuple('Token', ['type', 'value'])

def generate_tokens(text):
    if False:
        return 10
    scanner = master_pat.scanner(text)
    for m in iter(scanner.match, None):
        tok = Token(m.lastgroup, m.group())
        if tok.type != 'WS':
            yield tok

class ExpressionEvaluator:
    """
    Implementation of a recursive descent parser. Each method
    implements a single grammar rule. Use the ._accept() method
    to test and accept the current lookahead token. Use the ._expect()
    method to exactly match and discard the next token on on the input
    (or raise a SyntaxError if it doesn't match).
    """

    def parse(self, text):
        if False:
            return 10
        self.tokens = generate_tokens(text)
        self.tok = None
        self.nexttok = None
        self._advance()
        return self.expr()

    def _advance(self):
        if False:
            for i in range(10):
                print('nop')
        'Advance one token ahead'
        (self.tok, self.nexttok) = (self.nexttok, next(self.tokens, None))

    def _accept(self, toktype):
        if False:
            print('Hello World!')
        'Test and consume the next token if it matches toktype'
        if self.nexttok and self.nexttok.type == toktype:
            self._advance()
            return True
        else:
            return False

    def _expect(self, toktype):
        if False:
            while True:
                i = 10
        'Consume next token if it matches toktype or raise SyntaxError'
        if not self._accept(toktype):
            raise SyntaxError('Expected ' + toktype)

    def expr(self):
        if False:
            i = 10
            return i + 15
        "expression ::= term { ('+'|'-') term }*"
        exprval = self.term()
        while self._accept('PLUS') or self._accept('MINUS'):
            op = self.tok.type
            right = self.term()
            if op == 'PLUS':
                exprval += right
            elif op == 'MINUS':
                exprval -= right
        return exprval

    def term(self):
        if False:
            print('Hello World!')
        "term ::= factor { ('*'|'/') factor }*"
        termval = self.factor()
        while self._accept('TIMES') or self._accept('DIVIDE'):
            op = self.tok.type
            right = self.factor()
            if op == 'TIMES':
                termval *= right
            elif op == 'DIVIDE':
                termval /= right
        return termval

    def factor(self):
        if False:
            i = 10
            return i + 15
        'factor ::= NUM | ( expr )'
        if self._accept('NUM'):
            return int(self.tok.value)
        elif self._accept('LPAREN'):
            exprval = self.expr()
            self._expect('RPAREN')
            return exprval
        else:
            raise SyntaxError('Expected NUMBER or LPAREN')

def descent_parser():
    if False:
        i = 10
        return i + 15
    e = ExpressionEvaluator()
    print(e.parse('2'))
    print(e.parse('2 + 3'))
    print(e.parse('2 + 3 * 4'))
    print(e.parse('2 + (3 + 4) * 5'))
if __name__ == '__main__':
    descent_parser()