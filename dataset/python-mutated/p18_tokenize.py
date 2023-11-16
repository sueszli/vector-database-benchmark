"""
Topic: 字符串令牌化
Desc : 
"""
import re
from collections import namedtuple

def tokenize_str():
    if False:
        for i in range(10):
            print('nop')
    text = 'foo = 23 + 42 * 10'
    tokens = [('NAME', 'foo'), ('EQ', '='), ('NUM', '23'), ('PLUS', '+'), ('NUM', '42'), ('TIMES', '*'), ('NUM', '10')]
    NAME = '(?P<NAME>[a-zA-Z_][a-zA-Z_0-9]*)'
    NUM = '(?P<NUM>\\d+)'
    PLUS = '(?P<PLUS>\\+)'
    TIMES = '(?P<TIMES>\\*)'
    EQ = '(?P<EQ>=)'
    WS = '(?P<WS>\\s+)'
    master_pat = re.compile('|'.join([NAME, NUM, PLUS, TIMES, EQ, WS]))
    scanner = master_pat.scanner('foo = 42')
    a = scanner.match()
    print(a)
    print((a.lastgroup, a.group()))
    a = scanner.match()
    print(a)
    print((a.lastgroup, a.group()))
    a = scanner.match()
    print(a)
    print((a.lastgroup, a.group()))
    a = scanner.match()
    print(a)
    print((a.lastgroup, a.group()))
    a = scanner.match()
    print(a)
    print((a.lastgroup, a.group()))
    a = scanner.match()
    print(a)
    for tok in generate_tokens(master_pat, 'foo = 42'):
        print(tok)
    tokens = (tok for tok in generate_tokens(master_pat, text) if tok.type != 'WS')
    for tok in tokens:
        print(tok)
    print('*' * 40)
    LT = '(?P<LT><)'
    LE = '(?P<LE><=)'
    EQ = '(?P<EQ>=)'
    master_pat = re.compile('|'.join([LE, LT, EQ]))

def generate_tokens(pat, text):
    if False:
        return 10
    Token = namedtuple('Token', ['type', 'value'])
    scanner = pat.scanner(text)
    for m in iter(scanner.match, None):
        yield Token(m.lastgroup, m.group())
if __name__ == '__main__':
    tokenize_str()