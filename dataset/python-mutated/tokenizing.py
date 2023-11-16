__license__ = 'GPL v3'
__copyright__ = '2014, Kovid Goyal <kovid at kovidgoyal.net>'
from tinycss.tests import BaseTest, jsonify
from tinycss.tokenizer import python_tokenize_flat, c_tokenize_flat, regroup
if c_tokenize_flat is None:
    tokenizers = (python_tokenize_flat,)
else:
    tokenizers = (python_tokenize_flat, c_tokenize_flat)

def token_api(self, tokenize):
    if False:
        return 10
    for css_source in ['(8, foo, [z])', '[8, foo, (z)]', '{8, foo, [z]}', 'func(8, foo, [z])']:
        tokens = list(regroup(tokenize(css_source)))
        self.ae(len(tokens), 1)
        self.ae(len(tokens[0].content), 7)

def token_serialize_css(self, tokenize):
    if False:
        print('Hello World!')
    for tokenize in tokenizers:
        for css_source in ['p[example="\\\nfoo(int x) {\\\n    this.x = x;\\\n}\\\n"]', '"Lorem\\26Ipsum\ndolor" sit', '/* Lorem\nipsum */\x0ca {\n    color: red;\tcontent: "dolor\\\x0csit" }', 'not([[lorem]]{ipsum (42)})', 'a[b{d]e}', 'a[b{"d']:
            for _regroup in (regroup, lambda x: x):
                tokens = _regroup(tokenize(css_source, ignore_comments=False))
                result = ''.join((token.as_css() for token in tokens))
                self.ae(result, css_source)

def comments(self, tokenize):
    if False:
        return 10
    for (ignore_comments, expected_tokens) in [(False, [('COMMENT', '/* lorem */'), ('S', ' '), ('IDENT', 'ipsum'), ('[', [('IDENT', 'dolor'), ('COMMENT', '/* sit */')]), ('BAD_COMMENT', '/* amet')]), (True, [('S', ' '), ('IDENT', 'ipsum'), ('[', [('IDENT', 'dolor')])])]:
        css_source = '/* lorem */ ipsum[dolor/* sit */]/* amet'
        tokens = regroup(tokenize(css_source, ignore_comments))
        result = list(jsonify(tokens))
        self.ae(result, expected_tokens)

def token_grouping(self, tokenize):
    if False:
        return 10
    for (css_source, expected_tokens) in [('', []), ('Lorem\\26 "i\\psum"4px', [('IDENT', 'Lorem&'), ('STRING', 'ipsum'), ('DIMENSION', 4)]), ('not([[lorem]]{ipsum (42)})', [('FUNCTION', 'not', [('[', [('[', [('IDENT', 'lorem')])]), ('{', [('IDENT', 'ipsum'), ('S', ' '), ('(', [('INTEGER', 42)])])])]), ('a[b{"d', [('IDENT', 'a'), ('[', [('IDENT', 'b'), ('{', [('STRING', 'd')])])]), ('a[b{d]e}', [('IDENT', 'a'), ('[', [('IDENT', 'b'), ('{', [('IDENT', 'd'), (']', ']'), ('IDENT', 'e')])])]), ('a[b{d}e]', [('IDENT', 'a'), ('[', [('IDENT', 'b'), ('{', [('IDENT', 'd')]), ('IDENT', 'e')])])]:
        tokens = regroup(tokenize(css_source, ignore_comments=False))
        result = list(jsonify(tokens))
        self.ae(result, expected_tokens)

def positions(self, tokenize):
    if False:
        i = 10
        return i + 15
    css = '/* Lorem\nipsum */\x0ca {\n    color: red;\tcontent: "dolor\\\x0csit" }'
    tokens = tokenize(css, ignore_comments=False)
    result = [(token.type, token.line, token.column) for token in tokens]
    self.ae(result, [('COMMENT', 1, 1), ('S', 2, 9), ('IDENT', 3, 1), ('S', 3, 2), ('{', 3, 3), ('S', 3, 4), ('IDENT', 4, 5), (':', 4, 10), ('S', 4, 11), ('IDENT', 4, 12), (';', 4, 15), ('S', 4, 16), ('IDENT', 4, 17), (':', 4, 24), ('S', 4, 25), ('STRING', 4, 26), ('S', 5, 5), ('}', 5, 6)])

def tokens(self, tokenize):
    if False:
        i = 10
        return i + 15
    for (css_source, expected_tokens) in [('', []), ('red -->', [('IDENT', 'red'), ('S', ' '), ('CDC', '-->')]), ('red-->', [('IDENT', 'red--'), ('DELIM', '>')]), ('p[example="\\\nfoo(int x) {\\\n    this.x = x;\\\n}\\\n"]', [('IDENT', 'p'), ('[', '['), ('IDENT', 'example'), ('DELIM', '='), ('STRING', 'foo(int x) {    this.x = x;}'), (']', ']')]), ('42 .5 -4pX 1.25em 30%', [('INTEGER', 42), ('S', ' '), ('NUMBER', 0.5), ('S', ' '), ('DIMENSION', -4, 'px'), ('S', ' '), ('DIMENSION', 1.25, 'em'), ('S', ' '), ('PERCENTAGE', 30, '%')]), ('url(foo.png)', [('URI', 'foo.png')]), ('url("foo.png")', [('URI', 'foo.png')]), ('/* Comment with a \\ backslash */', [('COMMENT', '/* Comment with a \\ backslash */')]), ('"Lorem\\\nIpsum"', [('STRING', 'LoremIpsum')]), ('Lorem\\\nIpsum', [('IDENT', 'Lorem'), ('DELIM', '\\'), ('S', '\n'), ('IDENT', 'Ipsum')]), ('"Lore\\m Ipsum"', [('STRING', 'Lorem Ipsum')]), ('"Lorem \\49psum"', [('STRING', 'Lorem Ipsum')]), ('"Lorem \\49 psum"', [('STRING', 'Lorem Ipsum')]), ('"Lorem\\"Ipsum"', [('STRING', 'Lorem"Ipsum')]), ('"Lorem\\\\Ipsum"', [('STRING', 'Lorem\\Ipsum')]), ('"Lorem\\5c Ipsum"', [('STRING', 'Lorem\\Ipsum')]), ('Lorem\\+Ipsum', [('IDENT', 'Lorem+Ipsum')]), ('Lorem+Ipsum', [('IDENT', 'Lorem'), ('DELIM', '+'), ('IDENT', 'Ipsum')]), ('url(foo\\).png)', [('URI', 'foo).png')]), ('\\26 B', [('IDENT', '&B')]), ('\\&B', [('IDENT', '&B')]), ('@\\26\tB', [('ATKEYWORD', '@&B')]), ('@\\&B', [('ATKEYWORD', '@&B')]), ('#\\26\nB', [('HASH', '#&B')]), ('#\\&B', [('HASH', '#&B')]), ('\\26\r\nB(', [('FUNCTION', '&B(')]), ('\\&B(', [('FUNCTION', '&B(')]), ('12.5\\000026B', [('DIMENSION', 12.5, '&b')]), ('12.5\\0000263B', [('DIMENSION', 12.5, '&3b')]), ('12.5\\&B', [('DIMENSION', 12.5, '&b')]), ('"\\26 B"', [('STRING', '&B')]), ("'\\000026B'", [('STRING', '&B')]), ('"\\&B"', [('STRING', '&B')]), ('url("\\26 B")', [('URI', '&B')]), ('url(\\26 B)', [('URI', '&B')]), ('url("\\&B")', [('URI', '&B')]), ('url(\\&B)', [('URI', '&B')]), ('Lorem\\110000Ipsum', [('IDENT', 'Lorem�Ipsum')]), ('"Lorem\\26Ipsum', [('STRING', 'Lorem&Ipsum')]), ('"Lorem\\26Ipsum\n', [('BAD_STRING', '"Lorem\\26Ipsum'), ('S', '\n')]), ('"Lorem\\26Ipsum\ndolor" sit', [('BAD_STRING', '"Lorem\\26Ipsum'), ('S', '\n'), ('IDENT', 'dolor'), ('STRING', ' sit')])]:
        sources = [css_source]
        for css_source in sources:
            tokens = tokenize(css_source, ignore_comments=False)
            result = [(token.type, token.value) + (() if token.unit is None else (token.unit,)) for token in tokens]
            self.ae(result, expected_tokens)

class TestTokenizer(BaseTest):

    def run_test(self, func):
        if False:
            i = 10
            return i + 15
        for tokenize in tokenizers:
            func(self, tokenize)

    def test_token_api(self):
        if False:
            return 10
        self.run_test(token_api)

    def test_token_serialize_css(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_test(token_serialize_css)

    def test_comments(self):
        if False:
            print('Hello World!')
        self.run_test(comments)

    def test_token_grouping(self):
        if False:
            while True:
                i = 10
        self.run_test(token_grouping)

    def test_positions(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the reported line/column position of each token.'
        self.run_test(positions)

    def test_tokens(self):
        if False:
            i = 10
            return i + 15
        self.run_test(tokens)