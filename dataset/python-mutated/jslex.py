"""JsLex: a lexer for JavaScript"""
import re

class Tok:
    """
    A specification for a token class.
    """
    num = 0

    def __init__(self, name, regex, next=None):
        if False:
            print('Hello World!')
        self.id = Tok.num
        Tok.num += 1
        self.name = name
        self.regex = regex
        self.next = next

def literals(choices, prefix='', suffix=''):
    if False:
        i = 10
        return i + 15
    '\n    Create a regex from a space-separated list of literal `choices`.\n\n    If provided, `prefix` and `suffix` will be attached to each choice\n    individually.\n    '
    return '|'.join((prefix + re.escape(c) + suffix for c in choices.split()))

class Lexer:
    """
    A generic multi-state regex-based lexer.
    """

    def __init__(self, states, first):
        if False:
            print('Hello World!')
        self.regexes = {}
        self.toks = {}
        for (state, rules) in states.items():
            parts = []
            for tok in rules:
                groupid = 't%d' % tok.id
                self.toks[groupid] = tok
                parts.append('(?P<%s>%s)' % (groupid, tok.regex))
            self.regexes[state] = re.compile('|'.join(parts), re.MULTILINE | re.VERBOSE)
        self.state = first

    def lex(self, text):
        if False:
            for i in range(10):
                print('nop')
        '\n        Lexically analyze `text`.\n\n        Yield pairs (`name`, `tokentext`).\n        '
        end = len(text)
        state = self.state
        regexes = self.regexes
        toks = self.toks
        start = 0
        while start < end:
            for match in regexes[state].finditer(text, start):
                name = match.lastgroup
                tok = toks[name]
                toktext = match[name]
                start += len(toktext)
                yield (tok.name, toktext)
                if tok.next:
                    state = tok.next
                    break
        self.state = state

class JsLexer(Lexer):
    """
    A JavaScript lexer

    >>> lexer = JsLexer()
    >>> list(lexer.lex("a = 1"))
    [('id', 'a'), ('ws', ' '), ('punct', '='), ('ws', ' '), ('dnum', '1')]

    This doesn't properly handle non-ASCII characters in the JavaScript source.
    """
    both_before = [Tok('comment', '/\\*(.|\\n)*?\\*/'), Tok('linecomment', '//.*?$'), Tok('ws', '\\s+'), Tok('keyword', literals('\n                           break case catch class const continue debugger\n                           default delete do else enum export extends\n                           finally for function if import in instanceof\n                           new return super switch this throw try typeof\n                           var void while with\n                           ', suffix='\\b'), next='reg'), Tok('reserved', literals('null true false', suffix='\\b'), next='div'), Tok('id', '\n                  ([a-zA-Z_$   ]|\\\\u[0-9a-fA-Z]{4})   # first char\n                  ([a-zA-Z_$0-9]|\\\\u[0-9a-fA-F]{4})*  # rest chars\n                  ', next='div'), Tok('hnum', '0[xX][0-9a-fA-F]+', next='div'), Tok('onum', '0[0-7]+'), Tok('dnum', '\n                    (   (0|[1-9][0-9]*)     # DecimalIntegerLiteral\n                        \\.                  # dot\n                        [0-9]*              # DecimalDigits-opt\n                        ([eE][-+]?[0-9]+)?  # ExponentPart-opt\n                    |\n                        \\.                  # dot\n                        [0-9]+              # DecimalDigits\n                        ([eE][-+]?[0-9]+)?  # ExponentPart-opt\n                    |\n                        (0|[1-9][0-9]*)     # DecimalIntegerLiteral\n                        ([eE][-+]?[0-9]+)?  # ExponentPart-opt\n                    )\n                    ', next='div'), Tok('punct', literals('\n                         >>>= === !== >>> <<= >>= <= >= == != << >> &&\n                         || += -= *= %= &= |= ^=\n                         '), next='reg'), Tok('punct', literals('++ -- ) ]'), next='div'), Tok('punct', literals('{ } ( [ . ; , < > + - * % & | ^ ! ~ ? : ='), next='reg'), Tok('string', '"([^"\\\\]|(\\\\(.|\\n)))*?"', next='div'), Tok('string', "'([^'\\\\]|(\\\\(.|\\n)))*?'", next='div')]
    both_after = [Tok('other', '.')]
    states = {'div': both_before + [Tok('punct', literals('/= /'), next='reg')] + both_after, 'reg': both_before + [Tok('regex', '\n                    /                       # opening slash\n                    # First character is..\n                    (   [^*\\\\/[]            # anything but * \\ / or [\n                    |   \\\\.                 # or an escape sequence\n                    |   \\[                  # or a class, which has\n                            (   [^\\]\\\\]     #   anything but \\ or ]\n                            |   \\\\.         #   or an escape sequence\n                            )*              #   many times\n                        \\]\n                    )\n                    # Following characters are same, except for excluding a star\n                    (   [^\\\\/[]             # anything but \\ / or [\n                    |   \\\\.                 # or an escape sequence\n                    |   \\[                  # or a class, which has\n                            (   [^\\]\\\\]     #   anything but \\ or ]\n                            |   \\\\.         #   or an escape sequence\n                            )*              #   many times\n                        \\]\n                    )*                      # many times\n                    /                       # closing slash\n                    [a-zA-Z0-9]*            # trailing flags\n                ', next='div')] + both_after}

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__(self.states, 'reg')

def prepare_js_for_gettext(js):
    if False:
        while True:
            i = 10
    '\n    Convert the JavaScript source `js` into something resembling C for\n    xgettext.\n\n    What actually happens is that all the regex literals are replaced with\n    "REGEX".\n    '

    def escape_quotes(m):
        if False:
            print('Hello World!')
        'Used in a regex to properly escape double quotes.'
        s = m[0]
        if s == '"':
            return '\\"'
        else:
            return s
    lexer = JsLexer()
    c = []
    for (name, tok) in lexer.lex(js):
        if name == 'regex':
            tok = '"REGEX"'
        elif name == 'string':
            if tok.startswith("'"):
                guts = re.sub('\\\\.|.', escape_quotes, tok[1:-1])
                tok = '"' + guts + '"'
        elif name == 'id':
            tok = tok.replace('\\', 'U')
        c.append(tok)
    return ''.join(c)