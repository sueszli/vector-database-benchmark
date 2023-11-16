import re
from string import whitespace
from pygments.lexer import RegexLexer, bygroups, include, using, this
from pygments.token import *

class MathematicaLexer(RegexLexer):
    name = 'Mathematica'
    aliases = ['mathematica', 'Mathematica']
    filenames = ['*.m']

    def definition(lexer, match):
        if False:
            print('Hello World!')
        yield (match.start(1), Name.Function, match.group(1))
        args = []
        if match.group(2):
            posn = match.start(2)
            yield (posn, Punctuation, '[')
            posn += 1
            args = [arg + ',' for arg in match.group(2)[1:-1].split(',')]
            args[-1] = args[-1][:-1]
            for arg in args:
                argmatch = re.match('(\\s*)({?)([a-z][A-Za-z0-9]*)(_\\??)([A-Za-z0-9]*)(}?)(,?)', arg)
                for (t, v) in zip((Whitespace, Punctuation, Name.Variable, Punctuation, None, Punctuation, Punctuation), argmatch.groups()):
                    if v:
                        if t is None:
                            t = Name.Builtin if v[0].isupper() else Name
                        yield (posn, t, v)
                        posn += len(t)
            yield (match.end(2) - 1, Punctuation, ']')
            args = [arg.partition('_')[0].strip('_[]{}?' + whitespace) for arg in args]
        if match.group(3):
            yield (match.start(3), Whitespace, match.group(3))
        yield (match.start(4), Operator, match.group(4))
        if match.group(5):
            yield (match.start(5), Whitespace, match.group(5))
        posn = match.start(6)
        for (i, t, v) in lexer.get_tokens_unprocessed(match.group(6)):
            if v in args:
                yield (i + posn, Name.Variable, v)
            else:
                yield (i + posn, t, v)

    def LHS(lexer, match, ctx=None):
        if False:
            while True:
                i = 10
        s = match.start()
        for (i, t, v) in lexer.get_tokens_unprocessed(match.group()):
            new_t = Name.Function if t is Name else t
            yield (i + s, new_t, v)
        if ctx:
            ctx.pos = match.end()
    tokens = {'root': [('\\(\\*', Comment, 'comment'), ('\\s+', Whitespace), ('(?s)".*?"', String), ('\\\\\\[\\w*?\\]', Error), include('numbers'), include('names'), include('symbols')], 'numbers': [('[\\+\\-]?[0-9]+\\.[0-9]*[eE]?[\\+\\-]?[0-9]*', Number.Float), ('[\\+\\-]?[0-9]+', Number.Integer)], 'symbols': [('[!`\\$\\+\\-/*=^:<>@~.]', Operator), ('[\\[\\]\\(\\){}]', Punctuation), ('[\\,;_&\\?]', Punctuation)], 'comment': [('[^\\*\\(\\)]+', Comment), ('\\(\\*', Comment, '#push'), ('\\*\\)', Comment, '#pop'), ('[\\(\\)\\*]', Comment)], 'names': [('[A-Z][A-Za-z0-9]*', Name.Builtin), ('^([A-Za-z0-9,\\s{}]+)(\\[?.*?\\]?)(\\s*)(:?=)(\\s*)(.*?)$', bygroups(LHS, using(this), Whitespace, Operator, Whitespace, using(this))), ('[a-z][A-Za-z0-9]*', Name), ('#[0-9]*', Name.Variable)]}