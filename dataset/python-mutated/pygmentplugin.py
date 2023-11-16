from pygments import highlight
from pygments.formatters.html import HtmlFormatter
from pygments.lexer import bygroups
from pygments.lexer import DelegatingLexer
from pygments.lexer import include
from pygments.lexer import RegexLexer
from pygments.lexer import using
from pygments.lexers.agile import Python3Lexer
from pygments.lexers.agile import PythonLexer
from pygments.lexers.web import CssLexer
from pygments.lexers.web import HtmlLexer
from pygments.lexers.web import JavascriptLexer
from pygments.lexers.web import XmlLexer
from pygments.token import Comment
from pygments.token import Keyword
from pygments.token import Name
from pygments.token import Operator
from pygments.token import Other
from pygments.token import String
from pygments.token import Text

class MakoLexer(RegexLexer):
    name = 'Mako'
    aliases = ['mako']
    filenames = ['*.mao']
    tokens = {'root': [('(\\s*)(\\%)(\\s*end(?:\\w+))(\\n|\\Z)', bygroups(Text, Comment.Preproc, Keyword, Other)), ('(\\s*)(\\%(?!%))([^\\n]*)(\\n|\\Z)', bygroups(Text, Comment.Preproc, using(PythonLexer), Other)), ('(\\s*)(##[^\\n]*)(\\n|\\Z)', bygroups(Text, Comment.Preproc, Other)), ('(?s)<%doc>.*?</%doc>', Comment.Preproc), ('(<%)([\\w\\.\\:]+)', bygroups(Comment.Preproc, Name.Builtin), 'tag'), ('(</%)([\\w\\.\\:]+)(>)', bygroups(Comment.Preproc, Name.Builtin, Comment.Preproc)), ('<%(?=([\\w\\.\\:]+))', Comment.Preproc, 'ondeftags'), ('(?s)(<%(?:!?))(.*?)(%>)', bygroups(Comment.Preproc, using(PythonLexer), Comment.Preproc)), ('(\\$\\{)(.*?)(\\})', bygroups(Comment.Preproc, using(PythonLexer), Comment.Preproc)), ("(?sx)\n                (.+?)               # anything, followed by:\n                (?:\n                 (?<=\\n)(?=%(?!%)|\\#\\#) |  # an eval or comment line\n                 (?=\\#\\*) |          # multiline comment\n                 (?=</?%) |         # a python block\n                                    # call start or end\n                 (?=\\$\\{) |         # a substitution\n                 (?<=\\n)(?=\\s*%) |\n                                    # - don't consume\n                 (\\\\\\n) |           # an escaped newline\n                 \\Z                 # end of string\n                )\n            ", bygroups(Other, Operator)), ('\\s+', Text)], 'ondeftags': [('<%', Comment.Preproc), ('(?<=<%)(include|inherit|namespace|page)', Name.Builtin), include('tag')], 'tag': [('((?:\\w+)\\s*=)\\s*(".*?")', bygroups(Name.Attribute, String)), ('/?\\s*>', Comment.Preproc, '#pop'), ('\\s+', Text)], 'attr': [('".*?"', String, '#pop'), ("'.*?'", String, '#pop'), ('[^\\s>]+', String, '#pop')]}

class MakoHtmlLexer(DelegatingLexer):
    name = 'HTML+Mako'
    aliases = ['html+mako']

    def __init__(self, **options):
        if False:
            print('Hello World!')
        super().__init__(HtmlLexer, MakoLexer, **options)

class MakoXmlLexer(DelegatingLexer):
    name = 'XML+Mako'
    aliases = ['xml+mako']

    def __init__(self, **options):
        if False:
            while True:
                i = 10
        super().__init__(XmlLexer, MakoLexer, **options)

class MakoJavascriptLexer(DelegatingLexer):
    name = 'JavaScript+Mako'
    aliases = ['js+mako', 'javascript+mako']

    def __init__(self, **options):
        if False:
            return 10
        super().__init__(JavascriptLexer, MakoLexer, **options)

class MakoCssLexer(DelegatingLexer):
    name = 'CSS+Mako'
    aliases = ['css+mako']

    def __init__(self, **options):
        if False:
            i = 10
            return i + 15
        super().__init__(CssLexer, MakoLexer, **options)
pygments_html_formatter = HtmlFormatter(cssclass='syntax-highlighted', linenos=True)

def syntax_highlight(filename='', language=None):
    if False:
        for i in range(10):
            print('nop')
    mako_lexer = MakoLexer()
    python_lexer = Python3Lexer()
    if filename.startswith('memory:') or language == 'mako':
        return lambda string: highlight(string, mako_lexer, pygments_html_formatter)
    return lambda string: highlight(string, python_lexer, pygments_html_formatter)