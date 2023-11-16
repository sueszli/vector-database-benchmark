from __future__ import with_statement
import unittest
from bottle import SimpleTemplate, TemplateError, view, template, touni, tob, html_quote
import re, os
import traceback
from .tools import chdir

class TestSimpleTemplate(unittest.TestCase):

    def assertRenders(self, tpl, to, *args, **vars):
        if False:
            i = 10
            return i + 15
        if isinstance(tpl, str):
            tpl = SimpleTemplate(tpl, lookup=[os.path.join(os.path.dirname(__file__), 'views')])
        self.assertEqual(touni(to), tpl.render(*args, **vars))

    def test_string(self):
        if False:
            for i in range(10):
                print('nop')
        ' Templates: Parse string'
        self.assertRenders('start {{var}} end', 'start var end', var='var')

    def test_self_as_variable_name(self):
        if False:
            while True:
                i = 10
        self.assertRenders('start {{self}} end', 'start var end', {'self': 'var'})

    def test_file(self):
        if False:
            i = 10
            return i + 15
        with chdir(__file__):
            t = SimpleTemplate(name='./views/stpl_simple.tpl', lookup=['.'])
            self.assertRenders(t, 'start var end\n', var='var')

    def test_name(self):
        if False:
            print('Hello World!')
        with chdir(__file__):
            t = SimpleTemplate(name='stpl_simple', lookup=['./views/'])
            self.assertRenders(t, 'start var end\n', var='var')

    def test_unicode(self):
        if False:
            i = 10
            return i + 15
        self.assertRenders('start {{var}} end', 'start äöü end', var=touni('äöü'))
        self.assertRenders('start {{var}} end', 'start äöü end', var=tob('äöü'))

    def test_unicode_code(self):
        if False:
            print('Hello World!')
        ' Templates: utf8 code in file'
        with chdir(__file__):
            t = SimpleTemplate(name='./views/stpl_unicode.tpl', lookup=['.'])
            self.assertRenders(t, 'start ñç äöü end\n', var=touni('äöü'))

    def test_import(self):
        if False:
            i = 10
            return i + 15
        ' Templates: import statement'
        t = '%from base64 import b64encode\nstart {{b64encode(var.encode("ascii") if hasattr(var, "encode") else var)}} end'
        self.assertRenders(t, 'start dmFy end', var='var')

    def test_data(self):
        if False:
            return 10
        ' Templates: Data representation '
        t = SimpleTemplate('<{{var}}>')
        self.assertRenders('<{{var}}>', '<True>', var=True)
        self.assertRenders('<{{var}}>', '<False>', var=False)
        self.assertRenders('<{{var}}>', '<>', var=None)
        self.assertRenders('<{{var}}>', '<0>', var=0)
        self.assertRenders('<{{var}}>', '<5>', var=5)
        self.assertRenders('<{{var}}>', '<b>', var=tob('b'))
        self.assertRenders('<{{var}}>', '<1.0>', var=1.0)
        self.assertRenders('<{{var}}>', '<[1, 2]>', var=[1, 2])

    def test_htmlutils_quote(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual('"&lt;&#039;&#13;&#10;&#9;&quot;\\&gt;"', html_quote('<\'\r\n\t"\\>'))

    def test_escape(self):
        if False:
            return 10
        self.assertRenders('<{{var}}>', '<b>', var='b')
        self.assertRenders('<{{var}}>', '<&lt;&amp;&gt;>', var='<&>')

    def test_noescape(self):
        if False:
            return 10
        self.assertRenders('<{{!var}}>', '<b>', var='b')
        self.assertRenders('<{{!var}}>', '<<&>>', var='<&>')

    def test_noescape_setting(self):
        if False:
            return 10
        t = SimpleTemplate('<{{var}}>', noescape=True)
        self.assertRenders(t, '<b>', var='b')
        self.assertRenders(t, '<<&>>', var='<&>')
        t = SimpleTemplate('<{{!var}}>', noescape=True)
        self.assertRenders(t, '<b>', var='b')
        self.assertRenders(t, '<&lt;&amp;&gt;>', var='<&>')

    def test_blocks(self):
        if False:
            print('Hello World!')
        ' Templates: Code blocks and loops '
        t = 'start\n%for i in l:\n{{i}} \n%end\nend'
        self.assertRenders(t, 'start\n1 \n2 \n3 \nend', l=[1, 2, 3])
        self.assertRenders(t, 'start\nend', l=[])
        t = 'start\n%if i:\n{{i}} \n%end\nend'
        self.assertRenders(t, 'start\nTrue \nend', i=True)
        self.assertRenders(t, 'start\nend', i=False)

    def test_elsebug(self):
        if False:
            for i in range(10):
                print('nop')
        ' Whirespace between block keyword and colon is allowed '
        self.assertRenders('%if 1:\nyes\n%else:\nno\n%end\n', 'yes\n')
        self.assertRenders('%if 1:\nyes\n%else     :\nno\n%end\n', 'yes\n')

    def test_commentbug(self):
        if False:
            return 10
        ' A "#" sign within an string is not a comment '
        self.assertRenders("%if '#':\nyes\n%end\n", 'yes\n')

    def test_multiline(self):
        if False:
            i = 10
            return i + 15
        ' Block statements with non-terminating newlines '
        self.assertRenders('%if 1\\\n%and 1:\nyes\n%end\n', 'yes\n')

    def test_newline_in_parameterlist(self):
        if False:
            print('Hello World!')
        ' Block statements with non-terminating newlines in list '
        self.assertRenders('%a=[1,\n%2]\n{{len(a)}}', '2')

    def test_dedentbug(self):
        if False:
            while True:
                i = 10
        ' One-Line dednet blocks should not change indention '
        t = '%if x: a="if"\n%else: a="else"\n%end\n{{a}}'
        self.assertRenders(t, 'if', x=True)
        self.assertRenders(t, 'else', x=False)
        t = '%if x:\n%a="if"\n%else: a="else"\n%end\n{{a}}'
        self.assertRenders(t, 'if', x=True)
        self.assertRenders(t, 'else', x=False)
        t = SimpleTemplate('%if x: a="if"\n%else: a="else"\n%end')
        self.assertRaises(NameError, t.render)

    def test_onelinebugs(self):
        if False:
            for i in range(10):
                print('nop')
        ' One-Line blocks should not change indention '
        t = '%if x:\n%a=1\n%end\n{{a}}'
        self.assertRenders(t, '1', x=True)
        t = '%if x: a=1; end\n{{a}}'
        self.assertRenders(t, '1', x=True)
        t = '%if x:\n%a=1\n%else:\n%a=2\n%end\n{{a}}'
        self.assertRenders(t, '1', x=True)
        self.assertRenders(t, '2', x=False)
        t = '%if x:   a=1\n%else:\n%a=2\n%end\n{{a}}'
        self.assertRenders(t, '1', x=True)
        self.assertRenders(t, '2', x=False)
        t = '%if x:\n%a=1\n%else:   a=2; end\n{{a}}'
        self.assertRenders(t, '1', x=True)
        self.assertRenders(t, '2', x=False)
        t = '%if x:   a=1\n%else:   a=2; end\n{{a}}'
        self.assertRenders(t, '1', x=True)
        self.assertRenders(t, '2', x=False)

    def test_onelineblocks(self):
        if False:
            for i in range(10):
                print('nop')
        ' Templates: one line code blocks '
        t = "start\n%a=''\n%for i in l: a += str(i); end\n{{a}}\nend"
        self.assertRenders(t, 'start\n123\nend', l=[1, 2, 3])
        self.assertRenders(t, 'start\n\nend', l=[])

    def test_escaped_codelines(self):
        if False:
            i = 10
            return i + 15
        self.assertRenders('\\% test', '% test')
        self.assertRenders('\\%% test', '%% test')
        self.assertRenders('    \\% test', '    % test')

    def test_nobreak(self):
        if False:
            i = 10
            return i + 15
        ' Templates: Nobreak statements'
        self.assertRenders('start\\\\\n%pass\nend', 'startend')

    def test_nonobreak(self):
        if False:
            return 10
        ' Templates: Escaped nobreak statements'
        self.assertRenders('start\\\\\n\\\\\n%pass\nend', 'start\\\\\nend')

    def test_include(self):
        if False:
            i = 10
            return i + 15
        ' Templates: Include statements'
        with chdir(__file__):
            t = SimpleTemplate(name='stpl_include', lookup=['./views/'])
            self.assertRenders(t, 'before\nstart var end\nafter\n', var='var')

    def test_rebase(self):
        if False:
            while True:
                i = 10
        ' Templates: %rebase and method passing '
        with chdir(__file__):
            t = SimpleTemplate(name='stpl_t2main', lookup=['./views/'])
            result = '+base+\n+main+\n!1234!\n+include+\n-main-\n+include+\n-base-\n'
            self.assertRenders(t, result, content='1234')

    def test_get(self):
        if False:
            print('Hello World!')
        self.assertRenders('{{get("x", "default")}}', '1234', x='1234')
        self.assertRenders('{{get("x", "default")}}', 'default')

    def test_setdefault(self):
        if False:
            for i in range(10):
                print('nop')
        t = '%setdefault("x", "default")\n{{x}}'
        self.assertRenders(t, '1234', x='1234')
        self.assertRenders(t, 'default')

    def test_defnied(self):
        if False:
            return 10
        self.assertRenders('{{x if defined("x") else "no"}}', 'yes', x='yes')
        self.assertRenders('{{x if defined("x") else "no"}}', 'no')

    def test_notfound(self):
        if False:
            i = 10
            return i + 15
        ' Templates: Unavailable templates'
        self.assertRaises(TemplateError, SimpleTemplate, name='abcdef', lookup=['.'])

    def test_error(self):
        if False:
            i = 10
            return i + 15
        ' Templates: Exceptions'
        self.assertRaises(SyntaxError, lambda : SimpleTemplate('%for badsyntax').co)
        self.assertRaises(IndexError, SimpleTemplate('{{i[5]}}', lookup=['.']).render, i=[0])

    def test_winbreaks(self):
        if False:
            for i in range(10):
                print('nop')
        ' Templates: Test windows line breaks '
        self.assertRenders('%var+=1\r\n{{var}}\r\n', '6\r\n', var=5)

    def test_winbreaks_end_bug(self):
        if False:
            for i in range(10):
                print('nop')
        d = {'test': [1, 2, 3]}
        self.assertRenders('%for i in test:\n{{i}}\n%end\n', '1\n2\n3\n', **d)
        self.assertRenders('%for i in test:\n{{i}}\r\n%end\n', '1\r\n2\r\n3\r\n', **d)
        self.assertRenders('%for i in test:\r\n{{i}}\n%end\r\n', '1\n2\n3\n', **d)
        self.assertRenders('%for i in test:\r\n{{i}}\r\n%end\r\n', '1\r\n2\r\n3\r\n', **d)

    def test_commentonly(self):
        if False:
            print('Hello World!')
        ' Templates: Commentd should behave like code-lines (e.g. flush text-lines) '
        t = SimpleTemplate('...\n%#test\n...')
        self.assertNotEqual('#test', t.code.splitlines()[0])

    def test_template_shortcut(self):
        if False:
            print('Hello World!')
        result = template('start {{var}} end', var='middle')
        self.assertEqual(touni('start middle end'), result)

    def test_view_decorator(self):
        if False:
            print('Hello World!')

        @view('start {{var}} end')
        def test():
            if False:
                print('Hello World!')
            return dict(var='middle')
        self.assertEqual(touni('start middle end'), test())

    def test_view_decorator_issue_407(self):
        if False:
            for i in range(10):
                print('nop')
        with chdir(__file__):

            @view('stpl_no_vars')
            def test():
                if False:
                    while True:
                        i = 10
                pass
            self.assertEqual(touni('hihi'), test())

            @view('aaa {{x}}', x='bbb')
            def test2():
                if False:
                    print('Hello World!')
                pass
            self.assertEqual(touni('aaa bbb'), test2())

    def test_global_config(self):
        if False:
            while True:
                i = 10
        SimpleTemplate.global_config('meh', 1)
        t = SimpleTemplate('anything')
        self.assertEqual(touni('anything'), t.render())

    def test_bug_no_whitespace_before_stmt(self):
        if False:
            i = 10
            return i + 15
        self.assertRenders('\n{{var}}', '\nx', var='x')

    def test_bug_block_keywords_eat_prefixed_code(self):
        if False:
            print('Hello World!')
        " #595: Everything before an 'if' statement is removed, resulting in\n            SyntaxError. "
        tpl = "% m = 'x' if True else 'y'\n{{m}}"
        self.assertRenders(tpl, 'x')

class TestSTPLDir(unittest.TestCase):

    def fix_ident(self, string):
        if False:
            while True:
                i = 10
        lines = string.splitlines(True)
        if not lines:
            return string
        if not lines[0].strip():
            lines.pop(0)
        whitespace = re.match('([ \t]*)', lines[0]).group(0)
        if not whitespace:
            return string
        for i in range(len(lines)):
            lines[i] = lines[i][len(whitespace):]
        return lines[0][:0].join(lines)

    def assertRenders(self, source, result, syntax=None, *args, **vars):
        if False:
            print('Hello World!')
        source = self.fix_ident(source)
        result = self.fix_ident(result)
        tpl = SimpleTemplate(source, syntax=syntax)
        try:
            tpl.co
            self.assertEqual(touni(result), tpl.render(*args, **vars))
        except SyntaxError:
            self.fail('Syntax error in template:\n%s\n\nTemplate code:\n##########\n%s\n##########' % (traceback.format_exc(), tpl.code))

    def test_multiline_block(self):
        if False:
            i = 10
            return i + 15
        source = '\n            <% a = 5\n            b = 6\n            c = 7 %>\n            {{a+b+c}}\n        '
        result = '\n            18\n        '
        self.assertRenders(source, result)
        source_wineol = '<% a = 5\r\nb = 6\r\nc = 7\r\n%>\r\n{{a+b+c}}'
        result_wineol = '18'
        self.assertRenders(source_wineol, result_wineol)

    def test_multiline_ignore_eob_in_string(self):
        if False:
            print('Hello World!')
        source = "\n            <% x=5 # a comment\n               y = '%>' # a string\n               # this is still code\n               # lets end this %>\n            {{x}}{{!y}}\n        "
        result = '\n            5%>\n        '
        self.assertRenders(source, result)

    def test_multiline_find_eob_in_comments(self):
        if False:
            for i in range(10):
                print('nop')
        source = '\n            <% # a comment\n               # %> ignore because not end of line\n               # this is still code\n               x=5\n               # lets end this here %>\n            {{x}}\n        '
        result = '\n            5\n        '
        self.assertRenders(source, result)

    def test_multiline_indention(self):
        if False:
            i = 10
            return i + 15
        source = '\n            <%   if True:\n                   a = 2\n                     else:\n                       a = 0\n                         end\n            %>\n            {{a}}\n        '
        result = '\n            2\n        '
        self.assertRenders(source, result)

    def test_multiline_eob_after_end(self):
        if False:
            print('Hello World!')
        source = '\n            <%   if True:\n                   a = 2\n                 end %>\n            {{a}}\n        '
        result = '\n            2\n        '
        self.assertRenders(source, result)

    def test_multiline_eob_in_single_line_code(self):
        if False:
            for i in range(10):
                print('nop')
        source = '\n            cline eob=5; eob\n            xxx\n        '
        result = '\n            xxx\n        '
        self.assertRenders(source, result, syntax='sob eob cline foo bar')

    def test_multiline_strings_in_code_line(self):
        if False:
            for i in range(10):
                print('nop')
        source = '\n            % a = """line 1\n                  line 2"""\n            {{a}}\n        '
        result = '\n            line 1\n                  line 2\n        '
        self.assertRenders(source, result)

    def test_multiline_comprehensions_in_code_line(self):
        if False:
            while True:
                i = 10
        self.assertRenders(source='\n            % a = [\n            %    (i + 1)\n            %    for i in range(5)\n            %    if i%2 == 0\n            % ]\n            {{a}}\n        ', result='\n            [1, 3, 5]\n        ')

    def test_end_keyword_on_same_line(self):
        if False:
            i = 10
            return i + 15
        self.assertRenders('\n            % if 1:\n            %    1; end\n            foo\n        ', '\n            foo\n        ')