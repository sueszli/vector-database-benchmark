import pytest
from jinja2 import Environment
from jinja2 import nodes
from jinja2 import Template
from jinja2 import TemplateSyntaxError
from jinja2 import UndefinedError
from jinja2.lexer import Token
from jinja2.lexer import TOKEN_BLOCK_BEGIN
from jinja2.lexer import TOKEN_BLOCK_END
from jinja2.lexer import TOKEN_EOF
from jinja2.lexer import TokenStream

class TestTokenStream:
    test_tokens = [Token(1, TOKEN_BLOCK_BEGIN, ''), Token(2, TOKEN_BLOCK_END, '')]

    def test_simple(self, env):
        if False:
            while True:
                i = 10
        ts = TokenStream(self.test_tokens, 'foo', 'bar')
        assert ts.current.type is TOKEN_BLOCK_BEGIN
        assert bool(ts)
        assert not bool(ts.eos)
        next(ts)
        assert ts.current.type is TOKEN_BLOCK_END
        assert bool(ts)
        assert not bool(ts.eos)
        next(ts)
        assert ts.current.type is TOKEN_EOF
        assert not bool(ts)
        assert bool(ts.eos)

    def test_iter(self, env):
        if False:
            for i in range(10):
                print('nop')
        token_types = [t.type for t in TokenStream(self.test_tokens, 'foo', 'bar')]
        assert token_types == ['block_begin', 'block_end']

class TestLexer:

    def test_raw1(self, env):
        if False:
            for i in range(10):
                print('nop')
        tmpl = env.from_string('{% raw %}foo{% endraw %}|{%raw%}{{ bar }}|{% baz %}{%       endraw    %}')
        assert tmpl.render() == 'foo|{{ bar }}|{% baz %}'

    def test_raw2(self, env):
        if False:
            return 10
        tmpl = env.from_string('1  {%- raw -%}   2   {%- endraw -%}   3')
        assert tmpl.render() == '123'

    def test_raw3(self, env):
        if False:
            return 10
        env = Environment(lstrip_blocks=True, trim_blocks=True)
        tmpl = env.from_string('bar\n{% raw %}\n  {{baz}}2 spaces\n{% endraw %}\nfoo')
        assert tmpl.render(baz='test') == 'bar\n\n  {{baz}}2 spaces\nfoo'

    def test_raw4(self, env):
        if False:
            for i in range(10):
                print('nop')
        env = Environment(lstrip_blocks=True, trim_blocks=False)
        tmpl = env.from_string('bar\n{%- raw -%}\n\n  \n  2 spaces\n space{%- endraw -%}\nfoo')
        assert tmpl.render() == 'bar2 spaces\n spacefoo'

    def test_balancing(self, env):
        if False:
            i = 10
            return i + 15
        env = Environment('{%', '%}', '${', '}')
        tmpl = env.from_string("{% for item in seq\n            %}${{'foo': item}|upper}{% endfor %}")
        assert tmpl.render(seq=list(range(3))) == "{'FOO': 0}{'FOO': 1}{'FOO': 2}"

    def test_comments(self, env):
        if False:
            while True:
                i = 10
        env = Environment('<!--', '-->', '{', '}')
        tmpl = env.from_string('<ul>\n<!--- for item in seq -->\n  <li>{item}</li>\n<!--- endfor -->\n</ul>')
        assert tmpl.render(seq=list(range(3))) == '<ul>\n  <li>0</li>\n  <li>1</li>\n  <li>2</li>\n</ul>'

    def test_string_escapes(self, env):
        if False:
            return 10
        for char in ('\x00', '♨', 'ä', '\t', '\r', '\n'):
            tmpl = env.from_string(f'{{{{ {char!r} }}}}')
            assert tmpl.render() == char
        assert env.from_string('{{ "♨" }}').render() == '♨'

    def test_bytefallback(self, env):
        if False:
            i = 10
            return i + 15
        from pprint import pformat
        tmpl = env.from_string("{{ 'foo'|pprint }}|{{ 'bär'|pprint }}")
        assert tmpl.render() == pformat('foo') + '|' + pformat('bär')

    def test_operators(self, env):
        if False:
            return 10
        from jinja2.lexer import operators
        for (test, expect) in operators.items():
            if test in '([{}])':
                continue
            stream = env.lexer.tokenize(f'{{{{ {test} }}}}')
            next(stream)
            assert stream.current.type == expect

    def test_normalizing(self, env):
        if False:
            print('Hello World!')
        for seq in ('\r', '\r\n', '\n'):
            env = Environment(newline_sequence=seq)
            tmpl = env.from_string('1\n2\r\n3\n4\n')
            result = tmpl.render()
            assert result.replace(seq, 'X') == '1X2X3X4'

    def test_trailing_newline(self, env):
        if False:
            return 10
        for keep in [True, False]:
            env = Environment(keep_trailing_newline=keep)
            for (template, expected) in [('', {}), ('no\nnewline', {}), ('with\nnewline\n', {False: 'with\nnewline'}), ('with\nseveral\n\n\n', {False: 'with\nseveral\n\n'})]:
                tmpl = env.from_string(template)
                expect = expected.get(keep, template)
                result = tmpl.render()
                assert result == expect, (keep, template, result, expect)

    @pytest.mark.parametrize(('name', 'valid'), [('foo', True), ('föö', True), ('き', True), ('_', True), ('1a', False), ('a-', False), ('🐍a', False), ('a🐍🐍', False), ('ᢅ', True), ('ᢆ', True), ('℘', True), ('℮', True), ('·', False), ('a·', True)])
    def test_name(self, env, name, valid):
        if False:
            return 10
        t = '{{ ' + name + ' }}'
        if valid:
            env.from_string(t)
        else:
            pytest.raises(TemplateSyntaxError, env.from_string, t)

    def test_lineno_with_strip(self, env):
        if False:
            for i in range(10):
                print('nop')
        tokens = env.lex('<html>\n    <body>\n    {%- block content -%}\n        <hr>\n        {{ item }}\n    {% endblock %}\n    </body>\n</html>')
        for tok in tokens:
            (lineno, token_type, value) = tok
            if token_type == 'name' and value == 'item':
                assert lineno == 5
                break

class TestParser:

    def test_php_syntax(self, env):
        if False:
            return 10
        env = Environment('<?', '?>', '<?=', '?>', '<!--', '-->')
        tmpl = env.from_string("<!-- I'm a comment, I'm not interesting --><? for item in seq -?>\n    <?= item ?>\n<?- endfor ?>")
        assert tmpl.render(seq=list(range(5))) == '01234'

    def test_erb_syntax(self, env):
        if False:
            return 10
        env = Environment('<%', '%>', '<%=', '%>', '<%#', '%>')
        tmpl = env.from_string("<%# I'm a comment, I'm not interesting %><% for item in seq -%>\n    <%= item %>\n<%- endfor %>")
        assert tmpl.render(seq=list(range(5))) == '01234'

    def test_comment_syntax(self, env):
        if False:
            i = 10
            return i + 15
        env = Environment('<!--', '-->', '${', '}', '<!--#', '-->')
        tmpl = env.from_string("<!--# I'm a comment, I'm not interesting --><!-- for item in seq --->\n    ${item}\n<!--- endfor -->")
        assert tmpl.render(seq=list(range(5))) == '01234'

    def test_balancing(self, env):
        if False:
            return 10
        tmpl = env.from_string("{{{'foo':'bar'}.foo}}")
        assert tmpl.render() == 'bar'

    def test_start_comment(self, env):
        if False:
            i = 10
            return i + 15
        tmpl = env.from_string('{# foo comment\nand bar comment #}\n{% macro blub() %}foo{% endmacro %}\n{{ blub() }}')
        assert tmpl.render().strip() == 'foo'

    def test_line_syntax(self, env):
        if False:
            for i in range(10):
                print('nop')
        env = Environment('<%', '%>', '${', '}', '<%#', '%>', '%')
        tmpl = env.from_string('<%# regular comment %>\n% for item in seq:\n    ${item}\n% endfor')
        assert [int(x.strip()) for x in tmpl.render(seq=list(range(5))).split()] == list(range(5))
        env = Environment('<%', '%>', '${', '}', '<%#', '%>', '%', '##')
        tmpl = env.from_string('<%# regular comment %>\n% for item in seq:\n    ${item} ## the rest of the stuff\n% endfor')
        assert [int(x.strip()) for x in tmpl.render(seq=list(range(5))).split()] == list(range(5))

    def test_line_syntax_priority(self, env):
        if False:
            while True:
                i = 10
        env = Environment('{%', '%}', '${', '}', '/*', '*/', '##', '#')
        tmpl = env.from_string("/* ignore me.\n   I'm a multiline comment */\n## for item in seq:\n* ${item}          # this is just extra stuff\n## endfor")
        assert tmpl.render(seq=[1, 2]).strip() == '* 1\n* 2'
        env = Environment('{%', '%}', '${', '}', '/*', '*/', '#', '##')
        tmpl = env.from_string("/* ignore me.\n   I'm a multiline comment */\n# for item in seq:\n* ${item}          ## this is just extra stuff\n    ## extra stuff i just want to ignore\n# endfor")
        assert tmpl.render(seq=[1, 2]).strip() == '* 1\n\n* 2'

    def test_error_messages(self, env):
        if False:
            for i in range(10):
                print('nop')

        def assert_error(code, expected):
            if False:
                print('Hello World!')
            with pytest.raises(TemplateSyntaxError, match=expected):
                Template(code)
        assert_error('{% for item in seq %}...{% endif %}', "Encountered unknown tag 'endif'. Jinja was looking for the following tags: 'endfor' or 'else'. The innermost block that needs to be closed is 'for'.")
        assert_error('{% if foo %}{% for item in seq %}...{% endfor %}{% endfor %}', "Encountered unknown tag 'endfor'. Jinja was looking for the following tags: 'elif' or 'else' or 'endif'. The innermost block that needs to be closed is 'if'.")
        assert_error('{% if foo %}', "Unexpected end of template. Jinja was looking for the following tags: 'elif' or 'else' or 'endif'. The innermost block that needs to be closed is 'if'.")
        assert_error('{% for item in seq %}', "Unexpected end of template. Jinja was looking for the following tags: 'endfor' or 'else'. The innermost block that needs to be closed is 'for'.")
        assert_error('{% block foo-bar-baz %}', 'Block names in Jinja have to be valid Python identifiers and may not contain hyphens, use an underscore instead.')
        assert_error('{% unknown_tag %}', "Encountered unknown tag 'unknown_tag'.")

class TestSyntax:

    def test_call(self, env):
        if False:
            i = 10
            return i + 15
        env = Environment()
        env.globals['foo'] = lambda a, b, c, e, g: a + b + c + e + g
        tmpl = env.from_string("{{ foo('a', c='d', e='f', *['b'], **{'g': 'h'}) }}")
        assert tmpl.render() == 'abdfh'

    def test_slicing(self, env):
        if False:
            return 10
        tmpl = env.from_string('{{ [1, 2, 3][:] }}|{{ [1, 2, 3][::-1] }}')
        assert tmpl.render() == '[1, 2, 3]|[3, 2, 1]'

    def test_attr(self, env):
        if False:
            while True:
                i = 10
        tmpl = env.from_string("{{ foo.bar }}|{{ foo['bar'] }}")
        assert tmpl.render(foo={'bar': 42}) == '42|42'

    def test_subscript(self, env):
        if False:
            return 10
        tmpl = env.from_string('{{ foo[0] }}|{{ foo[-1] }}')
        assert tmpl.render(foo=[0, 1, 2]) == '0|2'

    def test_tuple(self, env):
        if False:
            for i in range(10):
                print('nop')
        tmpl = env.from_string('{{ () }}|{{ (1,) }}|{{ (1, 2) }}')
        assert tmpl.render() == '()|(1,)|(1, 2)'

    def test_math(self, env):
        if False:
            while True:
                i = 10
        tmpl = env.from_string('{{ (1 + 1 * 2) - 3 / 2 }}|{{ 2**3 }}')
        assert tmpl.render() == '1.5|8'

    def test_div(self, env):
        if False:
            i = 10
            return i + 15
        tmpl = env.from_string('{{ 3 // 2 }}|{{ 3 / 2 }}|{{ 3 % 2 }}')
        assert tmpl.render() == '1|1.5|1'

    def test_unary(self, env):
        if False:
            i = 10
            return i + 15
        tmpl = env.from_string('{{ +3 }}|{{ -3 }}')
        assert tmpl.render() == '3|-3'

    def test_concat(self, env):
        if False:
            while True:
                i = 10
        tmpl = env.from_string("{{ [1, 2] ~ 'foo' }}")
        assert tmpl.render() == '[1, 2]foo'

    @pytest.mark.parametrize(('a', 'op', 'b'), [(1, '>', 0), (1, '>=', 1), (2, '<', 3), (3, '<=', 4), (4, '==', 4), (4, '!=', 5)])
    def test_compare(self, env, a, op, b):
        if False:
            i = 10
            return i + 15
        t = env.from_string(f'{{{{ {a} {op} {b} }}}}')
        assert t.render() == 'True'

    def test_compare_parens(self, env):
        if False:
            while True:
                i = 10
        t = env.from_string('{{ i * (j < 5) }}')
        assert t.render(i=2, j=3) == '2'

    @pytest.mark.parametrize(('src', 'expect'), [('{{ 4 < 2 < 3 }}', 'False'), ('{{ a < b < c }}', 'False'), ('{{ 4 > 2 > 3 }}', 'False'), ('{{ a > b > c }}', 'False'), ('{{ 4 > 2 < 3 }}', 'True'), ('{{ a > b < c }}', 'True')])
    def test_compare_compound(self, env, src, expect):
        if False:
            return 10
        t = env.from_string(src)
        assert t.render(a=4, b=2, c=3) == expect

    def test_inop(self, env):
        if False:
            print('Hello World!')
        tmpl = env.from_string('{{ 1 in [1, 2, 3] }}|{{ 1 not in [1, 2, 3] }}')
        assert tmpl.render() == 'True|False'

    @pytest.mark.parametrize('value', ('[]', '{}', '()'))
    def test_collection_literal(self, env, value):
        if False:
            for i in range(10):
                print('nop')
        t = env.from_string(f'{{{{ {value} }}}}')
        assert t.render() == value

    @pytest.mark.parametrize(('value', 'expect'), (('1', '1'), ('123', '123'), ('12_34_56', '123456'), ('1.2', '1.2'), ('34.56', '34.56'), ('3_4.5_6', '34.56'), ('1e0', '1.0'), ('10e1', '100.0'), ('2.5e100', '2.5e+100'), ('2.5e+100', '2.5e+100'), ('25.6e-10', '2.56e-09'), ('1_2.3_4e5_6', '1.234e+57'), ('0', '0'), ('0_00', '0'), ('0b1001_1111', '159'), ('0o123', '83'), ('0o1_23', '83'), ('0x123abc', '1194684'), ('0x12_3abc', '1194684')))
    def test_numeric_literal(self, env, value, expect):
        if False:
            for i in range(10):
                print('nop')
        t = env.from_string(f'{{{{ {value} }}}}')
        assert t.render() == expect

    def test_bool(self, env):
        if False:
            print('Hello World!')
        tmpl = env.from_string('{{ true and false }}|{{ false or true }}|{{ not false }}')
        assert tmpl.render() == 'False|True|True'

    def test_grouping(self, env):
        if False:
            print('Hello World!')
        tmpl = env.from_string('{{ (true and false) or (false and true) and not false }}')
        assert tmpl.render() == 'False'

    def test_django_attr(self, env):
        if False:
            return 10
        tmpl = env.from_string('{{ [1, 2, 3].0 }}|{{ [[1]].0.0 }}')
        assert tmpl.render() == '1|1'

    def test_conditional_expression(self, env):
        if False:
            return 10
        tmpl = env.from_string('{{ 0 if true else 1 }}')
        assert tmpl.render() == '0'

    def test_short_conditional_expression(self, env):
        if False:
            for i in range(10):
                print('nop')
        tmpl = env.from_string('<{{ 1 if false }}>')
        assert tmpl.render() == '<>'
        tmpl = env.from_string('<{{ (1 if false).bar }}>')
        pytest.raises(UndefinedError, tmpl.render)

    def test_filter_priority(self, env):
        if False:
            i = 10
            return i + 15
        tmpl = env.from_string('{{ "foo"|upper + "bar"|upper }}')
        assert tmpl.render() == 'FOOBAR'

    def test_function_calls(self, env):
        if False:
            print('Hello World!')
        tests = [(True, '*foo, bar'), (True, '*foo, *bar'), (True, '**foo, *bar'), (True, '**foo, bar'), (True, '**foo, **bar'), (True, '**foo, bar=42'), (False, 'foo, bar'), (False, 'foo, bar=42'), (False, 'foo, bar=23, *args'), (False, 'foo, *args, bar=23'), (False, 'a, b=c, *d, **e'), (False, '*foo, bar=42'), (False, '*foo, **bar'), (False, '*foo, bar=42, **baz'), (False, 'foo, *args, bar=23, **baz')]
        for (should_fail, sig) in tests:
            if should_fail:
                with pytest.raises(TemplateSyntaxError):
                    env.from_string(f'{{{{ foo({sig}) }}}}')
            else:
                env.from_string(f'foo({sig})')

    def test_tuple_expr(self, env):
        if False:
            for i in range(10):
                print('nop')
        for tmpl in ['{{ () }}', '{{ (1, 2) }}', '{{ (1, 2,) }}', '{{ 1, }}', '{{ 1, 2 }}', '{% for foo, bar in seq %}...{% endfor %}', '{% for x in foo, bar %}...{% endfor %}', '{% for x in foo, %}...{% endfor %}']:
            assert env.from_string(tmpl)

    def test_trailing_comma(self, env):
        if False:
            i = 10
            return i + 15
        tmpl = env.from_string('{{ (1, 2,) }}|{{ [1, 2,] }}|{{ {1: 2,} }}')
        assert tmpl.render().lower() == '(1, 2)|[1, 2]|{1: 2}'

    def test_block_end_name(self, env):
        if False:
            while True:
                i = 10
        env.from_string('{% block foo %}...{% endblock foo %}')
        pytest.raises(TemplateSyntaxError, env.from_string, '{% block x %}{% endblock y %}')

    def test_constant_casing(self, env):
        if False:
            while True:
                i = 10
        for const in (True, False, None):
            const = str(const)
            tmpl = env.from_string(f'{{{{ {const} }}}}|{{{{ {const.lower()} }}}}|{{{{ {const.upper()} }}}}')
            assert tmpl.render() == f'{const}|{const}|'

    def test_test_chaining(self, env):
        if False:
            i = 10
            return i + 15
        pytest.raises(TemplateSyntaxError, env.from_string, '{{ foo is string is sequence }}')
        assert env.from_string('{{ 42 is string or 42 is number }}').render() == 'True'

    def test_string_concatenation(self, env):
        if False:
            i = 10
            return i + 15
        tmpl = env.from_string('{{ "foo" "bar" "baz" }}')
        assert tmpl.render() == 'foobarbaz'

    def test_notin(self, env):
        if False:
            print('Hello World!')
        bar = range(100)
        tmpl = env.from_string('{{ not 42 in bar }}')
        assert tmpl.render(bar=bar) == 'False'

    def test_operator_precedence(self, env):
        if False:
            i = 10
            return i + 15
        tmpl = env.from_string('{{ 2 * 3 + 4 % 2 + 1 - 2 }}')
        assert tmpl.render() == '5'

    def test_implicit_subscribed_tuple(self, env):
        if False:
            for i in range(10):
                print('nop')

        class Foo:

            def __getitem__(self, x):
                if False:
                    i = 10
                    return i + 15
                return x
        t = env.from_string('{{ foo[1, 2] }}')
        assert t.render(foo=Foo()) == '(1, 2)'

    def test_raw2(self, env):
        if False:
            i = 10
            return i + 15
        tmpl = env.from_string('{% raw %}{{ FOO }} and {% BAR %}{% endraw %}')
        assert tmpl.render() == '{{ FOO }} and {% BAR %}'

    def test_const(self, env):
        if False:
            i = 10
            return i + 15
        tmpl = env.from_string('{{ true }}|{{ false }}|{{ none }}|{{ none is defined }}|{{ missing is defined }}')
        assert tmpl.render() == 'True|False|None|True|False'

    def test_neg_filter_priority(self, env):
        if False:
            return 10
        node = env.parse('{{ -1|foo }}')
        assert isinstance(node.body[0].nodes[0], nodes.Filter)
        assert isinstance(node.body[0].nodes[0].node, nodes.Neg)

    def test_const_assign(self, env):
        if False:
            i = 10
            return i + 15
        constass1 = '{% set true = 42 %}'
        constass2 = '{% for none in seq %}{% endfor %}'
        for tmpl in (constass1, constass2):
            pytest.raises(TemplateSyntaxError, env.from_string, tmpl)

    def test_localset(self, env):
        if False:
            for i in range(10):
                print('nop')
        tmpl = env.from_string('{% set foo = 0 %}{% for item in [1, 2] %}{% set foo = 1 %}{% endfor %}{{ foo }}')
        assert tmpl.render() == '0'

    def test_parse_unary(self, env):
        if False:
            print('Hello World!')
        tmpl = env.from_string('{{ -foo["bar"] }}')
        assert tmpl.render(foo={'bar': 42}) == '-42'
        tmpl = env.from_string('{{ -foo["bar"]|abs }}')
        assert tmpl.render(foo={'bar': 42}) == '42'

class TestLstripBlocks:

    def test_lstrip(self, env):
        if False:
            i = 10
            return i + 15
        env = Environment(lstrip_blocks=True, trim_blocks=False)
        tmpl = env.from_string('    {% if True %}\n    {% endif %}')
        assert tmpl.render() == '\n'

    def test_lstrip_trim(self, env):
        if False:
            return 10
        env = Environment(lstrip_blocks=True, trim_blocks=True)
        tmpl = env.from_string('    {% if True %}\n    {% endif %}')
        assert tmpl.render() == ''

    def test_no_lstrip(self, env):
        if False:
            print('Hello World!')
        env = Environment(lstrip_blocks=True, trim_blocks=False)
        tmpl = env.from_string('    {%+ if True %}\n    {%+ endif %}')
        assert tmpl.render() == '    \n    '

    def test_lstrip_blocks_false_with_no_lstrip(self, env):
        if False:
            return 10
        env = Environment(lstrip_blocks=False, trim_blocks=False)
        tmpl = env.from_string('    {% if True %}\n    {% endif %}')
        assert tmpl.render() == '    \n    '
        tmpl = env.from_string('    {%+ if True %}\n    {%+ endif %}')
        assert tmpl.render() == '    \n    '

    def test_lstrip_endline(self, env):
        if False:
            for i in range(10):
                print('nop')
        env = Environment(lstrip_blocks=True, trim_blocks=False)
        tmpl = env.from_string('    hello{% if True %}\n    goodbye{% endif %}')
        assert tmpl.render() == '    hello\n    goodbye'

    def test_lstrip_inline(self, env):
        if False:
            print('Hello World!')
        env = Environment(lstrip_blocks=True, trim_blocks=False)
        tmpl = env.from_string('    {% if True %}hello    {% endif %}')
        assert tmpl.render() == 'hello    '

    def test_lstrip_nested(self, env):
        if False:
            for i in range(10):
                print('nop')
        env = Environment(lstrip_blocks=True, trim_blocks=False)
        tmpl = env.from_string('    {% if True %}a {% if True %}b {% endif %}c {% endif %}')
        assert tmpl.render() == 'a b c '

    def test_lstrip_left_chars(self, env):
        if False:
            for i in range(10):
                print('nop')
        env = Environment(lstrip_blocks=True, trim_blocks=False)
        tmpl = env.from_string('    abc {% if True %}\n        hello{% endif %}')
        assert tmpl.render() == '    abc \n        hello'

    def test_lstrip_embeded_strings(self, env):
        if False:
            while True:
                i = 10
        env = Environment(lstrip_blocks=True, trim_blocks=False)
        tmpl = env.from_string('    {% set x = " {% str %} " %}{{ x }}')
        assert tmpl.render() == ' {% str %} '

    def test_lstrip_preserve_leading_newlines(self, env):
        if False:
            for i in range(10):
                print('nop')
        env = Environment(lstrip_blocks=True, trim_blocks=False)
        tmpl = env.from_string('\n\n\n{% set hello = 1 %}')
        assert tmpl.render() == '\n\n\n'

    def test_lstrip_comment(self, env):
        if False:
            for i in range(10):
                print('nop')
        env = Environment(lstrip_blocks=True, trim_blocks=False)
        tmpl = env.from_string('    {# if True #}\nhello\n    {#endif#}')
        assert tmpl.render() == '\nhello\n'

    def test_lstrip_angle_bracket_simple(self, env):
        if False:
            while True:
                i = 10
        env = Environment('<%', '%>', '${', '}', '<%#', '%>', '%', '##', lstrip_blocks=True, trim_blocks=True)
        tmpl = env.from_string('    <% if True %>hello    <% endif %>')
        assert tmpl.render() == 'hello    '

    def test_lstrip_angle_bracket_comment(self, env):
        if False:
            while True:
                i = 10
        env = Environment('<%', '%>', '${', '}', '<%#', '%>', '%', '##', lstrip_blocks=True, trim_blocks=True)
        tmpl = env.from_string('    <%# if True %>hello    <%# endif %>')
        assert tmpl.render() == 'hello    '

    def test_lstrip_angle_bracket(self, env):
        if False:
            print('Hello World!')
        env = Environment('<%', '%>', '${', '}', '<%#', '%>', '%', '##', lstrip_blocks=True, trim_blocks=True)
        tmpl = env.from_string('    <%# regular comment %>\n    <% for item in seq %>\n${item} ## the rest of the stuff\n   <% endfor %>')
        assert tmpl.render(seq=range(5)) == ''.join((f'{x}\n' for x in range(5)))

    def test_lstrip_angle_bracket_compact(self, env):
        if False:
            i = 10
            return i + 15
        env = Environment('<%', '%>', '${', '}', '<%#', '%>', '%', '##', lstrip_blocks=True, trim_blocks=True)
        tmpl = env.from_string('    <%#regular comment%>\n    <%for item in seq%>\n${item} ## the rest of the stuff\n   <%endfor%>')
        assert tmpl.render(seq=range(5)) == ''.join((f'{x}\n' for x in range(5)))

    def test_lstrip_blocks_outside_with_new_line(self):
        if False:
            for i in range(10):
                print('nop')
        env = Environment(lstrip_blocks=True, trim_blocks=False)
        tmpl = env.from_string('  {% if kvs %}(\n   {% for k, v in kvs %}{{ k }}={{ v }} {% endfor %}\n  ){% endif %}')
        out = tmpl.render(kvs=[('a', 1), ('b', 2)])
        assert out == '(\na=1 b=2 \n  )'

    def test_lstrip_trim_blocks_outside_with_new_line(self):
        if False:
            print('Hello World!')
        env = Environment(lstrip_blocks=True, trim_blocks=True)
        tmpl = env.from_string('  {% if kvs %}(\n   {% for k, v in kvs %}{{ k }}={{ v }} {% endfor %}\n  ){% endif %}')
        out = tmpl.render(kvs=[('a', 1), ('b', 2)])
        assert out == '(\na=1 b=2   )'

    def test_lstrip_blocks_inside_with_new_line(self):
        if False:
            print('Hello World!')
        env = Environment(lstrip_blocks=True, trim_blocks=False)
        tmpl = env.from_string('  ({% if kvs %}\n   {% for k, v in kvs %}{{ k }}={{ v }} {% endfor %}\n  {% endif %})')
        out = tmpl.render(kvs=[('a', 1), ('b', 2)])
        assert out == '  (\na=1 b=2 \n)'

    def test_lstrip_trim_blocks_inside_with_new_line(self):
        if False:
            return 10
        env = Environment(lstrip_blocks=True, trim_blocks=True)
        tmpl = env.from_string('  ({% if kvs %}\n   {% for k, v in kvs %}{{ k }}={{ v }} {% endfor %}\n  {% endif %})')
        out = tmpl.render(kvs=[('a', 1), ('b', 2)])
        assert out == '  (a=1 b=2 )'

    def test_lstrip_blocks_without_new_line(self):
        if False:
            return 10
        env = Environment(lstrip_blocks=True, trim_blocks=False)
        tmpl = env.from_string('  {% if kvs %}   {% for k, v in kvs %}{{ k }}={{ v }} {% endfor %}  {% endif %}')
        out = tmpl.render(kvs=[('a', 1), ('b', 2)])
        assert out == '   a=1 b=2   '

    def test_lstrip_trim_blocks_without_new_line(self):
        if False:
            i = 10
            return i + 15
        env = Environment(lstrip_blocks=True, trim_blocks=True)
        tmpl = env.from_string('  {% if kvs %}   {% for k, v in kvs %}{{ k }}={{ v }} {% endfor %}  {% endif %}')
        out = tmpl.render(kvs=[('a', 1), ('b', 2)])
        assert out == '   a=1 b=2   '

    def test_lstrip_blocks_consume_after_without_new_line(self):
        if False:
            for i in range(10):
                print('nop')
        env = Environment(lstrip_blocks=True, trim_blocks=False)
        tmpl = env.from_string('  {% if kvs -%}   {% for k, v in kvs %}{{ k }}={{ v }} {% endfor -%}  {% endif -%}')
        out = tmpl.render(kvs=[('a', 1), ('b', 2)])
        assert out == 'a=1 b=2 '

    def test_lstrip_trim_blocks_consume_before_without_new_line(self):
        if False:
            for i in range(10):
                print('nop')
        env = Environment(lstrip_blocks=False, trim_blocks=False)
        tmpl = env.from_string('  {%- if kvs %}   {%- for k, v in kvs %}{{ k }}={{ v }} {% endfor -%}  {%- endif %}')
        out = tmpl.render(kvs=[('a', 1), ('b', 2)])
        assert out == 'a=1 b=2 '

    def test_lstrip_trim_blocks_comment(self):
        if False:
            while True:
                i = 10
        env = Environment(lstrip_blocks=True, trim_blocks=True)
        tmpl = env.from_string(' {# 1 space #}\n  {# 2 spaces #}    {# 4 spaces #}')
        out = tmpl.render()
        assert out == ' ' * 4

    def test_lstrip_trim_blocks_raw(self):
        if False:
            print('Hello World!')
        env = Environment(lstrip_blocks=True, trim_blocks=True)
        tmpl = env.from_string('{{x}}\n{%- raw %} {% endraw -%}\n{{ y }}')
        out = tmpl.render(x=1, y=2)
        assert out == '1 2'

    def test_php_syntax_with_manual(self, env):
        if False:
            for i in range(10):
                print('nop')
        env = Environment('<?', '?>', '<?=', '?>', '<!--', '-->', lstrip_blocks=True, trim_blocks=True)
        tmpl = env.from_string("    <!-- I'm a comment, I'm not interesting -->\n    <? for item in seq -?>\n        <?= item ?>\n    <?- endfor ?>")
        assert tmpl.render(seq=range(5)) == '01234'

    def test_php_syntax(self, env):
        if False:
            print('Hello World!')
        env = Environment('<?', '?>', '<?=', '?>', '<!--', '-->', lstrip_blocks=True, trim_blocks=True)
        tmpl = env.from_string("    <!-- I'm a comment, I'm not interesting -->\n    <? for item in seq ?>\n        <?= item ?>\n    <? endfor ?>")
        assert tmpl.render(seq=range(5)) == ''.join((f'        {x}\n' for x in range(5)))

    def test_php_syntax_compact(self, env):
        if False:
            print('Hello World!')
        env = Environment('<?', '?>', '<?=', '?>', '<!--', '-->', lstrip_blocks=True, trim_blocks=True)
        tmpl = env.from_string("    <!-- I'm a comment, I'm not interesting -->\n    <?for item in seq?>\n        <?=item?>\n    <?endfor?>")
        assert tmpl.render(seq=range(5)) == ''.join((f'        {x}\n' for x in range(5)))

    def test_erb_syntax(self, env):
        if False:
            for i in range(10):
                print('nop')
        env = Environment('<%', '%>', '<%=', '%>', '<%#', '%>', lstrip_blocks=True, trim_blocks=True)
        tmpl = env.from_string("<%# I'm a comment, I'm not interesting %>\n    <% for item in seq %>\n    <%= item %>\n    <% endfor %>\n")
        assert tmpl.render(seq=range(5)) == ''.join((f'    {x}\n' for x in range(5)))

    def test_erb_syntax_with_manual(self, env):
        if False:
            for i in range(10):
                print('nop')
        env = Environment('<%', '%>', '<%=', '%>', '<%#', '%>', lstrip_blocks=True, trim_blocks=True)
        tmpl = env.from_string("<%# I'm a comment, I'm not interesting %>\n    <% for item in seq -%>\n        <%= item %>\n    <%- endfor %>")
        assert tmpl.render(seq=range(5)) == '01234'

    def test_erb_syntax_no_lstrip(self, env):
        if False:
            while True:
                i = 10
        env = Environment('<%', '%>', '<%=', '%>', '<%#', '%>', lstrip_blocks=True, trim_blocks=True)
        tmpl = env.from_string("<%# I'm a comment, I'm not interesting %>\n    <%+ for item in seq -%>\n        <%= item %>\n    <%- endfor %>")
        assert tmpl.render(seq=range(5)) == '    01234'

    def test_comment_syntax(self, env):
        if False:
            i = 10
            return i + 15
        env = Environment('<!--', '-->', '${', '}', '<!--#', '-->', lstrip_blocks=True, trim_blocks=True)
        tmpl = env.from_string("<!--# I'm a comment, I'm not interesting --><!-- for item in seq --->\n    ${item}\n<!--- endfor -->")
        assert tmpl.render(seq=range(5)) == '01234'

class TestTrimBlocks:

    def test_trim(self, env):
        if False:
            while True:
                i = 10
        env = Environment(trim_blocks=True, lstrip_blocks=False)
        tmpl = env.from_string('    {% if True %}\n    {% endif %}')
        assert tmpl.render() == '        '

    def test_no_trim(self, env):
        if False:
            for i in range(10):
                print('nop')
        env = Environment(trim_blocks=True, lstrip_blocks=False)
        tmpl = env.from_string('    {% if True +%}\n    {% endif %}')
        assert tmpl.render() == '    \n    '

    def test_no_trim_outer(self, env):
        if False:
            while True:
                i = 10
        env = Environment(trim_blocks=True, lstrip_blocks=False)
        tmpl = env.from_string('{% if True %}X{% endif +%}\nmore things')
        assert tmpl.render() == 'X\nmore things'

    def test_lstrip_no_trim(self, env):
        if False:
            print('Hello World!')
        env = Environment(trim_blocks=True, lstrip_blocks=True)
        tmpl = env.from_string('    {% if True +%}\n    {% endif %}')
        assert tmpl.render() == '\n'

    def test_trim_blocks_false_with_no_trim(self, env):
        if False:
            i = 10
            return i + 15
        env = Environment(trim_blocks=False, lstrip_blocks=False)
        tmpl = env.from_string('    {% if True %}\n    {% endif %}')
        assert tmpl.render() == '    \n    '
        tmpl = env.from_string('    {% if True +%}\n    {% endif %}')
        assert tmpl.render() == '    \n    '
        tmpl = env.from_string('    {# comment #}\n    ')
        assert tmpl.render() == '    \n    '
        tmpl = env.from_string('    {# comment +#}\n    ')
        assert tmpl.render() == '    \n    '
        tmpl = env.from_string('    {% raw %}{% endraw %}\n    ')
        assert tmpl.render() == '    \n    '
        tmpl = env.from_string('    {% raw %}{% endraw +%}\n    ')
        assert tmpl.render() == '    \n    '

    def test_trim_nested(self, env):
        if False:
            i = 10
            return i + 15
        env = Environment(trim_blocks=True, lstrip_blocks=True)
        tmpl = env.from_string('    {% if True %}\na {% if True %}\nb {% endif %}\nc {% endif %}')
        assert tmpl.render() == 'a b c '

    def test_no_trim_nested(self, env):
        if False:
            print('Hello World!')
        env = Environment(trim_blocks=True, lstrip_blocks=True)
        tmpl = env.from_string('    {% if True +%}\na {% if True +%}\nb {% endif +%}\nc {% endif %}')
        assert tmpl.render() == '\na \nb \nc '

    def test_comment_trim(self, env):
        if False:
            print('Hello World!')
        env = Environment(trim_blocks=True, lstrip_blocks=True)
        tmpl = env.from_string('    {# comment #}\n\n  ')
        assert tmpl.render() == '\n  '

    def test_comment_no_trim(self, env):
        if False:
            print('Hello World!')
        env = Environment(trim_blocks=True, lstrip_blocks=True)
        tmpl = env.from_string('    {# comment +#}\n\n  ')
        assert tmpl.render() == '\n\n  '

    def test_multiple_comment_trim_lstrip(self, env):
        if False:
            for i in range(10):
                print('nop')
        env = Environment(trim_blocks=True, lstrip_blocks=True)
        tmpl = env.from_string('   {# comment #}\n\n{# comment2 #}\n   \n{# comment3 #}\n\n ')
        assert tmpl.render() == '\n   \n\n '

    def test_multiple_comment_no_trim_lstrip(self, env):
        if False:
            i = 10
            return i + 15
        env = Environment(trim_blocks=True, lstrip_blocks=True)
        tmpl = env.from_string('   {# comment +#}\n\n{# comment2 +#}\n   \n{# comment3 +#}\n\n ')
        assert tmpl.render() == '\n\n\n   \n\n\n '

    def test_raw_trim_lstrip(self, env):
        if False:
            print('Hello World!')
        env = Environment(trim_blocks=True, lstrip_blocks=True)
        tmpl = env.from_string('{{x}}{% raw %}\n\n    {% endraw %}\n\n{{ y }}')
        assert tmpl.render(x=1, y=2) == '1\n\n\n2'

    def test_raw_no_trim_lstrip(self, env):
        if False:
            print('Hello World!')
        env = Environment(trim_blocks=False, lstrip_blocks=True)
        tmpl = env.from_string('{{x}}{% raw %}\n\n      {% endraw +%}\n\n{{ y }}')
        assert tmpl.render(x=1, y=2) == '1\n\n\n\n2'
        with pytest.raises(TemplateSyntaxError):
            tmpl = env.from_string('{{x}}{% raw +%}\n\n  {% endraw +%}\n\n{{ y }}')

    def test_no_trim_angle_bracket(self, env):
        if False:
            i = 10
            return i + 15
        env = Environment('<%', '%>', '${', '}', '<%#', '%>', lstrip_blocks=True, trim_blocks=True)
        tmpl = env.from_string('    <% if True +%>\n\n    <% endif %>')
        assert tmpl.render() == '\n\n'
        tmpl = env.from_string('    <%# comment +%>\n\n   ')
        assert tmpl.render() == '\n\n   '

    def test_no_trim_php_syntax(self, env):
        if False:
            for i in range(10):
                print('nop')
        env = Environment('<?', '?>', '<?=', '?>', '<!--', '-->', lstrip_blocks=False, trim_blocks=True)
        tmpl = env.from_string('    <? if True +?>\n\n    <? endif ?>')
        assert tmpl.render() == '    \n\n    '
        tmpl = env.from_string('    <!-- comment +-->\n\n    ')
        assert tmpl.render() == '    \n\n    '