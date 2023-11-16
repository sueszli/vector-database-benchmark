"""Isolate Python 3.6 version-specific semantic actions here.
"""
from xdis import iscode
from spark_parser.ast import GenericASTTraversalPruningException
from uncompyle6.scanners.tok import Token
from uncompyle6.semantics.helper import flatten_list, escape_string, strip_quotes
from uncompyle6.semantics.consts import INDENT_PER_LEVEL, PRECEDENCE, TABLE_DIRECT, TABLE_R
from uncompyle6.util import get_code_name

def escape_format(s):
    if False:
        return 10
    return s.replace('\r', '\\r').replace('\n', '\\n').replace("'''", '"""')

def customize_for_version36(self, version):
    if False:
        print('Hello World!')
    PRECEDENCE['call_kw'] = 0
    PRECEDENCE['call_kw36'] = 1
    PRECEDENCE['call_ex'] = 1
    PRECEDENCE['call_ex_kw'] = 1
    PRECEDENCE['call_ex_kw2'] = 1
    PRECEDENCE['call_ex_kw3'] = 1
    PRECEDENCE['call_ex_kw4'] = 1
    PRECEDENCE['dict_pack'] = 0
    PRECEDENCE['formatted_value1'] = 100
    TABLE_DIRECT.update({'ann_assign_init_value': ('%|%c = %p\n', (-1, 'store_annotation'), (0, 'expr', 200)), 'ann_assign_no_init': ('%|%c\n', (0, 'store_annotation')), 'async_for_stmt36': ('%|async for %c in %c:\n%+%c%-\n\n', (9, 'store'), (1, 'expr'), (-9, ('for_block', 'pass'))), 'async_forelse_stmt36': ('%|async for %c in %c:\n%+%c%-%|else:\n%+%c%-\n\n', (9, 'store'), (1, 'expr'), (-10, 'for_block'), (-2, 'else_suite')), 'call_ex': ('%c(%p)', (0, 'expr'), (1, 100)), 'except_return': ('%|except:\n%+%c%-', 3), 'func_args36': ('%c(**', 0), 'if_exp37': ('%p if %c else %c', (1, 'expr', 27), 0, 3), 'ifstmtl': ('%|if %c:\n%+%c%-', (0, 'testexpr'), (1, '_ifstmts_jumpl')), 'list_afor': (' async for %[1]{%c} in %c%[1]{%c}', (1, 'store'), (0, 'get_aiter'), (3, 'list_iter')), 'try_except36': ('%|try:\n%+%c%-%c\n\n', 1, -2), 'tryfinally36': ('%|try:\n%+%c%-%|finally:\n%+%c%-\n\n', (1, 'returns'), 3), 'tryfinally_return_stmt': ('%|try:\n%+%c%-%|finally:\n%+%|return%-\n\n', 1), 'unpack_list': ('*%c', (0, 'list')), 'store_annotation': ('%[1]{pattr}: %c', 0)})
    TABLE_R.update({'CALL_FUNCTION_EX': ('%c(*%P)', 0, (1, 2, ', ', 100)), 'CALL_FUNCTION_EX_KW': ('%c(**%C)', 0, (2, 3, ','))})

    def build_unpack_tuple_with_call(node):
        if False:
            i = 10
            return i + 15
        n = node[0]
        if n == 'expr':
            n = n[0]
        if n == 'tuple':
            self.call36_tuple(n)
            first = 1
            sep = ', *'
        elif n == 'LOAD_STR':
            value = self.format_pos_args(n)
            self.f.write(value)
            first = 1
            sep = ', *'
        else:
            first = 0
            sep = '*'
        buwc = node[-1]
        assert buwc.kind.startswith('BUILD_TUPLE_UNPACK_WITH_CALL')
        for n in node[first:-1]:
            self.f.write(sep)
            self.preorder(n)
            sep = ', *'
            pass
        self.prune()
        return
    self.n_build_tuple_unpack_with_call = build_unpack_tuple_with_call

    def build_unpack_map_with_call(node):
        if False:
            return 10
        n = node[0]
        if n == 'expr':
            n = n[0]
        if n == 'dict':
            self.call36_dict(n)
            first = 1
            sep = ', **'
        else:
            first = 0
            sep = '**'
        for n in node[first:-1]:
            self.f.write(sep)
            self.preorder(n)
            sep = ', **'
            pass
        self.prune()
        return
    self.n_build_map_unpack_with_call = build_unpack_map_with_call

    def n_classdef36(node):
        if False:
            i = 10
            return i + 15
        cclass = self.currentclass
        subclass_info = None
        if node == 'classdefdeco2':
            if isinstance(node[1][1].attr, str):
                class_name = node[1][1].attr
                if self.is_pypy and class_name.find('<locals>') > 0:
                    class_name = class_name.split('.')[-1]
            else:
                class_name = node[1][2].attr
            build_class = node
        else:
            build_class = node[0]
            if build_class == 'build_class_kw':
                mkfunc = build_class[1]
                assert mkfunc == 'mkfunc'
                subclass_info = build_class
                if hasattr(mkfunc[0], 'attr') and iscode(mkfunc[0].attr):
                    subclass_code = mkfunc[0].attr
                else:
                    assert mkfunc[0] == 'load_closure'
                    subclass_code = mkfunc[1].attr
                    assert iscode(subclass_code)
            if build_class[1][0] == 'load_closure':
                code_node = build_class[1][1]
            else:
                code_node = build_class[1][0]
            class_name = get_code_name(code_node.attr)
        assert 'mkfunc' == build_class[1]
        mkfunc = build_class[1]
        if mkfunc[0] in ('kwargs', 'no_kwargs'):
            for n in mkfunc:
                if hasattr(n, 'attr') and iscode(n.attr):
                    subclass_code = n.attr
                    break
                pass
            if node == 'classdefdeco2':
                subclass_info = node
            else:
                subclass_info = node[0]
        elif build_class[1][0] == 'load_closure':
            load_closure = build_class[1]
            subclass_code = None
            for i in range(-4, -1):
                if load_closure[i] == 'LOAD_CODE':
                    subclass_code = load_closure[i].attr
                    break
            if subclass_code is None:
                raise RuntimeError('Internal Error n_classdef: cannot find class body')
            if hasattr(build_class[3], '__len__'):
                if not subclass_info:
                    subclass_info = build_class[3]
            elif hasattr(build_class[2], '__len__'):
                subclass_info = build_class[2]
            else:
                raise RuntimeError('Internal Error n_classdef: cannot superclass name')
        elif node == 'classdefdeco2':
            subclass_info = node
            subclass_code = build_class[1][0].attr
        elif not subclass_info:
            if mkfunc[0] in ('no_kwargs', 'kwargs'):
                subclass_code = mkfunc[1].attr
            else:
                subclass_code = mkfunc[0].attr
            if node == 'classdefdeco2':
                subclass_info = node
            else:
                subclass_info = node[0]
        if node == 'classdefdeco2':
            self.write('\n')
        else:
            self.write('\n\n')
        self.currentclass = str(class_name)
        self.write(self.indent, 'class ', self.currentclass)
        self.print_super_classes3(subclass_info)
        self.println(':')
        self.indent_more()
        self.build_class(subclass_code)
        self.indent_less()
        self.currentclass = cclass
        if len(self.param_stack) > 1:
            self.write('\n\n')
        else:
            self.write('\n\n\n')
        self.prune()
    self.n_classdef36 = n_classdef36

    def call_ex_kw(node):
        if False:
            i = 10
            return i + 15
        'Handle CALL_FUNCTION_EX 1 (have KW) but with\n        BUILD_MAP_UNPACK_WITH_CALL'
        expr = node[1]
        assert expr == 'expr'
        value = self.format_pos_args(expr)
        if value == '':
            fmt = '%c(%p)'
        else:
            fmt = '%%c(%s, %%p)' % value
        self.template_engine((fmt, (0, 'expr'), (2, 'build_map_unpack_with_call', 100)), node)
        self.prune()
    self.n_call_ex_kw = call_ex_kw

    def call_ex_kw2(node):
        if False:
            while True:
                i = 10
        'Handle CALL_FUNCTION_EX 2  (have KW) but with\n        BUILD_{MAP,TUPLE}_UNPACK_WITH_CALL'
        assert node[1] == 'build_tuple_unpack_with_call'
        value = self.format_pos_args(node[1])
        if value == '':
            fmt = '%c(%p)'
        else:
            fmt = '%%c(%s, %%p)' % value
        self.template_engine((fmt, (0, 'expr'), (2, 'build_map_unpack_with_call', 100)), node)
        self.prune()
    self.n_call_ex_kw2 = call_ex_kw2

    def call_ex_kw3(node):
        if False:
            print('Hello World!')
        'Handle CALL_FUNCTION_EX 1 (have KW) but without\n        BUILD_MAP_UNPACK_WITH_CALL'
        self.preorder(node[0])
        self.write('(')
        value = self.format_pos_args(node[1][0])
        if value == '':
            pass
        else:
            self.write(value)
            self.write(', ')
        self.write('*')
        self.preorder(node[1][1])
        self.write(', ')
        kwargs = node[2]
        if kwargs == 'expr' and kwargs[0] != 'dict':
            kwargs = kwargs[0]
        if kwargs == 'dict':
            self.call36_dict(kwargs)
        else:
            self.write('**')
            self.preorder(kwargs)
        self.write(')')
        self.prune()
    self.n_call_ex_kw3 = call_ex_kw3

    def call_ex_kw4(node):
        if False:
            while True:
                i = 10
        'Handle CALL_FUNCTION_EX {1 or 2} but without\n        BUILD_{MAP,TUPLE}_UNPACK_WITH_CALL'
        self.preorder(node[0])
        self.write('(')
        args = node[1][0]
        if args == 'tuple':
            if self.call36_tuple(args) > 0:
                self.write(', ')
                pass
            pass
        else:
            self.write('*')
            self.preorder(args)
            self.write(', ')
            pass
        kwargs = node[2]
        if kwargs == 'expr':
            kwargs = kwargs[0]
        call_function_ex = node[-1]
        assert call_function_ex == 'CALL_FUNCTION_EX_KW' or (self.version >= (3, 6) and call_function_ex == 'CALL_FUNCTION_EX')
        if call_function_ex.attr & 1 and (not isinstance(kwargs, Token) and kwargs != 'attribute') and (kwargs != 'call_kw36') and (not kwargs[0].kind.startswith('kvlist')):
            self.call36_dict(kwargs)
        else:
            self.write('**')
            self.preorder(kwargs)
        self.write(')')
        self.prune()
    self.n_call_ex_kw4 = call_ex_kw4

    def format_pos_args(node):
        if False:
            for i in range(10):
                print('nop')
        "\n        Positional args should format to:\n        (*(2, ), ...) -> (2, ...)\n        We remove starting and trailing parenthesis and ', ' if\n        tuple has only one element.\n        "
        value = self.traverse(node, indent='')
        if value.startswith('('):
            assert value.endswith(')')
            value = value[1:-1].rstrip(' ')
            if value == '':
                pass
            elif value.endswith(','):
                value = value[:-1]
        return value
    self.format_pos_args = format_pos_args

    def call36_tuple(node):
        if False:
            i = 10
            return i + 15
        "\n        A tuple used in a call; these are like normal tuples, but they\n        don't have the enclosing parenthesis.\n        "
        assert node == 'tuple'
        flat_elems = flatten_list(node[:-1])
        self.indent_more(INDENT_PER_LEVEL)
        sep = ''
        for elem in flat_elems:
            if elem in ('ROT_THREE', 'EXTENDED_ARG'):
                continue
            assert elem == 'expr'
            line_number = self.line_number
            value = self.traverse(elem)
            if line_number != self.line_number:
                sep += '\n' + self.indent + INDENT_PER_LEVEL[:-1]
            self.write(sep, value)
            sep = ', '
        self.indent_less(INDENT_PER_LEVEL)
        return len(flat_elems)
    self.call36_tuple = call36_tuple

    def call36_dict(node):
        if False:
            while True:
                i = 10
        '\n        A dict used in a call_ex_kw2, which are a dictionary items expressed\n        in a call. This should format to:\n             a=1, b=2\n        In other words, no braces, no quotes around keys and ":" becomes\n        "=".\n\n        We will source-code use line breaks to guide us when to break.\n        '
        p = self.prec
        self.prec = 100
        self.indent_more(INDENT_PER_LEVEL)
        sep = INDENT_PER_LEVEL[:-1]
        line_number = self.line_number
        if node[0].kind.startswith('kvlist'):
            kv_node = node[0]
            l = list(kv_node)
            i = 0
            length = len(l)
            if kv_node[-1].kind.startswith('BUILD_MAP'):
                length -= 1
            while i < length:
                self.write(sep)
                name = self.traverse(l[i], indent='')
                name = name[1:-1]
                if i > 0:
                    line_number = self.indent_if_source_nl(line_number, self.indent + INDENT_PER_LEVEL[:-1])
                line_number = self.line_number
                self.write(name, '=')
                value = self.traverse(l[i + 1], indent=self.indent + (len(name) + 2) * ' ')
                self.write(value)
                sep = ', '
                if line_number != self.line_number:
                    sep += '\n' + self.indent + INDENT_PER_LEVEL[:-1]
                    line_number = self.line_number
                i += 2
                pass
        elif node[-1].kind.startswith('BUILD_CONST_KEY_MAP'):
            keys_node = node[-2]
            keys = keys_node.attr
            assert keys_node == 'LOAD_CONST' and isinstance(keys, tuple)
            for i in range(node[-1].attr):
                self.write(sep)
                self.write(keys[i], '=')
                value = self.traverse(node[i], indent='')
                self.write(value)
                sep = ', '
                if line_number != self.line_number:
                    sep += '\n' + self.indent + INDENT_PER_LEVEL[:-1]
                    line_number = self.line_number
                    pass
                pass
        else:
            self.write('**')
            try:
                self.default(node)
            except GenericASTTraversalPruningException:
                pass
        self.prec = p
        self.indent_less(INDENT_PER_LEVEL)
        return
    self.call36_dict = call36_dict

    def n_call_kw36(node):
        if False:
            while True:
                i = 10
        self.template_engine(('%p(', (0, 100)), node)
        keys = node[-2].attr
        num_kwargs = len(keys)
        num_posargs = len(node) - (num_kwargs + 2)
        n = len(node)
        assert n >= len(keys) + 1, 'not enough parameters keyword-tuple values'
        sep = ''
        line_number = self.line_number
        for i in range(1, num_posargs):
            self.write(sep)
            self.preorder(node[i])
            if line_number != self.line_number:
                sep = ',\n' + self.indent + '  '
            else:
                sep = ', '
            line_number = self.line_number
        i = num_posargs
        j = 0
        while i < n - 2:
            self.write(sep)
            self.write(keys[j] + '=')
            self.preorder(node[i])
            if line_number != self.line_number:
                sep = ',\n' + self.indent + '  '
            else:
                sep = ', '
            i += 1
            j += 1
        self.write(')')
        self.prune()
        return
    self.n_call_kw36 = n_call_kw36
    FSTRING_CONVERSION_MAP = {1: '!s', 2: '!r', 3: '!a', 'X': ':X'}

    def n_except_suite_finalize(node):
        if False:
            print('Hello World!')
        if node[1] == 'returns' and self.hide_internal:
            self.indent_more()
            self.preorder(node[1])
            self.indent_less()
        else:
            self.default(node)
        self.prune()
    self.n_except_suite_finalize = n_except_suite_finalize

    def n_formatted_value(node):
        if False:
            return 10
        if node[0] in ('LOAD_STR', 'LOAD_CONST'):
            value = node[0].attr
            if isinstance(value, tuple):
                self.write(node[0].attr)
            else:
                self.write(escape_string(node[0].attr))
            self.prune()
        else:
            self.default(node)
    self.n_formatted_value = n_formatted_value

    def n_formatted_value_attr(node):
        if False:
            print('Hello World!')
        f_conversion(node)
        fmt_node = node.data[3]
        if fmt_node == 'expr' and fmt_node[0] == 'LOAD_STR':
            node.string = escape_format(fmt_node[0].attr)
        else:
            node.string = fmt_node
        self.default(node)
    self.n_formatted_value_attr = n_formatted_value_attr

    def f_conversion(node):
        if False:
            i = 10
            return i + 15
        fmt_node = node.data[1]
        if fmt_node == 'expr' and fmt_node[0] == 'LOAD_STR':
            data = fmt_node[0].attr
        else:
            data = fmt_node.attr
        node.conversion = FSTRING_CONVERSION_MAP.get(data, '')
        return node.conversion

    def n_formatted_value1(node):
        if False:
            i = 10
            return i + 15
        expr = node[0]
        assert expr == 'expr'
        conversion = f_conversion(node)
        if self.in_format_string and self.in_format_string != 'formatted_value1':
            value = self.traverse(expr, indent='')
            if value[0] == '{':
                fmt = '{ %s%s }'
            else:
                fmt = '{%s%s}'
            es = escape_string(fmt % (value, conversion))
            f_str = '%s' % es
        else:
            old_in_format_string = self.in_format_string
            self.in_format_string = 'formatted_value1'
            value = self.traverse(expr, indent='')
            self.in_format_string = old_in_format_string
            es = escape_string('{%s%s}' % (value, conversion))
            f_str = 'f%s' % es
        self.write(f_str)
        self.prune()
    self.n_formatted_value1 = n_formatted_value1

    def n_formatted_value2(node):
        if False:
            return 10
        p = self.prec
        self.prec = 100
        expr = node[0]
        assert expr == 'expr'
        old_in_format_string = self.in_format_string
        self.in_format_string = 'formatted_value2'
        value = self.traverse(expr, indent='')
        format_value_attr = node[-1]
        assert format_value_attr == 'FORMAT_VALUE_ATTR'
        attr = format_value_attr.attr
        if attr & 4:
            assert node[1] == 'expr'
            fmt = strip_quotes(self.traverse(node[1], indent=''))
            attr_flags = attr & 3
            if attr_flags:
                conversion = '%s:%s' % (FSTRING_CONVERSION_MAP.get(attr_flags, ''), fmt)
            else:
                conversion = ':%s' % fmt
        else:
            conversion = FSTRING_CONVERSION_MAP.get(attr, '')
        self.in_format_string = old_in_format_string
        f_str = 'f%s' % escape_string('{%s%s}' % (value, conversion))
        self.write(f_str)
        self.prec = p
        self.prune()
    self.n_formatted_value2 = n_formatted_value2

    def n_joined_str(node):
        if False:
            while True:
                i = 10
        p = self.prec
        self.prec = 100
        old_in_format_string = self.in_format_string
        self.in_format_string = 'joined_str'
        result = ''
        for expr in node[:-1]:
            assert expr == 'expr'
            value = self.traverse(expr, indent='')
            if expr[0].kind.startswith('formatted_value'):
                if value.startswith('f'):
                    value = value[1:]
                pass
            else:
                assert expr[0] == 'LOAD_STR' or (expr[0] == 'LOAD_CONST' and isinstance(expr[0].attr, unicode))
                value = value.replace('{', '{{').replace('}', '}}')
            result += strip_quotes(value)
            pass
        self.in_format_string = old_in_format_string
        if self.in_format_string:
            self.write(result)
        else:
            self.write('f%s' % escape_string(result))
        self.prec = p
        self.prune()
    self.n_joined_str = n_joined_str

    def n_list_comp_async(node):
        if False:
            for i in range(10):
                print('nop')
        self.write('[')
        if node[0].kind == 'load_closure':
            self.listcomp_closure3(node)
        else:
            self.comprehension_walk_newer(node, iter_index=3, code_index=0)
        self.write(']')
        self.prune()
    self.n_list_comp_async = n_list_comp_async

    def starred(node):
        if False:
            while True:
                i = 10
        l = len(node)
        assert l > 0
        pos_args = node[0]
        if pos_args == 'expr':
            pos_args = pos_args[0]
        if pos_args == 'tuple':
            build_tuple = pos_args[0]
            if build_tuple.kind.startswith('BUILD_TUPLE'):
                tuple_len = 0
            else:
                tuple_len = len(node) - 1
            star_start = 1
            template = ('%C', (0, -1, ', '))
            self.template_engine(template, pos_args)
            if tuple_len == 0:
                self.write('*()')
                self.prune()
            self.write(', ')
        else:
            star_start = 0
        if l > 1:
            template = ('*%C', (star_start, -1, ', *'))
        else:
            template = ('*%c', (star_start, 'expr'))
        self.template_engine(template, node)
        self.prune()
    self.n_starred = starred

    def return_closure(node):
        if False:
            return 10
        self.prune()
        return
    self.n_return_closure = return_closure