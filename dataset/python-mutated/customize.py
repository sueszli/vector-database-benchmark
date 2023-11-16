"""Isolate Python version-specific semantic actions here.
"""
from uncompyle6.parsers.treenode import SyntaxTree
from uncompyle6.semantics.consts import INDENT_PER_LEVEL, NO_PARENTHESIS_EVER, PRECEDENCE, TABLE_R, TABLE_DIRECT
from uncompyle6.semantics.helper import flatten_list
from uncompyle6.scanners.tok import Token

def customize_for_version(self, is_pypy, version):
    if False:
        while True:
            i = 10
    if is_pypy:
        TABLE_DIRECT.update({'assert': ('%|assert %c\n', 0), 'assert2': ('%|assert %c, %c\n', 0, 3), 'assert_pypy': ('%|assert %c\n', (1, 'assert_expr')), 'assert0_pypy': ('%|assert %c\n', 0), 'assert_not_pypy': ('%|assert not %c\n', (1, 'assert_exp')), 'assert2_not_pypy': ('%|assert not %c, %c\n', (1, 'assert_exp'), (4, 'expr')), 'try_except_pypy': ('%|try:\n%+%c%-%c\n\n', 1, 2), 'tryfinallystmt_pypy': ('%|try:\n%+%c%-%|finally:\n%+%c%-\n\n', 1, 3), 'assign3_pypy': ('%|%c, %c, %c = %c, %c, %c\n', 5, 4, 3, 0, 1, 2), 'assign2_pypy': ('%|%c, %c = %c, %c\n', 3, 2, 0, 1)})
        if version[:2] >= (3, 7):

            def n_call_kw_pypy37(node):
                if False:
                    while True:
                        i = 10
                self.template_engine(('%p(', (0, NO_PARENTHESIS_EVER)), node)
                assert node[-1] == 'CALL_METHOD_KW'
                arg_count = node[-1].attr
                kw_names = node[-2]
                assert kw_names == 'pypy_kw_keys'
                kwargs_names = kw_names[0].attr
                kwarg_count = len(kwargs_names)
                pos_argc = arg_count - kwarg_count
                flat_elems = flatten_list(node[1:-2])
                n = len(flat_elems)
                assert n == arg_count, 'n: %s, arg_count: %s\n%s' % (n, arg_count, node)
                sep = ''
                for i in range(pos_argc):
                    elem = flat_elems[i]
                    line_number = self.line_number
                    value = self.traverse(elem)
                    if line_number != self.line_number:
                        sep += '\n' + self.indent + INDENT_PER_LEVEL[:-1]
                        pass
                    self.write(f'{sep}{value}')
                    sep = ', '
                assert n >= len(kwargs_names)
                j = pos_argc
                for i in range(kwarg_count):
                    elem = flat_elems[j]
                    j += 1
                    assert elem == 'expr'
                    line_number = self.line_number
                    value = self.traverse(elem)
                    if line_number != self.line_number:
                        sep += '\n' + self.indent + INDENT_PER_LEVEL[:-1]
                        pass
                    self.write(f'{sep}{kwargs_names[i]}={value}')
                    sep = ', '
                    pass
                self.write(')')
                self.prune()
            self.n_call_kw_pypy37 = n_call_kw_pypy37
    else:
        TABLE_DIRECT.update({'assert': ('%|assert %c\n', 0), 'assert2': ('%|assert %c, %c\n', 0, 3), 'assertnot': ('%|assert not %p\n', (0, PRECEDENCE['unary_not'])), 'assert2not': ('%|assert not %p, %c\n', (0, PRECEDENCE['unary_not']), 3), 'assign2': ('%|%c, %c = %c, %c\n', 3, 4, 0, 1), 'assign3': ('%|%c, %c, %c = %c, %c, %c\n', 5, 6, 7, 0, 1, 2), 'try_except': ('%|try:\n%+%c%-%c\n\n', 1, 3)})
    if version >= (3, 0):
        if version >= (3, 2):
            TABLE_DIRECT.update({'del_deref_stmt': ('%|del %c\n', 0), 'DELETE_DEREF': ('%{pattr}', 0)})
        from uncompyle6.semantics.customize3 import customize_for_version3
        customize_for_version3(self, version)
    else:
        TABLE_DIRECT.update({'except_cond3': ('%|except %c, %c:\n', (1, 'expr'), (-2, 'store'))})
        if version <= (2, 6):
            TABLE_DIRECT['testtrue_then'] = TABLE_DIRECT['testtrue']
        if (2, 4) <= version <= (2, 6):
            TABLE_DIRECT.update({'comp_for': (' for %c in %c', 3, 1)})
        else:
            TABLE_DIRECT.update({'comp_for': (' for %c in %c%c', 2, 0, 3)})
        if version >= (2, 5):
            from uncompyle6.semantics.customize25 import customize_for_version25
            customize_for_version25(self, version)
            if version >= (2, 6):
                from uncompyle6.semantics.customize26_27 import customize_for_version26_27
                customize_for_version26_27(self, version)
                pass
        else:
            global NAME_MODULE
            NAME_MODULE = SyntaxTree('stmt', [SyntaxTree('assign', [SyntaxTree('expr', [Token('LOAD_GLOBAL', pattr='__name__', offset=0, has_arg=True)]), SyntaxTree('store', [Token('STORE_NAME', pattr='__module__', offset=3, has_arg=True)])])])
            TABLE_DIRECT.update({'importmultiple': ('%|import %c%c\n', 2, 3), 'import_cont': (', %c', 2), 'tryfinallystmt': ('%|try:\n%+%c%-%|finally:\n%+%c%-', (1, 'suite_stmts_opt'), (5, 'suite_stmts_opt'))})
            if version == (2, 4):

                def n_iftrue_stmt24(node):
                    if False:
                        return 10
                    self.template_engine(('%c', 0), node)
                    self.default(node)
                    self.prune()
                self.n_iftrue_stmt24 = n_iftrue_stmt24
            elif version < (1, 4):
                from uncompyle6.semantics.customize14 import customize_for_version14
                customize_for_version14(self, version)

                def n_call(node):
                    if False:
                        i = 10
                        return i + 15
                    expr = node[0]
                    assert expr == 'expr'
                    params = node[1]
                    if params == 'tuple':
                        self.template_engine(('%p(', (0, NO_PARENTHESIS_EVER)), expr)
                        sep = ''
                        for param in params[:-1]:
                            self.write(sep)
                            self.preorder(param)
                            sep = ', '
                        self.write(')')
                    else:
                        self.template_engine(('%p(%P)', (0, 'expr', 100), (1, -1, ', ', NO_PARENTHESIS_EVER)), node)
                    self.prune()
                self.n_call = n_call
            else:
                TABLE_DIRECT.update({'if1_stmt': ('%|if 1\n%+%c%-', 5)})
                if version <= (2, 1):
                    TABLE_DIRECT.update({'importmultiple': ('%c', 2), 'imports_cont': ('%C%,', (1, 100, '\n'))})
                    pass
                pass
            pass
        TABLE_R.update({'STORE_SLICE+0': ('%c[:]', 0), 'STORE_SLICE+1': ('%c[%p:]', 0, (1, -1)), 'STORE_SLICE+2': ('%c[:%p]', 0, (1, -1)), 'STORE_SLICE+3': ('%c[%p:%p]', 0, (1, -1), (2, -1)), 'DELETE_SLICE+0': ('%|del %c[:]\n', 0), 'DELETE_SLICE+1': ('%|del %c[%c:]\n', 0, 1), 'DELETE_SLICE+2': ('%|del %c[:%c]\n', 0, 1), 'DELETE_SLICE+3': ('%|del %c[%c:%c]\n', 0, 1, 2)})
        TABLE_DIRECT.update({'raise_stmt2': ('%|raise %c, %c\n', 0, 1)})

        def n_exec_stmt(node):
            if False:
                for i in range(10):
                    print('nop')
            '\n            exec_stmt ::= expr exprlist DUP_TOP EXEC_STMT\n            exec_stmt ::= expr exprlist EXEC_STMT\n            '
            self.write(self.indent, 'exec ')
            self.preorder(node[0])
            if not node[1][0].isNone():
                sep = ' in '
                for subnode in node[1]:
                    self.write(sep)
                    sep = ', '
                    self.preorder(subnode)
            self.println()
            self.prune()
        self.n_exec_smt = n_exec_stmt
        pass
    return