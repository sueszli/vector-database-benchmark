"""
All the crazy things we have to do to handle Python functions in 3.6 and above.
The saga of changes before 3.6 is in other files.
"""
from xdis import iscode, code_has_star_arg, code_has_star_star_arg, CO_GENERATOR, CO_ASYNC_GENERATOR
from uncompyle6.scanner import Code
from uncompyle6.semantics.parser_error import ParserError
from uncompyle6.parser import ParserError as ParserError2
from uncompyle6.semantics.helper import find_all_globals, find_globals_and_nonlocals, find_none
from uncompyle6.show import maybe_show_tree_param_default

def make_function36(self, node, is_lambda, nested=1, code_node=None):
    if False:
        return 10
    'Dump function definition, doc string, and function body in\n    Python version 3.6 and above.\n    '

    def build_param(ast, name, default, annotation=None):
        if False:
            for i in range(10):
                print('nop')
        'build parameters:\n        - handle defaults\n        - handle format tuple parameters\n        '
        value = default
        maybe_show_tree_param_default(self.showast, name, value)
        if annotation:
            result = '%s: %s=%s' % (name, annotation, value)
        else:
            result = '%s=%s' % (name, value)
        if result[-2:] == '= ':
            result += 'None'
        return result
    assert node[-1].kind.startswith('MAKE_')
    lambda_index = -3
    args_node = node[-1]
    annotate_dict = {}
    args_attr = args_node.attr
    if len(args_attr) == 3:
        (_, kw_args, annotate_argc) = args_attr
    else:
        (_, kw_args, annotate_argc, closure) = args_attr
    if node[-2] != 'docstring':
        i = -4
    else:
        i = -5
    if annotate_argc:
        annotate_node = node[i]
        if annotate_node == 'expr':
            annotate_node = annotate_node[0]
            annotate_name_node = annotate_node[-1]
            if annotate_node == 'dict' and annotate_name_node.kind.startswith('BUILD_CONST_KEY_MAP'):
                types = [self.traverse(n, indent='') for n in annotate_node[:-2]]
                names = annotate_node[-2].attr
                length = len(types)
                assert length == len(names)
                for i in range(length):
                    annotate_dict[names[i]] = types[i]
                pass
            pass
        i -= 1
    if closure:
        i -= 1
    defparams = []
    (default, kw_args, annotate_argc) = args_node.attr[0:3]
    if default:
        expr_node = node[0]
        if node[0] == 'pos_arg':
            expr_node = expr_node[0]
        assert expr_node == 'expr', 'expecting mkfunc default node to be an expr'
        if expr_node[0] == 'LOAD_CONST' and isinstance(expr_node[0].attr, tuple):
            defparams = [repr(a) for a in expr_node[0].attr]
        elif expr_node[0] in frozenset(('list', 'tuple', 'dict', 'set')):
            defparams = [self.traverse(n, indent='') for n in expr_node[0][:-1]]
    else:
        defparams = []
    pass
    if lambda_index and is_lambda and iscode(node[lambda_index].attr):
        assert node[lambda_index].kind == 'LOAD_LAMBDA'
        code = node[lambda_index].attr
    else:
        code = code_node.attr
    assert iscode(code)
    debug_opts = self.debug_opts['asm'] if self.debug_opts else None
    scanner_code = Code(code, self.scanner, self.currentclass, debug_opts)
    argc = code.co_argcount
    kwonlyargcount = code.co_kwonlyargcount
    paramnames = list(scanner_code.co_varnames[:argc])
    kwargs = list(scanner_code.co_varnames[argc:argc + kwonlyargcount])
    paramnames.reverse()
    defparams.reverse()
    try:
        tree = self.build_ast(scanner_code._tokens, scanner_code._customize, scanner_code, is_lambda=is_lambda, noneInNames='None' in code.co_names)
    except (ParserError, ParserError2) as p:
        self.write(str(p))
        if not self.tolerate_errors:
            self.ERROR = p
        return
    i = len(paramnames) - len(defparams)
    params = []
    if defparams:
        for (i, defparam) in enumerate(defparams):
            params.append(build_param(tree, paramnames[i], defparam, annotate_dict.get(paramnames[i])))
        for param in paramnames[i + 1:]:
            if param in annotate_dict:
                params.append('%s: %s' % (param, annotate_dict[param]))
            else:
                params.append(param)
    else:
        for param in paramnames:
            if param in annotate_dict:
                params.append('%s: %s' % (param, annotate_dict[param]))
            else:
                params.append(param)
    params.reverse()
    if code_has_star_arg(code):
        star_arg = code.co_varnames[argc + kwonlyargcount]
        if star_arg in annotate_dict:
            params.append('*%s: %s' % (star_arg, annotate_dict[star_arg]))
        else:
            params.append('*%s' % star_arg)
        argc += 1
    if is_lambda:
        self.write('lambda')
        if len(params):
            self.write(' ', ', '.join(params))
        elif kwonlyargcount > 0 and (not 4 & code.co_flags):
            assert argc == 0
            self.write(' ')
        if len(tree) > 1 and self.traverse(tree[-1]) == 'None' and self.traverse(tree[-2]).strip().startswith('yield'):
            del tree[-1]
            tree_expr = tree[-1]
            while tree_expr.kind != 'expr':
                tree_expr = tree_expr[0]
            tree[-1] = tree_expr
            pass
    else:
        self.write('(', ', '.join(params))
    ends_in_comma = False
    if kwonlyargcount > 0:
        if not 4 & code.co_flags:
            if argc > 0:
                self.write(', *, ')
            else:
                self.write('*, ')
            pass
        elif argc > 0:
            self.write(', ')
        kw_dict = None
        fn_bits = node[-1].attr
        index = -5 if node[-2] == 'docstring' else -4
        if fn_bits[-1]:
            index -= 1
        if fn_bits[-2]:
            index -= 1
        if fn_bits[-3]:
            kw_dict = node[index]
            index -= 1
        if fn_bits[-4]:
            pass
        if kw_dict == 'expr':
            kw_dict = kw_dict[0]
        kw_args = [None] * kwonlyargcount
        if kw_dict:
            assert kw_dict == 'dict'
            const_list = kw_dict[0]
            if kw_dict[0] == 'const_list':
                add_consts = const_list[1]
                assert add_consts == 'add_consts'
                names = add_consts[-1].attr
                defaults = [v.pattr for v in add_consts[:-1]]
            else:
                defaults = [self.traverse(n, indent='') for n in kw_dict[:-2]]
                names = eval(self.traverse(kw_dict[-2]))
            assert len(defaults) == len(names)
            for (i, n) in enumerate(names):
                idx = kwargs.index(n)
                if annotate_dict and n in annotate_dict:
                    t = '%s: %s=%s' % (n, annotate_dict[n], defaults[i])
                else:
                    t = '%s=%s' % (n, defaults[i])
                kw_args[idx] = t
                pass
            pass
        other_kw = [c is None for c in kw_args]
        for (i, flag) in enumerate(other_kw):
            if flag:
                n = kwargs[i]
                if n in annotate_dict:
                    kw_args[i] = '%s: %s' % (n, annotate_dict[n])
                else:
                    kw_args[i] = '%s' % n
        self.write(', '.join(kw_args))
        ends_in_comma = False
        pass
    elif argc == 0:
        ends_in_comma = True
    if code_has_star_star_arg(code):
        if not ends_in_comma:
            self.write(', ')
        star_star_arg = code.co_varnames[argc + kwonlyargcount]
        if annotate_dict and star_star_arg in annotate_dict:
            self.write('**%s: %s' % (star_star_arg, annotate_dict[star_star_arg]))
        else:
            self.write('**%s' % star_star_arg)
    if is_lambda:
        self.write(': ')
    else:
        self.write(')')
        if annotate_dict and 'return' in annotate_dict:
            self.write(' -> %s' % annotate_dict['return'])
        self.println(':')
    if node[-2] == 'docstring' and (not is_lambda):
        self.println(self.traverse(node[-2]))
    assert tree in ('stmts', 'lambda_start')
    all_globals = find_all_globals(tree, set())
    (globals, nonlocals) = find_globals_and_nonlocals(tree, set(), set(), code, self.version)
    for g in sorted(all_globals & self.mod_globs | globals):
        self.println(self.indent, 'global ', g)
    for nl in sorted(nonlocals):
        self.println(self.indent, 'nonlocal ', nl)
    self.mod_globs -= all_globals
    has_none = 'None' in code.co_names
    rn = has_none and (not find_none(tree))
    self.gen_source(tree, code.co_name, scanner_code._customize, is_lambda=is_lambda, returnNone=rn, debug_opts=self.debug_opts)
    if not is_lambda and code.co_flags & (CO_GENERATOR | CO_ASYNC_GENERATOR):
        need_bogus_yield = True
        for token in scanner_code._tokens:
            if token == 'YIELD_VALUE':
                need_bogus_yield = False
                break
            pass
        if need_bogus_yield:
            self.template_engine(('%|if False:\n%+%|yield None%-',), node)
    scanner_code._tokens = None
    scanner_code._customize = None