"""
All the crazy things we have to do to handle Python functions in 3.0-3.5 or so.
The saga of changes before and after is in other files.
"""
from xdis import iscode, code_has_star_arg, code_has_star_star_arg, CO_GENERATOR
from uncompyle6.scanner import Code
from uncompyle6.parsers.treenode import SyntaxTree
from uncompyle6.semantics.parser_error import ParserError
from uncompyle6.parser import ParserError as ParserError2
from uncompyle6.semantics.helper import print_docstring, find_all_globals, find_globals_and_nonlocals, find_none
from uncompyle6.show import maybe_show_tree_param_default

def make_function3_annotate(self, node, is_lambda, nested=1, code_node=None, annotate_last=-1):
    if False:
        print('Hello World!')
    '\n    Dump function defintion, doc string, and function\n    body. This code is specialized for Python 3'

    def build_param(ast, name, default):
        if False:
            print('Hello World!')
        'build parameters:\n            - handle defaults\n            - handle format tuple parameters\n        '
        if default:
            value = self.traverse(default, indent='')
            maybe_show_tree_param_default(self, name, value)
            result = '%s=%s' % (name, value)
            if result[-2:] == '= ':
                result += 'None'
            return result
        else:
            return name
    assert node[-1].kind.startswith('MAKE_')
    annotate_tuple = None
    for annotate_last in range(len(node) - 1, -1, -1):
        if node[annotate_last] == 'annotate_tuple':
            annotate_tuple = node[annotate_last]
            break
    annotate_args = {}
    if annotate_tuple == 'annotate_tuple' and annotate_tuple[0] in ('LOAD_CONST', 'LOAD_NAME') and isinstance(annotate_tuple[0].attr, tuple):
        annotate_tup = annotate_tuple[0].attr
        i = -1
        j = annotate_last - 1
        l = -len(node)
        while j >= l and node[j].kind in ('annotate_arg', 'annotate_tuple'):
            annotate_args[annotate_tup[i]] = node[j][0]
            i -= 1
            j -= 1
    args_node = node[-1]
    if isinstance(args_node.attr, tuple):
        defparams = node[:args_node.attr[0]]
        (pos_args, kw_args, annotate_argc) = args_node.attr
    else:
        defparams = node[:args_node.attr]
        kw_args = 0
        pass
    annotate_dict = {}
    for name in annotate_args.keys():
        n = self.traverse(annotate_args[name], indent='')
        annotate_dict[name] = n
    if (3, 0) <= self.version < (3, 3):
        lambda_index = -2
    elif self.version < (3, 4):
        lambda_index = -3
    else:
        lambda_index = None
    if lambda_index and is_lambda and iscode(node[lambda_index].attr):
        assert node[lambda_index].kind == 'LOAD_LAMBDA'
        code = node[lambda_index].attr
    else:
        code = code_node.attr
    assert iscode(code)
    code = Code(code, self.scanner, self.currentclass)
    argc = code.co_argcount
    kwonlyargcount = code.co_kwonlyargcount
    paramnames = list(code.co_varnames[:argc])
    if kwonlyargcount > 0:
        kwargs = list(code.co_varnames[argc:argc + kwonlyargcount])
    try:
        ast = self.build_ast(code._tokens, code._customize, code, is_lambda=is_lambda, noneInNames='None' in code.co_names)
    except (ParserError, ParserError2) as p:
        self.write(str(p))
        if not self.tolerate_errors:
            self.ERROR = p
        return
    indent = self.indent
    if is_lambda:
        self.write('lambda ')
    else:
        self.write('(')
    last_line = self.f.getvalue().split('\n')[-1]
    l = len(last_line)
    indent = ' ' * l
    line_number = self.line_number
    i = len(paramnames) - len(defparams)
    suffix = ''
    for param in paramnames[:i]:
        self.write(suffix, param)
        suffix = ', '
        if param in annotate_dict:
            self.write(': %s' % annotate_dict[param])
            if line_number != self.line_number:
                suffix = ',\n' + indent
                line_number = self.line_number
    suffix = ', ' if i > 0 else ''
    for n in node:
        if n == 'pos_arg':
            self.write(suffix)
            param = paramnames[i]
            self.write(param)
            if param in annotate_args:
                aa = annotate_args[param]
                if isinstance(aa, tuple):
                    aa = aa[0]
                    self.write(': "%s"' % aa)
                elif isinstance(aa, SyntaxTree):
                    self.write(': ')
                    self.preorder(aa)
            self.write('=')
            i += 1
            self.preorder(n)
            if line_number != self.line_number:
                suffix = ',\n' + indent
                line_number = self.line_number
            else:
                suffix = ', '
    if code_has_star_arg(code):
        star_arg = code.co_varnames[argc + kwonlyargcount]
        if annotate_dict and star_arg in annotate_dict:
            self.write(suffix, '*%s: %s' % (star_arg, annotate_dict[star_arg]))
        else:
            self.write(suffix, '*%s' % star_arg)
        argc += 1
    ends_in_comma = False
    if kwonlyargcount > 0:
        if not code_has_star_arg(code):
            if argc > 0:
                self.write(', *, ')
            else:
                self.write('*, ')
            pass
            ends_in_comma = True
        elif argc > 0:
            self.write(', ')
            ends_in_comma = True
        kw_args = [None] * kwonlyargcount
        for n in node:
            if n == 'kwargs':
                n = n[0]
            if n == 'kwarg':
                name = eval(n[0].pattr)
                idx = kwargs.index(name)
                default = self.traverse(n[1], indent='')
                if annotate_dict and name in annotate_dict:
                    kw_args[idx] = '%s: %s=%s' % (name, annotate_dict[name], default)
                else:
                    kw_args[idx] = '%s=%s' % (name, default)
                pass
            pass
        other_kw = [c == None for c in kw_args]
        for (i, flag) in enumerate(other_kw):
            if flag:
                n = kwargs[i]
                if n in annotate_dict:
                    kw_args[i] = '%s: %s' % (n, annotate_dict[n])
                else:
                    kw_args[i] = '%s' % n
        self.write(', '.join(kw_args))
        ends_in_comma = False
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
        if 'return' in annotate_tuple[0].attr:
            if line_number != self.line_number and (not no_paramnames):
                self.write('\n' + indent)
                line_number = self.line_number
            self.write(' -> ')
            if 'return' in annotate_dict:
                self.write(annotate_dict['return'])
            else:
                self.preorder(node[annotate_last - 1])
        self.println(':')
    if len(code.co_consts) > 0 and code.co_consts[0] is not None and (not is_lambda):
        print_docstring(self, self.indent, code.co_consts[0])
    code._tokens = None
    assert ast == 'stmts'
    all_globals = find_all_globals(ast, set())
    (globals, nonlocals) = find_globals_and_nonlocals(ast, set(), set(), code, self.version)
    for g in sorted(all_globals & self.mod_globs | globals):
        self.println(self.indent, 'global ', g)
    for nl in sorted(nonlocals):
        self.println(self.indent, 'nonlocal ', nl)
    self.mod_globs -= all_globals
    has_none = 'None' in code.co_names
    rn = has_none and (not find_none(ast))
    self.gen_source(ast, code.co_name, code._customize, is_lambda=is_lambda, returnNone=rn)
    code._tokens = code._customize = None

def make_function3(self, node, is_lambda, nested=1, code_node=None):
    if False:
        return 10
    'Dump function definition, doc string, and function body in\n      Python version 3.0 and above\n    '

    def build_param(ast, name, default, annotation=None):
        if False:
            return 10
        'build parameters:\n            - handle defaults\n            - handle format tuple parameters\n        '
        value = self.traverse(default, indent='')
        maybe_show_tree_param_default(self.showast, name, value)
        if annotation:
            result = '%s: %s=%s' % (name, annotation, value)
        else:
            result = '%s=%s' % (name, value)
        if result[-2:] == '= ':
            result += 'None'
        return result
    assert node[-1].kind.startswith('MAKE_')
    if (3, 0) <= self.version <= (3, 2):
        lambda_index = -2
    elif (3, 3) <= self.version:
        lambda_index = -3
    else:
        lambda_index = None
    args_node = node[-1]
    annotate_dict = {}
    args_attr = args_node.attr
    if isinstance(args_attr, tuple):
        if len(args_attr) == 3:
            (pos_args, kw_args, annotate_argc) = args_attr
        else:
            (pos_args, kw_args, annotate_argc, closure) = args_attr
            i = -4
            kw_pairs = 0
            if closure:
                i -= 1
            if annotate_argc:
                annotate_node = node[i]
                if annotate_node == 'expr':
                    annotate_node = annotate_node[0]
                    annotate_name_node = annotate_node[-1]
                    if annotate_node == 'dict' and annotate_name_node.kind.startswith('BUILD_CONST_KEY_MAP'):
                        types = [self.traverse(n, indent='') for n in annotate_node[:-2]]
                        names = annotate_node[-2].attr
                        l = len(types)
                        assert l == len(names)
                        for i in range(l):
                            annotate_dict[names[i]] = types[i]
                        pass
                    pass
                i -= 1
            if kw_args:
                kw_node = node[i]
                if kw_node == 'expr':
                    kw_node = kw_node[0]
                if kw_node == 'dict':
                    kw_pairs = kw_node[-1].attr
        have_kwargs = node[0].kind.startswith('kwarg') or node[0] == 'no_kwargs'
        if len(node) >= 4:
            lc_index = -4
        else:
            lc_index = -3
            pass
        if len(node) > 2 and (have_kwargs or node[lc_index].kind != 'load_closure'):
            default_values_start = 0
            if node[0] == 'no_kwargs':
                default_values_start += 1
            if node[default_values_start] == 'kwarg':
                assert node[lambda_index] == 'LOAD_LAMBDA'
                i = default_values_start
                defparams = []
                while node[i] == 'kwarg':
                    defparams.append(node[i][1])
                    i += 1
            else:
                if node[default_values_start] == 'kwargs':
                    default_values_start += 1
                defparams = node[default_values_start:default_values_start + args_node.attr[0]]
        else:
            defparams = node[:args_node.attr[0]]
            kw_args = 0
    else:
        defparams = node[:args_node.attr]
        kw_args = 0
        pass
    if lambda_index and is_lambda and iscode(node[lambda_index].attr):
        assert node[lambda_index].kind == 'LOAD_LAMBDA'
        code = node[lambda_index].attr
    else:
        code = code_node.attr
    assert iscode(code)
    scanner_code = Code(code, self.scanner, self.currentclass)
    argc = code.co_argcount
    kwonlyargcount = code.co_kwonlyargcount
    paramnames = list(scanner_code.co_varnames[:argc])
    if kwonlyargcount > 0:
        if is_lambda:
            kwargs = []
            for i in range(kwonlyargcount):
                paramnames.append(scanner_code.co_varnames[argc + i])
            pass
        else:
            kwargs = list(scanner_code.co_varnames[argc:argc + kwonlyargcount])
    paramnames.reverse()
    defparams.reverse()
    try:
        ast = self.build_ast(scanner_code._tokens, scanner_code._customize, scanner_code, is_lambda=is_lambda, noneInNames='None' in code.co_names)
    except (ParserError, ParserError2) as p:
        self.write(str(p))
        if not self.tolerate_errors:
            self.ERROR = p
        return
    kw_pairs = 0
    i = len(paramnames) - len(defparams)
    params = []
    if defparams:
        for (i, defparam) in enumerate(defparams):
            params.append(build_param(ast, paramnames[i], defparam, annotate_dict.get(paramnames[i])))
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
        if annotate_dict and star_arg in annotate_dict:
            params.append('*%s: %s' % (star_arg, annotate_dict[star_arg]))
        else:
            params.append('*%s' % star_arg)
            pass
        if is_lambda:
            params.reverse()
        if not is_lambda:
            argc += 1
        pass
    elif is_lambda and kwonlyargcount > 0:
        params.insert(0, '*')
        kwonlyargcount = 0
    if is_lambda:
        self.write('lambda ', ', '.join(params))
        if len(ast) > 1 and self.traverse(ast[-1]) == 'None' and self.traverse(ast[-2]).strip().startswith('yield'):
            del ast[-1]
            ast_expr = ast[-1]
            while ast_expr.kind != 'expr':
                ast_expr = ast_expr[0]
            ast[-1] = ast_expr
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
            ends_in_comma = True
        elif argc > 0 and node[0] != 'kwarg':
            self.write(', ')
            ends_in_comma = True
        kw_args = [None] * kwonlyargcount
        if self.version <= (3, 3):
            kw_nodes = node[0]
        else:
            kw_nodes = node[args_node.attr[0]]
        if kw_nodes == 'kwargs':
            for n in kw_nodes:
                name = eval(n[0].pattr)
                default = self.traverse(n[1], indent='')
                idx = kwargs.index(name)
                kw_args[idx] = '%s=%s' % (name, default)
                pass
            pass
        if kw_nodes != 'kwarg' or self.version == 3.5:
            other_kw = [c == None for c in kw_args]
            for (i, flag) in enumerate(other_kw):
                if flag:
                    if i < len(kwargs):
                        kw_args[i] = '%s' % kwargs[i]
                    else:
                        del kw_args[i]
                    pass
            self.write(', '.join(kw_args))
            ends_in_comma = False
            pass
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
    if len(code.co_consts) > 0 and code.co_consts[0] is not None and (not is_lambda):
        print_docstring(self, self.indent, code.co_consts[0])
    assert ast == 'stmts'
    all_globals = find_all_globals(ast, set())
    (globals, nonlocals) = find_globals_and_nonlocals(ast, set(), set(), code, self.version)
    for g in sorted(all_globals & self.mod_globs | globals):
        self.println(self.indent, 'global ', g)
    for nl in sorted(nonlocals):
        self.println(self.indent, 'nonlocal ', nl)
    self.mod_globs -= all_globals
    has_none = 'None' in code.co_names
    rn = has_none and (not find_none(ast))
    self.gen_source(ast, code.co_name, scanner_code._customize, is_lambda=is_lambda, returnNone=rn, debug_opts=self.debug_opts)
    if not is_lambda and code.co_flags & CO_GENERATOR:
        need_bogus_yield = True
        for token in scanner_code._tokens:
            if token in ('YIELD_VALUE', 'YIELD_FROM'):
                need_bogus_yield = False
                break
            pass
        if need_bogus_yield:
            self.template_engine(('%|if False:\n%+%|yield None%-',), node)
    scanner_code._tokens = None
    scanner_code._customize = None