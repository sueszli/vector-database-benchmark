"""
Generators and comprehension functions
"""
from typing import Optional
from xdis import co_flags_is_async, iscode
from uncompyle6.parser import get_python_parser
from uncompyle6.scanner import Code
from uncompyle6.semantics.consts import PRECEDENCE
from uncompyle6.semantics.helper import is_lambda_mode
from uncompyle6.scanners.tok import Token

class ComprehensionMixin:
    """
    These functions hand nonterminal common actions that occur
    when encountering a generator or some sort of comprehension.

    What is common about these is that often the nonterminal has
    a code object whose decompilation needs to be melded into the resulting
    Python source code. In source code, the implicit function calls
    are not seen.
    """

    def closure_walk(self, node, collection_index):
        if False:
            while True:
                i = 10
        '\n        Dictionary and comprehensions using closure the way they are done in Python3.\n        '
        p = self.prec
        self.prec = PRECEDENCE['lambda_body'] - 1
        code_index = 0 if node[0] == 'load_genexpr' else 1
        tree = self.get_comprehension_function(node, code_index=code_index)
        while len(tree) == 1:
            tree = tree[0]
        store = tree[3]
        collection = node[collection_index]
        iter_index = 3 if tree == 'genexpr_func_async' else 4
        n = tree[iter_index]
        list_if = None
        assert n == 'comp_iter'
        while n == 'comp_iter':
            n = n[0]
            if n == 'list_for':
                store = n[2]
                n = n[3]
            elif n in ('list_if', 'list_if_not', 'comp_if', 'comp_if_not'):
                if n[0].kind == 'expr':
                    list_if = n
                else:
                    list_if = n[1]
                n = n[-1]
                pass
            elif n == 'list_if37':
                list_if.append(n)
                n = n[-1]
                pass
            pass
        assert n == 'comp_body', tree
        self.preorder(n[0])
        self.write(' for ')
        self.preorder(store)
        self.write(' in ')
        self.preorder(collection)
        if list_if:
            self.preorder(list_if)
        self.prec = p

    def comprehension_walk(self, node, iter_index: Optional[int], code_index: int=-5):
        if False:
            for i in range(10):
                print('nop')
        p: int = self.prec
        self.prec = PRECEDENCE['lambda_body'] - 1
        if self.version >= (3, 0) and node == 'dict_comp':
            cn = node[1]
        elif self.version <= (2, 7) and node == 'generator_exp':
            if node[0] == 'LOAD_GENEXPR':
                cn = node[0]
            elif node[0] == 'load_closure':
                cn = node[1]
        elif self.version >= (3, 0) and node in ('generator_exp', 'generator_exp_async'):
            if node[0] == 'load_genexpr':
                load_genexpr = node[0]
            elif node[1] == 'load_genexpr':
                load_genexpr = node[1]
            cn = load_genexpr[0]
        elif hasattr(node[code_index], 'attr'):
            cn = node[code_index]
        elif len(node[1]) > 1 and hasattr(node[1][1], 'attr'):
            cn = node[1][1]
        elif hasattr(node[1][0], 'attr'):
            cn = node[1][0]
        else:
            assert False, "Can't find code for comprehension"
        assert iscode(cn.attr)
        code = Code(cn.attr, self.scanner, self.currentclass, self.debug_opts['asm'])
        if is_lambda_mode(self.compile_mode):
            p_save = self.p
            self.p = get_python_parser(self.version, compile_mode='exec', is_pypy=self.is_pypy)
            tree = self.build_ast(code._tokens, code._customize, code)
            self.p = p_save
        else:
            tree = self.build_ast(code._tokens, code._customize, code)
        self.customize(code._customize)
        while len(tree) == 1:
            tree = tree[0]
        if tree == 'stmts':
            tree = tree[0]
        elif tree == 'lambda_start':
            assert len(tree) <= 3
            tree = tree[-2]
            if tree == 'return_expr_lambda':
                tree = tree[1]
            pass
        if tree in ('genexpr_func', 'genexpr_func_async'):
            for i in range(3, 5):
                if tree[i] == 'comp_iter':
                    iter_index = i
                    break
        n = tree[iter_index]
        assert n == 'comp_iter', n.kind
        while n == 'comp_iter':
            n = n[0]
            if n == 'comp_for':
                if n[0] == 'SETUP_LOOP':
                    n = n[4]
                else:
                    n = n[3]
            elif n == 'comp_if':
                n = n[2]
            elif n == 'comp_if_not':
                n = n[2]
        assert n == 'comp_body', n
        self.preorder(n[0])
        if node == 'generator_exp_async':
            self.write(' async')
            iter_var_index = iter_index - 2
        else:
            iter_var_index = iter_index - 1
        self.write(' for ')
        self.preorder(tree[iter_var_index])
        self.write(' in ')
        if node[2] == 'expr':
            iter_expr = node[2]
        elif node[3] in ('expr', 'get_aiter'):
            iter_expr = node[3]
        else:
            iter_expr = node[-3]
        assert iter_expr in ('expr', 'get_aiter'), iter_expr
        self.preorder(iter_expr)
        self.preorder(tree[iter_index])
        self.prec = p

    def comprehension_walk_newer(self, node, iter_index: Optional[int], code_index: int=-5, collection_node=None):
        if False:
            while True:
                i = 10
        'Non-closure-based comprehensions the way they are done in Python3\n        and some Python 2.7. Note: there are also other set comprehensions.\n\n        Note: there are also other comprehensions.\n        '
        p = self.prec
        self.prec = PRECEDENCE['lambda_body'] - 1
        comp_for = None
        if isinstance(node[0], Token) and node[0].kind.startswith('LOAD') and iscode(node[0].attr):
            if node[3] == 'get_aiter':
                compile_mode = self.compile_mode
                self.compile_mode = 'genexpr'
                is_lambda = self.is_lambda
                self.is_lambda = True
                tree = self.get_comprehension_function(node, code_index)
                self.compile_mode = compile_mode
                self.is_lambda = is_lambda
            else:
                tree = self.get_comprehension_function(node, code_index)
        elif len(node) > 2 and isinstance(node[2], Token) and node[2].kind.startswith('LOAD') and iscode(node[2].attr):
            tree = self.get_comprehension_function(node, 2)
        else:
            tree = node
        is_30_dict_comp = False
        store = None
        if node == 'list_comp_async':
            if tree[0] == 'expr' and tree[0][0] == 'list_comp_async':
                tree = tree[0][0]
            if tree[0] == 'BUILD_LIST_0':
                list_afor2 = tree[2]
                assert list_afor2 == 'list_afor2'
                store = list_afor2[1]
                assert store == 'store'
                n = list_afor2[3] if list_afor2[3] == 'list_iter' else list_afor2[2]
            else:
                pass
        elif node.kind in ('dict_comp_async', 'set_comp_async'):
            if tree[0] == 'expr':
                tree = tree[0]
            if tree[0].kind in ('BUILD_MAP_0', 'BUILD_SET_0'):
                genexpr_func_async = tree[1]
                if genexpr_func_async == 'genexpr_func_async':
                    store = genexpr_func_async[2]
                    assert store.kind.startswith('store')
                    n = genexpr_func_async[4]
                    assert n == 'comp_iter'
                    comp_for = collection_node
                else:
                    set_afor2 = genexpr_func_async
                    assert set_afor2 == 'set_afor2'
                    n = set_afor2[1]
                    store = n[1]
                    comp_for = node[3]
            else:
                pass
        elif node == 'list_afor':
            comp_for = node[0]
            list_afor2 = node[1]
            assert list_afor2 == 'list_afor2'
            store = list_afor2[1]
            assert store == 'store'
            n = list_afor2[2]
        elif node == 'set_afor2':
            comp_for = node[0]
            set_iter_async = node[1]
            assert set_iter_async == 'set_iter_async'
            store = set_iter_async[1]
            assert store == 'store'
            n = set_iter_async[2]
        elif node == 'list_comp' and tree[0] == 'expr':
            tree = tree[0][0]
            n = tree[iter_index]
        else:
            n = tree[iter_index]
        if tree in ('dict_comp_func', 'genexpr_func_async', 'generator_exp', 'list_comp', 'set_comp', 'set_comp_func', 'set_comp_func_header'):
            for k in tree:
                if k.kind in ('comp_iter', 'list_iter', 'set_iter', 'await_expr'):
                    n = k
                elif k == 'store':
                    store = k
                    pass
                pass
            pass
        elif tree.kind in ('list_comp_async', 'dict_comp_async', 'set_afor2'):
            if self.version == (3, 0):
                for k in tree:
                    if k in ('dict_comp_header', 'set_comp_header'):
                        n = k
                    elif k == 'store':
                        store = k
                    elif k == 'dict_comp_iter':
                        is_30_dict_comp = True
                        n = (k[3], k[1])
                        pass
                    elif k == 'comp_iter':
                        n = k[0]
                        pass
                    pass
        elif tree == 'list_comp_async':
            store = tree[2][1]
        else:
            if n.kind in ('RETURN_VALUE_LAMBDA', 'return_expr_lambda'):
                self.prune()
            assert n in ('list_iter', 'comp_iter'), n
        if_node = None
        comp_store = None
        if n == 'comp_iter' and store is None:
            comp_for = n
            comp_store = tree[3]
        have_not = False
        while n in ('list_iter', 'list_afor', 'list_afor2', 'comp_iter'):
            if self.version == (3, 0) and len(n) == 3:
                assert n[0] == 'expr' and n[1] == 'expr'
                n = n[1]
            elif n == 'list_afor':
                n = n[1]
            elif n == 'list_afor2':
                if n[1] == 'store':
                    store = n[1]
                n = n[3]
            else:
                n = n[0]
            if n in ('list_for', 'comp_for'):
                n_index = 3
                if n[2] == 'store' or ((self.version == (3, 0) and n[4] == 'store') and (not store)):
                    if self.version == (3, 0):
                        store = n[4]
                        n_index = 5
                    else:
                        store = n[2]
                    if not comp_store:
                        comp_store = store
                n = n[n_index]
            elif n in ('list_if', 'list_if_not', 'list_if37', 'list_if37_not', 'comp_if', 'comp_if_not'):
                have_not = n in ('list_if_not', 'comp_if_not', 'list_if37_not')
                if n in ('list_if37', 'list_if37_not'):
                    n = n[1]
                else:
                    if_node = n[0]
                    if n[1] == 'store':
                        store = n[1]
                    n = n[2]
                    pass
            pass
        if self.version != (3, 0) and self.version < (3, 7):
            assert n.kind in ('lc_body', 'list_if37', 'comp_body', 'set_comp_func', 'set_comp_body'), tree
        assert store, "Couldn't find store in list/set comprehension"
        if is_30_dict_comp:
            self.preorder(n[0])
            self.write(': ')
            self.preorder(n[1])
        else:
            if self.version == (3, 0):
                if isinstance(n, Token):
                    body = store
                elif len(n) > 1:
                    body = n[1]
                else:
                    body = n[0]
            else:
                body = n[0]
            self.preorder(body)
        if node == 'list_comp_async':
            self.write(' async')
            in_node_index = 3
        else:
            in_node_index = -3
        self.write(' for ')
        if comp_store:
            self.preorder(comp_store)
            comp_store = None
        else:
            self.preorder(store)
        self.write(' in ')
        if comp_for:
            self.preorder(comp_for)
        else:
            self.preorder(node[in_node_index])
        if tree == 'list_comp' and self.version != (3, 0):
            list_iter = tree[1]
            assert list_iter == 'list_iter'
            if list_iter[0] == 'list_for':
                self.preorder(list_iter[0][3])
                self.prec = p
                return
            pass
        if comp_store:
            self.preorder(comp_for)
        if if_node:
            self.write(' if ')
            if have_not:
                self.write('not ')
                pass
            self.prec = PRECEDENCE['lambda_body'] - 1
            self.preorder(if_node)
            pass
        self.prec = p

    def get_comprehension_function(self, node, code_index: int):
        if False:
            i = 10
            return i + 15
        '\n        Build the body of a comprehension function and then\n        find the comprehension node buried in the tree which may\n        be surrounded with start-like symbols or dominiators,.\n        '
        self.prec = PRECEDENCE['lambda_body'] - 1
        code_node = node[code_index]
        if code_node == 'load_genexpr':
            code_node = code_node[0]
        code_obj = code_node.attr
        assert iscode(code_obj), code_node
        code = Code(code_obj, self.scanner, self.currentclass, self.debug_opts['asm'])
        if self.compile_mode in ('listcomp',):
            p_save = self.p
            self.p = get_python_parser(self.version, compile_mode='exec', is_pypy=self.is_pypy)
            tree = self.build_ast(code._tokens, code._customize, code, is_lambda=self.is_lambda)
            self.p = p_save
        else:
            tree = self.build_ast(code._tokens, code._customize, code, is_lambda=self.is_lambda)
        self.customize(code._customize)
        if tree == 'lambda_start':
            if tree[0] in ('dom_start', 'dom_start_opt'):
                tree = tree[1]
        while len(tree) == 1 or tree in ('stmt', 'sstmt', 'return', 'return_expr'):
            self.prec = 100
            tree = tree[1] if tree[0] in ('dom_start', 'dom_start_opt') else tree[0]
        return tree

    def listcomp_closure3(self, node):
        if False:
            i = 10
            return i + 15
        '\n        List comprehensions in Python 3 when handled as a closure.\n        See if we can combine code.\n        '
        p = self.prec
        self.prec = 27
        code_obj = node[1].attr
        assert iscode(code_obj), node[1]
        code = Code(code_obj, self.scanner, self.currentclass, self.debug_opts['asm'])
        tree = self.build_ast(code._tokens, code._customize, code)
        self.customize(code._customize)
        while len(tree) == 1 or (tree in ('sstmt', 'return') and tree[-1] in ('RETURN_LAST', 'RETURN_VALUE')):
            self.prec = 100
            tree = tree[0]
        n = tree[1]
        collections = [node[-3]]
        list_ifs = []
        if self.version[:2] == (3, 0) and n.kind != 'list_iter':
            stores = [tree[3]]
            assert tree[4] == 'comp_iter'
            n = tree[4]
            while n == 'comp_iter':
                if n[0] == 'comp_for':
                    n = n[0]
                    stores.append(n[2])
                    n = n[3]
                elif n[0] in ('comp_if', 'comp_if_not'):
                    n = n[0]
                    if n[0].kind == 'expr':
                        list_ifs.append(n)
                    else:
                        list_ifs.append([1])
                    n = n[2]
                    pass
                else:
                    break
                pass
            self.preorder(n[1])
        else:
            assert n == 'list_iter'
            stores = []
            while n == 'list_iter':
                n = n[0]
                if n == 'list_for':
                    stores.append(n[2])
                    if self.version[:2] == (3, 0):
                        body_index = 5
                    else:
                        body_index = 3
                    n = n[body_index]
                    if n[0] == 'list_for':
                        c = n[0][0]
                        if c == 'expr':
                            c = c[0]
                        if c == 'attribute':
                            c = c[0]
                        collections.append(c)
                        pass
                elif n in ('list_if', 'list_if_not', 'list_if_or_not'):
                    if n[0].kind == 'expr':
                        list_ifs.append(n)
                    else:
                        list_ifs.append([1])
                    if self.version[:2] == (3, 0) and n[2] == 'list_iter':
                        n = n[2]
                    else:
                        n = n[-2] if n[-1] == 'come_from_opt' else n[-1]
                    pass
                elif n == 'list_if37':
                    list_ifs.append(n)
                    n = n[-1]
                    pass
                elif n == 'list_afor':
                    collections.append(n[0][0])
                    n = n[1]
                    stores.append(n[1][0])
                    n = n[2] if n[2].kind == 'list_iter' else n[3]
                pass
            assert n == 'lc_body', tree
            if self.version[:2] == (3, 0):
                body_index = 1
            else:
                body_index = 0
            self.preorder(n[body_index])
        n_colls = len(collections)
        for (i, store) in enumerate(stores):
            if i >= n_colls:
                break
            token = collections[i]
            if not isinstance(token, Token):
                token = token.first_child()
            if token == 'LOAD_DEREF' and co_flags_is_async(code_obj.co_flags):
                self.write(' async')
                pass
            self.write(' for ')
            if self.version[:2] == (3, 0):
                store = token
            self.preorder(store)
            self.write(' in ')
            self.preorder(collections[i])
            if i < len(list_ifs):
                self.preorder(list_ifs[i])
                pass
            pass
        self.prec = p