"""AST manipulation utilities."""
import ast
import gast
from nvidia.dali._autograph.pyct import anno
from nvidia.dali._autograph.pyct import parser
from nvidia.dali._autograph.pyct import qual_names

class CleanCopier(object):
    """NodeTransformer-like visitor that copies an AST."""

    def __init__(self, preserve_annos):
        if False:
            return 10
        super(CleanCopier, self).__init__()
        self.preserve_annos = preserve_annos

    def copy(self, node):
        if False:
            while True:
                i = 10
        'Returns a deep copy of node (excluding some fields, see copy_clean).'
        if isinstance(node, list):
            return [self.copy(n) for n in node]
        elif isinstance(node, tuple):
            return tuple((self.copy(n) for n in node))
        elif not isinstance(node, (gast.AST, ast.AST)):
            return node
        assert isinstance(node, (gast.AST, ast.AST))
        new_fields = {}
        for f in node._fields:
            if not f.startswith('__') and hasattr(node, f):
                new_fields[f] = self.copy(getattr(node, f))
        new_node = type(node)(**new_fields)
        if self.preserve_annos:
            for k in self.preserve_annos:
                anno.copyanno(node, new_node, k)
        return new_node

def copy_clean(node, preserve_annos=None):
    if False:
        while True:
            i = 10
    "Creates a deep copy of an AST.\n\n  The copy will not include fields that are prefixed by '__', with the\n  exception of user-specified annotations.\n\n  Args:\n    node: ast.AST\n    preserve_annos: Optional[Set[Hashable]], annotation keys to include in the\n        copy\n  Returns:\n    ast.AST\n  "
    return CleanCopier(preserve_annos).copy(node)

class SymbolRenamer(gast.NodeTransformer):
    """Transformer that can rename symbols to a simple names."""

    def __init__(self, name_map):
        if False:
            for i in range(10):
                print('nop')
        self.name_map = name_map

    def _process_name_node(self, node):
        if False:
            print('Hello World!')
        qn = anno.getanno(node, anno.Basic.QN)
        if qn in self.name_map:
            new_node = gast.Name(str(self.name_map[qn]), ctx=node.ctx, annotation=None, type_comment=None)
            for k in anno.keys(node):
                anno.copyanno(node, new_node, k)
            return new_node
        return self.generic_visit(node)

    def _process_list_of_strings(self, names):
        if False:
            print('Hello World!')
        for i in range(len(names)):
            qn = qual_names.QN(names[i])
            if qn in self.name_map:
                names[i] = str(self.name_map[qn])
        return names

    def visit_Nonlocal(self, node):
        if False:
            i = 10
            return i + 15
        node.names = self._process_list_of_strings(node.names)
        return node

    def visit_Global(self, node):
        if False:
            while True:
                i = 10
        node.names = self._process_list_of_strings(node.names)
        return node

    def visit_Name(self, node):
        if False:
            while True:
                i = 10
        return self._process_name_node(node)

    def visit_Attribute(self, node):
        if False:
            while True:
                i = 10
        if anno.hasanno(node, anno.Basic.QN):
            return self._process_name_node(node)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if False:
            i = 10
            return i + 15
        qn = qual_names.QN(node.name)
        if qn in self.name_map:
            node.name = str(self.name_map[qn])
        return self.generic_visit(node)

def rename_symbols(node, name_map):
    if False:
        for i in range(10):
            print('nop')
    'Renames symbols in an AST. Requires qual_names annotations.'
    renamer = SymbolRenamer(name_map)
    if isinstance(node, list):
        return [renamer.visit(n) for n in node]
    elif isinstance(node, tuple):
        return tuple((renamer.visit(n) for n in node))
    return renamer.visit(node)

def keywords_to_dict(keywords):
    if False:
        return 10
    'Converts a list of ast.keyword objects to a dict.'
    keys = []
    values = []
    for kw in keywords:
        keys.append(gast.Constant(kw.arg, kind=None))
        values.append(kw.value)
    return gast.Dict(keys=keys, values=values)

class PatternMatcher(gast.NodeVisitor):
    """Matches a node against a pattern represented by a node."""

    def __init__(self, pattern):
        if False:
            return 10
        self.pattern = pattern
        self.pattern_stack = []
        self.matches = True

    def compare_and_visit(self, node, pattern):
        if False:
            return 10
        self.pattern_stack.append(self.pattern)
        self.pattern = pattern
        self.generic_visit(node)
        self.pattern = self.pattern_stack.pop()

    def no_match(self):
        if False:
            print('Hello World!')
        self.matches = False
        return False

    def is_wildcard(self, p):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(p, (list, tuple)) and len(p) == 1:
            (p,) = p
        if isinstance(p, gast.Name) and p.id == '_':
            return True
        if p == '_':
            return True
        return False

    def generic_visit(self, node):
        if False:
            for i in range(10):
                print('nop')
        if not self.matches:
            return
        pattern = self.pattern
        for f in node._fields:
            if f.startswith('__'):
                continue
            if not hasattr(node, f):
                if hasattr(pattern, f) and getattr(pattern, f):
                    return self.no_match()
                else:
                    continue
            if not hasattr(pattern, f):
                return self.no_match()
            v = getattr(node, f)
            p = getattr(pattern, f)
            if self.is_wildcard(p):
                continue
            if isinstance(v, (list, tuple)):
                if not isinstance(p, (list, tuple)) or len(v) != len(p):
                    return self.no_match()
                for (v_item, p_item) in zip(v, p):
                    self.compare_and_visit(v_item, p_item)
            elif isinstance(v, (gast.AST, ast.AST)):
                if not isinstance(v, type(p)) and (not isinstance(p, type(v))):
                    return self.no_match()
                self.compare_and_visit(v, p)
            elif v != p:
                return self.no_match()

def matches(node, pattern):
    if False:
        return 10
    "Basic pattern matcher for AST.\n\n  The pattern may contain wildcards represented by the symbol '_'. A node\n  matches a pattern if for every node in the tree, either there is a node of\n  the same type in pattern, or a Name node with id='_'.\n\n  Args:\n    node: ast.AST\n    pattern: ast.AST\n  Returns:\n    bool\n  "
    if isinstance(pattern, str):
        pattern = parser.parse_str(pattern)
    matcher = PatternMatcher(pattern)
    matcher.visit(node)
    return matcher.matches

def apply_to_single_assignments(targets, values, apply_fn):
    if False:
        return 10
    'Applies a function to each individual assignment.\n\n  This function can process a possibly-unpacked (e.g. a, b = c, d) assignment.\n  It tries to break down the unpacking if possible. In effect, it has the same\n  effect as passing the assigned values in SSA form to apply_fn.\n\n  Examples:\n\n  The following will result in apply_fn(a, c), apply_fn(b, d):\n\n      a, b = c, d\n\n  The following will result in apply_fn(a, c[0]), apply_fn(b, c[1]):\n\n      a, b = c\n\n  The following will result in apply_fn(a, (b, c)):\n\n      a = b, c\n\n  It uses the visitor pattern to allow subclasses to process single\n  assignments individually.\n\n  Args:\n    targets: Union[List[ast.AST, ...], Tuple[ast.AST, ...], ast.AST, should be\n        used with the targets field of an ast.Assign node\n    values: ast.AST\n    apply_fn: Callable[[ast.AST, ast.AST], None], called with the\n        respective nodes of each single assignment\n  '
    if not isinstance(targets, (list, tuple)):
        targets = (targets,)
    for target in targets:
        if isinstance(target, (gast.Tuple, gast.List)):
            for i in range(len(target.elts)):
                target_el = target.elts[i]
                if isinstance(values, (gast.Tuple, gast.List)):
                    value_el = values.elts[i]
                else:
                    idx = parser.parse_expression(str(i))
                    value_el = gast.Subscript(values, idx, ctx=gast.Load())
                apply_to_single_assignments(target_el, value_el, apply_fn)
        else:
            apply_fn(target, values)

def parallel_walk(node, other):
    if False:
        return 10
    "Walks two ASTs in parallel.\n\n  The two trees must have identical structure.\n\n  Args:\n    node: Union[ast.AST, Iterable[ast.AST]]\n    other: Union[ast.AST, Iterable[ast.AST]]\n  Yields:\n    Tuple[ast.AST, ast.AST]\n  Raises:\n    ValueError: if the two trees don't have identical structure.\n  "
    if isinstance(node, (list, tuple)):
        node_stack = list(node)
    else:
        node_stack = [node]
    if isinstance(other, (list, tuple)):
        other_stack = list(other)
    else:
        other_stack = [other]
    while node_stack and other_stack:
        assert len(node_stack) == len(other_stack)
        n = node_stack.pop()
        o = other_stack.pop()
        if not isinstance(n, (ast.AST, gast.AST, str)) and n is not None or (not isinstance(o, (ast.AST, gast.AST, str)) and n is not None) or n.__class__.__name__ != o.__class__.__name__:
            raise ValueError('inconsistent nodes: {} ({}) and {} ({})'.format(n, n.__class__.__name__, o, o.__class__.__name__))
        yield (n, o)
        if isinstance(n, str):
            assert isinstance(o, str), 'The check above should have ensured this'
            continue
        if n is None:
            assert o is None, 'The check above should have ensured this'
            continue
        for f in n._fields:
            n_child = getattr(n, f, None)
            o_child = getattr(o, f, None)
            if f.startswith('__') or n_child is None or o_child is None:
                continue
            if isinstance(n_child, (list, tuple)):
                if not isinstance(o_child, (list, tuple)) or len(n_child) != len(o_child):
                    raise ValueError('inconsistent values for field {}: {} and {}'.format(f, n_child, o_child))
                node_stack.extend(n_child)
                other_stack.extend(o_child)
            elif isinstance(n_child, (gast.AST, ast.AST)):
                node_stack.append(n_child)
                other_stack.append(o_child)
            elif n_child != o_child:
                raise ValueError('inconsistent values for field {}: {} and {}'.format(f, n_child, o_child))