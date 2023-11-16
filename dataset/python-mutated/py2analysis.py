from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
from renpy.compat import PY2, basestring, bchr, bord, chr, open, pystr, range, round, str, tobytes, unicode
import renpy
from renpy.python import py_compile
import ast
import zlib
from renpy.compat.pickle import loads, dumps
always_constants = {'True', 'False', 'None'}
pure_functions = {'abs', 'all', 'any', 'apply', 'bin', 'bool', 'bytes', 'callable', 'chr', 'cmp', 'dict', 'divmod', 'filter', 'float', 'frozenset', 'getattr', 'globals', 'hasattr', 'hash', 'hex', 'int', 'isinstance', 'len', 'list', 'long', 'map', 'max', 'min', 'oct', 'ord', 'pow', 'range', 'reduce', 'repr', 'round', 'set', 'sorted', 'str', 'sum', 'tuple', 'unichr', 'unicode', 'vars', 'zip', '_', '_p', 'absolute', '__renpy__list__', '__renpy__dict__', '__renpy__set__', 'ImageReference', 'Image', 'Frame', 'Solid', 'LiveComposite', 'LiveCrop', 'LiveTile', 'Flatten', 'Null', 'Window', 'Viewport', 'DynamicDisplayable', 'ConditionSwitch', 'ShowingSwitch', 'Transform', 'Animation', 'Movie', 'Particles', 'SnowBlossom', 'Text', 'ParameterizedText', 'FontGroup', 'Drag', 'Alpha', 'AlphaMask', 'Position', 'Pan', 'Move', 'Motion', 'Revolve', 'Zoom', 'RotoZoom', 'FactorZoom', 'SizeZoom', 'Fade', 'Dissolve', 'ImageDissolve', 'AlphaDissolve', 'CropMove', 'PushMove', 'Pixellate', 'OldMoveTransition', 'MoveTransition', 'MoveFactory', 'MoveIn', 'MoveOut', 'ZoomInOut', 'RevolveInOut', 'MultipleTransition', 'ComposeTransition', 'Pause', 'SubTransition', 'ADVSpeaker', 'ADVCharacter', 'Speaker', 'Character', 'DynamicCharacter', 'Fixed', 'HBox', 'VBox', 'Grid', 'AlphaBlend', 'At', 'color', 'Color', 'ui.returns', 'ui.jumps', 'ui.jumpsoutofcontext', 'ui.callsinnewcontext', 'ui.invokesinnewcontext', 'ui.gamemenus', 'renpy.version_string', 'renpy.version_only', 'renpy.version_tuple', 'renpy.version_name', 'renpy.license'}
constants = {'config', 'style'} | always_constants | pure_functions
not_constants = set()
local_constants = set()

def const(name):
    if False:
        while True:
            i = 10
    '\n    :doc: const\n\n    Declares a variable in the store to be constant.\n\n    A variable is constant if nothing can change its value, or any value\n    reached by indexing it or accessing its attributes. Variables must\n    remain constant out of define, init, and translate python blocks.\n\n    `name`\n        A string giving the name of the variable to declare constant.\n    '
    if name not in not_constants:
        constants.add(name)

def not_const(name):
    if False:
        print('Hello World!')
    '\n    :doc: const\n\n    Declares a name in the store to be not constant.\n\n    This undoes the effect of calls to :func:`renpy.const` and\n    :func:`renpy.pure`.\n\n    `name`\n        The name to declare not constant.\n    '
    constants.discard(name)
    pure_functions.discard(name)
    not_constants.add(name)

def pure(fn):
    if False:
        while True:
            i = 10
    '\n    :doc: const\n\n    Declares a function as pure. A pure function must always return the\n    same value when it is called with the same arguments, outside of\n    define, init, and translate python blocks.\n\n    `fn`\n        The name of the function to declare pure. This may either be a string\n        containing the name of the function, or the function itself.\n        If a string is passed and the function is inside the module,\n        this string should contain the module name with the dot.\n\n    Returns `fn`, allowing this function to be used as a decorator.\n    '
    name = fn
    if not isinstance(name, basestring):
        name = fn.__name__
        module = fn.__module__
        name = module + '.' + name
    if name.startswith('store.'):
        name = name[6:]
    if name not in not_constants:
        pure_functions.add(name)
        constants.add(name)
    return fn

class Control(object):
    """
    Represents control flow.

    `const`
        True if this statement always executes.

    `loop`
        True if this corresponds to a loop.

    `imagemap`
        True if this control is in a non-constant imagemap.
    """

    def __init__(self, const, loop, imagemap):
        if False:
            return 10
        self.const = const
        self.loop = loop
        self.imagemap = imagemap
GLOBAL_CONST = 2
LOCAL_CONST = 1
NOT_CONST = 0

class DeltaSet(object):

    def __init__(self, base, copy=None):
        if False:
            i = 10
            return i + 15
        '\n        Represents a set that stores its contents as differences from a base\n        set.\n        '
        self.base = base
        if copy is not None:
            self.added = set(copy.added)
            self.removed = set(copy.removed)
        else:
            self.added = set()
            self.removed = set()
        self.changed = False

    def add(self, v):
        if False:
            while True:
                i = 10
        if v in self.removed:
            self.removed.discard(v)
            self.changed = True
        elif v not in self.base and v not in self.added:
            self.added.add(v)
            self.changed = True

    def discard(self, v):
        if False:
            print('Hello World!')
        if v in self.added:
            self.added.discard(v)
            self.changed = True
        elif v in self.base and v not in self.removed:
            self.removed.add(v)
            self.changed = True

    def __contains__(self, v):
        if False:
            while True:
                i = 10
        return v in self.added or (v in self.base and v not in self.removed)

    def copy(self):
        if False:
            return 10
        return DeltaSet(self.base, self)

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        for i in self.base:
            if i not in self.removed:
                yield i
        for i in self.added:
            yield i

class Analysis(object):
    """
    Represents the result of code analysis, and provides tools to perform
    code analysis.
    """

    def __init__(self, parent=None):
        if False:
            i = 10
            return i + 15
        self.parent = parent
        self.children = {}
        self.not_constant = DeltaSet(not_constants)
        self.local_constant = DeltaSet(local_constants)
        self.global_constant = DeltaSet(always_constants)
        self.pure_functions = DeltaSet(pure_functions)
        self.control = Control(True, False, False)
        self.control_stack = [self.control]

    def get_child(self, identifier):
        if False:
            while True:
                i = 10
        if identifier in self.children:
            return self.children[identifier]
        rv = Analysis(self)
        self.children[identifier] = rv
        return rv

    def push_control(self, const=True, loop=False, imagemap=False):
        if False:
            for i in range(10):
                print('nop')
        self.control = Control(self.control.const and const, loop, self.control.imagemap or imagemap)
        self.control_stack.append(self.control)

    def pop_control(self):
        if False:
            return 10
        rv = self.control_stack.pop()
        self.control = self.control_stack[-1]
        return rv

    def imagemap(self):
        if False:
            i = 10
            return i + 15
        "\n        Returns NOT_CONST if we're in a non-constant imagemap.\n        "
        if self.control.imagemap:
            return NOT_CONST
        else:
            return GLOBAL_CONST

    def exit_loop(self):
        if False:
            print('Hello World!')
        '\n        Call this to indicate the current loop is being exited by the\n        continue or break statements.\n        '
        l = list(self.control_stack)
        l.reverse()
        for i in l:
            i.const = False
            if i.loop:
                break

    def at_fixed_point(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns True if we've reached a fixed point, where the analysis has\n        not changed since the last time we called this function.\n        "
        for i in self.children.values():
            if not i.at_fixed_point():
                return False
        if self.not_constant.changed or self.global_constant.changed or self.local_constant.changed or self.pure_functions.changed:
            self.not_constant.changed = False
            self.global_constant.changed = False
            self.local_constant.changed = False
            self.pure_functions.changed = False
            return False
        return True

    def mark_constant(self, name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Marks `name` as a potential local constant.\n        '
        if not name in self.not_constant:
            self.local_constant.add(name)
            self.global_constant.discard(name)
            self.pure_functions.discard(name)

    def mark_not_constant(self, name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Marks `name` as definitely not-constant.\n        '
        self.not_constant.add(name)
        self.pure_functions.discard(name)
        self.local_constant.discard(name)
        self.global_constant.discard(name)

    def is_constant(self, node):
        if False:
            return 10
        '\n        Returns true if `node` is constant for the purpose of screen\n        language. Node should be a python AST node.\n\n        Screen language ignores object identity for the purposes of\n        object equality.\n        '

        def check_slice(slice):
            if False:
                print('Hello World!')
            if isinstance(slice, ast.Index):
                return check_node(slice.value)
            elif isinstance(slice, ast.Slice):
                consts = []
                if slice.lower:
                    consts.append(check_node(slice.lower))
                if slice.upper:
                    consts.append(check_node(slice.upper))
                if slice.step:
                    consts.append(check_node(slice.step))
                if not consts:
                    return GLOBAL_CONST
                else:
                    return min(consts)
            return NOT_CONST

        def check_name(node):
            if False:
                i = 10
                return i + 15
            '\n            Check nodes that make up a name. This returns a pair:\n\n            * The first element is True if the node is constant, and False\n              otherwise.\n            * The second element is None if the node is constant or the name is\n              not known, and the name otherwise.\n            '
            if isinstance(node, ast.Name):
                const = NOT_CONST
                name = node.id
            elif isinstance(node, ast.Attribute):
                (const, name) = check_name(node.value)
                if name is not None:
                    name = name + '.' + node.attr
            else:
                return (check_node(node), None)
            if name in self.not_constant:
                return (NOT_CONST, name)
            elif name in self.global_constant:
                return (GLOBAL_CONST, name)
            elif name in self.local_constant:
                return (LOCAL_CONST, name)
            else:
                return (const, name)

        def check_nodes(nodes):
            if False:
                i = 10
                return i + 15
            '\n            Checks a list of nodes for constness.\n            '
            nodes = list(nodes)
            if not nodes:
                return GLOBAL_CONST
            return min((check_node(i) for i in nodes))

        def check_node(node):
            if False:
                while True:
                    i = 10
            '\n            Returns true if the ast node `node` is constant.\n            '
            if node is None:
                return GLOBAL_CONST
            if isinstance(node, (ast.Num, ast.Str)):
                return GLOBAL_CONST
            elif isinstance(node, (ast.List, ast.Tuple)):
                return check_nodes(node.elts)
            elif isinstance(node, (ast.Attribute, ast.Name)):
                return check_name(node)[0]
            elif isinstance(node, ast.BoolOp):
                return check_nodes(node.values)
            elif isinstance(node, ast.BinOp):
                return min(check_node(node.left), check_node(node.right))
            elif isinstance(node, ast.UnaryOp):
                return check_node(node.operand)
            elif isinstance(node, ast.Call):
                (const, name) = check_name(node.func)
                if const != GLOBAL_CONST or name not in self.pure_functions:
                    return NOT_CONST
                consts = []
                consts.append(check_nodes(node.args))
                consts.append(check_nodes((i.value for i in node.keywords)))
                if node.starargs is not None:
                    consts.append(check_node(node.starargs))
                if node.kwargs is not None:
                    consts.append(check_node(node.kwargs))
                return min(consts)
            elif isinstance(node, ast.IfExp):
                return min(check_node(node.test), check_node(node.body), check_node(node.orelse))
            elif isinstance(node, ast.Dict):
                return min(check_nodes(node.keys), check_nodes(node.values))
            elif isinstance(node, ast.Set):
                return check_nodes(node.elts)
            elif isinstance(node, ast.Compare):
                return min(check_node(node.left), check_nodes(node.comparators))
            elif isinstance(node, ast.Repr):
                return check_node(node.value)
            elif isinstance(node, ast.Subscript):
                return min(check_node(node.value), check_slice(node.slice))
            return NOT_CONST
        return check_node(node)

    def is_constant_expr(self, expr):
        if False:
            while True:
                i = 10
        '\n        Compiles `expr` into an AST node, then returns the result of\n        self.is_constant called on that node.\n        '
        (node, literal) = ccache.ast_eval_literal(expr)
        if literal:
            return GLOBAL_CONST
        else:
            return self.is_constant(node)

    def python(self, code):
        if False:
            for i in range(10):
                print('nop')
        '\n        Performs analysis on a block of python code.\n        '
        nodes = ccache.ast_exec(code)
        a = PyAnalysis(self)
        for i in nodes:
            a.visit(i)

    def parameters(self, parameters):
        if False:
            for i in range(10):
                print('nop')
        '\n        Analyzes the parameters to the screen.\n        '
        self.global_constant = DeltaSet(constants)
        for (name, _default) in parameters.parameters:
            self.mark_not_constant(name)
        if parameters.extrapos is not None:
            self.mark_not_constant(parameters.extrapos)
        if parameters.extrakw is not None:
            self.mark_not_constant(parameters.extrakw)

class PyAnalysis(ast.NodeVisitor):
    """
    This analyzes Python code to determine which variables should be
    marked const, and which should be marked non-const.
    """

    def __init__(self, analysis):
        if False:
            return 10
        self.analysis = analysis

    def visit_Name(self, node):
        if False:
            return 10
        if isinstance(node.ctx, ast.AugStore):
            self.analysis.mark_not_constant(node.id)
        elif isinstance(node.ctx, ast.Store):
            if self.analysis.control.const:
                self.analysis.mark_constant(node.id)
            else:
                self.analysis.mark_not_constant(node.id)

    def visit_Assign(self, node):
        if False:
            for i in range(10):
                print('nop')
        const = self.analysis.is_constant(node.value)
        self.analysis.push_control(const, False)
        self.generic_visit(node)
        self.analysis.pop_control()

    def visit_AugAssign(self, node):
        if False:
            print('Hello World!')
        self.analysis.push_control(False, False)
        self.generic_visit(node)
        self.analysis.pop_control()

    def visit_For(self, node):
        if False:
            return 10
        const = self.analysis.is_constant(node.iter)
        self.analysis.push_control(const=const, loop=True)
        old_const = self.analysis.control.const
        self.generic_visit(node)
        if self.analysis.control.const != old_const:
            self.generic_visit(node)
        self.analysis.pop_control()

    def visit_While(self, node):
        if False:
            while True:
                i = 10
        const = self.analysis.is_constant(node.test)
        self.analysis.push_control(const=const, loop=True)
        old_const = self.analysis.control.const
        self.generic_visit(node)
        if self.analysis.control.const != old_const:
            self.generic_visit(node)
        self.analysis.pop_control()

    def visit_If(self, node):
        if False:
            for i in range(10):
                print('nop')
        const = self.analysis.is_constant(node.test)
        self.analysis.push_control(const, False)
        self.generic_visit(node)
        self.analysis.pop_control()

    def visit_Break(self, node):
        if False:
            while True:
                i = 10
        self.analysis.exit_loop()

    def visit_Continue(self, node):
        if False:
            i = 10
            return i + 15
        self.analysis.exit_loop()

class CompilerCache(object):
    """
    Objects of this class are used to cache the compiliation of Python code.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.ast_eval_cache = {}
        self.ast_exec_cache = {}
        self.updated = False
        self.version = 1

    def ast_eval_literal(self, expr):
        if False:
            print('Hello World!')
        '\n        Compiles an expression into an AST.\n        '
        if isinstance(expr, renpy.ast.PyExpr):
            filename = expr.filename
            linenumber = expr.linenumber
        else:
            filename = None
            linenumber = None
        key = (expr, filename, linenumber)
        rv = self.ast_eval_cache.get(key, None)
        if rv is None:
            expr = py_compile(expr, 'eval', ast_node=True)
            try:
                ast.literal_eval(expr)
                literal = True
            except Exception:
                literal = False
            rv = (expr, literal)
            self.ast_eval_cache[key] = rv
            self.updated = True
        new_ccache.ast_eval_cache[key] = rv
        return rv

    def ast_eval(self, expr):
        if False:
            return 10
        return self.ast_eval_literal(expr)[0]

    def ast_exec(self, code):
        if False:
            print('Hello World!')
        '\n        Compiles a block into an AST.\n        '
        if isinstance(code, renpy.ast.PyExpr):
            key = (code, code.filename, code.linenumber)
        else:
            key = (code, None, None)
        rv = self.ast_exec_cache.get(key, None)
        if rv is None:
            rv = py_compile(code, 'exec', ast_node=True)
            self.ast_exec_cache[key] = rv
            self.updated = True
        new_ccache.ast_exec_cache[key] = rv
        return rv
ccache = CompilerCache()
new_ccache = CompilerCache()
CACHE_FILENAME = 'cache/pyanalysis.rpyb'

def load_cache():
    if False:
        while True:
            i = 10
    if renpy.game.args.compile:
        return
    try:
        with renpy.loader.load(CACHE_FILENAME) as f:
            c = loads(zlib.decompress(f.read()))
        if c.version == ccache.version:
            ccache.ast_eval_cache.update(c.ast_eval_cache)
            ccache.ast_exec_cache.update(c.ast_exec_cache)
    except Exception:
        pass

def save_cache():
    if False:
        print('Hello World!')
    if not ccache.updated:
        return
    if renpy.macapp:
        return
    try:
        data = zlib.compress(dumps(new_ccache, True), 3)
        with open(renpy.loader.get_path(CACHE_FILENAME), 'wb') as f:
            f.write(data)
    except Exception:
        pass
__all__ = ['always_constants', 'Analysis', 'ccache', 'CompilerCache', 'const', 'constants', 'Control', 'DeltaSet', 'GLOBAL_CONST', 'load_cache', 'LOCAL_CONST', 'local_constants', 'new_ccache', 'not_const', 'NOT_CONST', 'not_constants', 'pure', 'pure_functions', 'PyAnalysis', 'save_cache']