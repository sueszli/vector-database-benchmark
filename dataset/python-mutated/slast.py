from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
from renpy.compat import PY2, basestring, bchr, bord, chr, open, pystr, range, round, str, tobytes, unicode
from typing import Optional, Any
from renpy.compat.pickle import loads, dumps
import ast
import collections
import linecache
import zlib
import weakref
import renpy
from renpy.display.transform import Transform, ATLTransform
from renpy.display.layout import Fixed
from renpy.display.predict import displayable as predict_displayable
from renpy.python import py_eval_bytecode
from renpy.pyanalysis import Analysis, NOT_CONST, LOCAL_CONST, GLOBAL_CONST, ccache
import hashlib
import time
serial = int(time.time() * 1000000)
use_expression = renpy.object.Sentinel('use_expression')
filename = '<screen language>'
profile_log = renpy.log.open('profile_screen', developer=True, append=False, flush=False)

def compile_expr(loc, node):
    if False:
        return 10
    '\n    Wraps the node in a python AST, and compiles it.\n    '
    filename = loc[0]
    if filename in renpy.python.py3_files:
        flags = renpy.python.py3_compile_flags
    else:
        flags = renpy.python.new_compile_flags
    expr = ast.Expression(body=node)
    renpy.python.fix_locations(expr, 1, 0)
    return compile(expr, filename, 'eval', flags, True)

class SLContext(renpy.ui.Addable):
    """
    A context object that can be passed to the execute methods, and can also
    be placed in renpy.ui.stack.
    """

    def __init__(self, parent=None):
        if False:
            while True:
                i = 10
        if parent is not None:
            self.__dict__.update(parent.__dict__)
            return
        self.scope = {}
        self.root_scope = self.scope
        self.globals = {}
        self.children = []
        self.keywords = {}
        self.style_prefix = None
        self.new_cache = {}
        self.old_cache = {}
        self.miss_cache = {}
        self.use_index = collections.defaultdict(int)
        self.uses_scope = None
        self.widgets = None
        self.debug = False
        self.predicting = False
        self.updating = False
        self.predicted = set()
        self.showif = None
        self.fail = False
        self.parent = None
        self.transclude = None
        self.unlikely = False
        self.new_use_cache = {}
        self.old_use_cache = {}

    def add(self, d, key):
        if False:
            return 10
        self.children.append(d)

    def close(self, d):
        if False:
            i = 10
            return i + 15
        raise Exception('Spurious ui.close().')

class SLNode(object):
    """
    The base class for screen language nodes.
    """
    constant = GLOBAL_CONST
    has_keyword = False
    last_keyword = False

    def __init__(self, loc):
        if False:
            print('Hello World!')
        global serial
        serial += 1
        self.serial = serial
        self.location = loc

    def instantiate(self, transclude):
        if False:
            i = 10
            return i + 15
        '\n        Instantiates a new instance of this class, copying the global\n        attributes of this class onto the new instance.\n        '
        cls = type(self)
        rv = cls.__new__(cls)
        rv.serial = self.serial
        rv.location = self.location
        return rv

    def copy(self, transclude):
        if False:
            for i in range(10):
                print('nop')
        '\n        Makes a copy of this node.\n\n        `transclude`\n            The constness of transclude statements.\n        '
        raise Exception('copy not implemented by ' + type(self).__name__)

    def report_traceback(self, name, last):
        if False:
            for i in range(10):
                print('nop')
        if last:
            return None
        (filename, line) = self.location
        return [(filename, line, name, None)]

    def analyze(self, analysis):
        if False:
            print('Hello World!')
        '\n        Performs static analysis on Python code used in this statement.\n        '

    def prepare(self, analysis):
        if False:
            i = 10
            return i + 15
        '\n        This should be called before the execute code is called, and again\n        after init-level code (like the code in a .rpym module or an init\n        python block) is called.\n\n        `analysis`\n            A pyanalysis.Analysis object containing the analysis of this screen.\n        '

    def execute(self, context):
        if False:
            return 10
        '\n        Execute this node, updating context as appropriate.\n        '
        raise Exception('execute not implemented by ' + type(self).__name__)

    def keywords(self, context):
        if False:
            for i in range(10):
                print('nop')
        '\n        Execute this node, updating context.keywords as appropriate.\n        '
        return

    def copy_on_change(self, cache):
        if False:
            return 10
        '\n        Flags the displayables that are created by this node and its children\n        as copy-on-change.\n        '
        return

    def debug_line(self):
        if False:
            print('Hello World!')
        "\n        Writes information about the line we're on to the debug log.\n        "
        (filename, lineno) = self.location
        full_filename = renpy.exports.unelide_filename(filename)
        line = linecache.getline(full_filename, lineno) or ''
        profile_log.write('  %s:%d %s', filename, lineno, line.rstrip())
        if self.constant:
            profile_log.write('    potentially constant')

    def used_screens(self, callback):
        if False:
            while True:
                i = 10
        '\n        Calls callback with the name of each screen this node and its\n        children use.\n        '
        return

    def has_transclude(self):
        if False:
            return 10
        '\n        Returns true if this node is a transclude or has a transclude as a child.\n        '
        return False

    def has_python(self):
        if False:
            print('Hello World!')
        '\n        Returns true if this node is Python or has a python node as a child.\n        '
        return False

    def dump_const(self, prefix):
        if False:
            return 10
        "\n        Dumps a tree-representation of this node, to help determine what\n        Ren'Py is treating as const and not.\n        "
        raise Exception('dump_const not implemented by ' + type(self).__name__)

    def dc(self, prefix, text, *args):
        if False:
            for i in range(10):
                print('nop')
        '\n        Adds a line of const dump information to the debug log.\n        '
        if self.constant == GLOBAL_CONST:
            const_type = 'global '
        elif self.constant == LOCAL_CONST:
            const_type = 'local  '
        else:
            const_type = 'not    '
        formatted = text.format(*args)
        profile_log.write('%s', '    {}{}{} ({}:{})'.format(const_type, prefix, formatted, self.location[0], self.location[1]))

def analyze_keywords(node, analysis, conditional=GLOBAL_CONST):
    if False:
        while True:
            i = 10
    '\n    Analyzes the keywords that can be applied to this statement,\n    including those provided by if statements.\n    '
    rv = GLOBAL_CONST
    for (_, expr) in node.keyword:
        rv = min(rv, analysis.is_constant_expr(expr), conditional)
    for n in node.children:
        if isinstance(n, SLIf):
            for (cond, block) in n.entries:
                if cond is not None:
                    conditional = min(conditional, analysis.is_constant_expr(cond))
                rv = min(rv, analyze_keywords(block, analysis, conditional))
    return rv
NotGiven = renpy.object.Sentinel('NotGiven')

class SLBlock(SLNode):
    """
    Represents a screen language block that can contain keyword arguments
    and child displayables.
    """
    atl_transform = None
    transform = None

    def __init__(self, loc):
        if False:
            for i in range(10):
                print('nop')
        SLNode.__init__(self, loc)
        self.keyword = []
        self.children = []

    def instantiate(self, transclude):
        if False:
            while True:
                i = 10
        rv = SLNode.instantiate(self, transclude)
        rv.keyword = self.keyword
        rv.children = [i.copy(transclude) for i in self.children]
        rv.atl_transform = self.atl_transform
        return rv

    def copy(self, transclude):
        if False:
            for i in range(10):
                print('nop')
        return self.instantiate(transclude)

    def analyze(self, analysis):
        if False:
            return 10
        for i in self.children:
            i.analyze(analysis)

    def prepare(self, analysis):
        if False:
            return 10
        for i in self.children:
            i.prepare(analysis)
            self.constant = min(self.constant, i.constant)
        keyword_values = {}
        keyword_keys = []
        keyword_exprs = []
        for (k, expr) in self.keyword:
            node = ccache.ast_eval(expr)
            const = analysis.is_constant(node)
            if const == GLOBAL_CONST:
                keyword_values[k] = py_eval_bytecode(compile_expr(self.location, node))
            else:
                keyword_keys.append(ast.Str(s=k))
                keyword_exprs.append(node)
            self.constant = min(self.constant, const)
        if keyword_values:
            self.keyword_values = keyword_values
        else:
            self.keyword_values = None
        if keyword_keys:
            node = ast.Dict(keys=keyword_keys, values=keyword_exprs)
            ast.copy_location(node, keyword_exprs[0])
            self.keyword_exprs = compile_expr(self.location, node)
        else:
            self.keyword_exprs = None
        self.has_keyword = bool(self.keyword)
        self.keyword_children = []
        if self.atl_transform is not None:
            self.has_keyword = True
            self.atl_transform.mark_constant(analysis)
            if self.atl_transform.constant == GLOBAL_CONST:
                self.atl_transform.compile_block()
            const = self.atl_transform.constant
            self.constant = min(self.constant, const)
            self.transform = renpy.display.transform.ATLTransform(self.atl_transform)
            renpy.atl.compile_queue.append(self.transform)
        was_last_keyword = False
        for i in self.children:
            if i.has_keyword:
                if was_last_keyword:
                    raise Exception('Properties are not allowed here.')
                self.keyword_children.append(i)
                self.has_keyword = True
            if i.last_keyword:
                self.last_keyword = True
                was_last_keyword = True
                if not renpy.config.developer:
                    break

    def execute(self, context):
        if False:
            for i in range(10):
                print('nop')
        for i in self.children:
            try:
                i.execute(context)
            except Exception:
                if not context.predicting:
                    raise

    def keywords(self, context):
        if False:
            return 10
        keyword_values = self.keyword_values
        if keyword_values is not None:
            context.keywords.update(keyword_values)
        keyword_exprs = self.keyword_exprs
        if keyword_exprs is not None:
            context.keywords.update(eval(keyword_exprs, context.globals, context.scope))
        for i in self.keyword_children:
            i.keywords(context)
        if self.atl_transform is not None:
            transform = ATLTransform(self.atl_transform, context=context.scope)
            transform.parent_transform = self.transform
            if 'at' in context.keywords:
                try:
                    at_list = list(context.keywords['at'])
                except TypeError:
                    at_list = [context.keywords['at']]
                at_list.append(transform)
                context.keywords['at'] = at_list
            else:
                context.keywords['at'] = transform
        style_prefix = context.keywords.pop('style_prefix', NotGiven)
        if style_prefix is NotGiven:
            style_prefix = context.keywords.pop('style_group', NotGiven)
        if style_prefix is not NotGiven:
            context.style_prefix = style_prefix

    def copy_on_change(self, cache):
        if False:
            i = 10
            return i + 15
        for i in self.children:
            i.copy_on_change(cache)

    def used_screens(self, callback):
        if False:
            return 10
        for i in self.children:
            i.used_screens(callback)

    def has_transclude(self):
        if False:
            return 10
        for i in self.children:
            if i.has_transclude():
                return True
        return False

    def has_python(self):
        if False:
            while True:
                i = 10
        return any((i.has_python() for i in self.children))

    def has_noncondition_child(self):
        if False:
            while True:
                i = 10
        '\n        Returns true if this block has a child that is not an SLIf statement,\n        or false otherwise.\n        '
        worklist = list(self.children)
        while worklist:
            n = worklist.pop(0)
            if type(n) is SLBlock:
                worklist.extend(n.children)
            elif isinstance(n, SLIf):
                for (_, block) in n.entries:
                    worklist.append(block)
            else:
                return True
        return False

    def keyword_exist(self, name):
        if False:
            return 10
        "\n        Returns true if this block or it's SLIf children have parsed `name` keyword,\n        or false otherwise.\n        "
        if name in dict(self.keyword):
            return True
        for n in self.children:
            if isinstance(n, SLIf):
                if n.keyword_exist(name):
                    return True
        return False

    def dump_const(self, prefix):
        if False:
            print('Hello World!')
        self.dc(prefix, 'block')
        for i in self.children:
            i.dump_const(prefix + '  ')
list_or_tuple = (list, tuple)

class SLCache(object):
    """
    The type of cache associated with an SLDisplayable.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.displayable = None
        self.positional = None
        self.keywords = None
        self.children = None
        self.outer_transform = None
        self.inner_transform = None
        self.raw_transform = None
        self.imagemap = None
        self.constant = None
        self.constant_uses_scope = []
        self.constant_widgets = {}
        self.copy_on_change = False
        self.old_showif = None
        self.transclude = None
        self.style_prefix = None
NO_DISPLAYABLE = renpy.display.layout.Null()

class SLDisplayable(SLBlock):
    """
    A screen language AST node that corresponds to a displayable being
    added to the tree.
    """
    hotspot = False
    variable = None
    name = ''
    unique = False
    local_constant = []

    def __init__(self, loc, displayable, scope=False, child_or_fixed=False, style=None, text_style=None, pass_context=False, imagemap=False, replaces=False, default_keywords={}, hotspot=False, variable=None, name='', unique=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        `displayable`\n            A function that, when called with the positional and keyword\n            arguments, causes the displayable to be displayed.\n\n        `scope`\n            If true, the scope is supplied as an argument to the displayable.\n\n        `child_or_fixed`\n            If true and the number of children of this displayable is not one,\n            the children are added to a Fixed, and the Fixed is added to the\n            displayable.\n\n        `style`\n            The base name of the main style.\n\n        `pass_context`\n            If given, the context is passed in as the first positional argument\n            of the displayable.\n\n        `imagemap`\n            True if this is an imagemap, and should be handled as one.\n\n        `hotspot`\n            True if this is a hotspot that depends on the imagemap it was\n            first displayed with.\n\n        `replaces`\n            True if the object this displayable replaces should be\n            passed to it.\n\n        `default_keywords`\n            The default keyword arguments to supply to the displayable.\n\n        `variable`\n            A variable that the main displayable is assigned to.\n\n        `name`\n            The name of the displayable, used for debugging.\n        '
        SLBlock.__init__(self, loc)
        self.displayable = displayable
        self.scope = scope
        self.child_or_fixed = child_or_fixed
        self.style = style
        self.pass_context = pass_context
        self.imagemap = imagemap
        self.hotspot = hotspot
        self.replaces = replaces
        self.default_keywords = default_keywords
        self.variable = variable
        self.unique = unique
        self.positional = []
        self.name = name

    def copy(self, transclude):
        if False:
            while True:
                i = 10
        rv = self.instantiate(transclude)
        rv.displayable = self.displayable
        rv.scope = self.scope
        rv.child_or_fixed = self.child_or_fixed
        rv.style = self.style
        rv.pass_context = self.pass_context
        rv.imagemap = self.imagemap
        rv.hotspot = self.hotspot
        rv.replaces = self.replaces
        rv.default_keywords = self.default_keywords
        rv.variable = self.variable
        rv.positional = self.positional
        rv.name = self.name
        rv.unique = self.unique
        return rv

    def analyze(self, analysis):
        if False:
            i = 10
            return i + 15
        if self.imagemap:
            const = analyze_keywords(self, analysis)
            analysis.push_control(imagemap=const != GLOBAL_CONST)
        if self.hotspot:
            self.constant = min(analysis.imagemap(), self.constant)
        SLBlock.analyze(self, analysis)
        if self.imagemap:
            analysis.pop_control()
        if self.scope:
            self.local_constant = list(analysis.local_constant)
        if self.variable is not None:
            const = self.constant
            for i in self.positional:
                const = min(self.constant, analysis.is_constant_expr(i))
            for (_k, v) in self.keyword:
                const = min(self.constant, analysis.is_constant_expr(v))
            if self.keyword_exist('id'):
                const = NOT_CONST
            if const == LOCAL_CONST:
                analysis.mark_constant(self.variable)
            elif const == NOT_CONST:
                analysis.mark_not_constant(self.variable)

    def prepare(self, analysis):
        if False:
            print('Hello World!')
        SLBlock.prepare(self, analysis)
        exprs = []
        values = []
        has_exprs = False
        has_values = False
        for a in self.positional:
            node = ccache.ast_eval(a)
            const = analysis.is_constant(node)
            if const == GLOBAL_CONST:
                values.append(py_eval_bytecode(compile_expr(self.location, node)))
                exprs.append(ast.Num(n=0))
                has_values = True
            else:
                values.append(use_expression)
                exprs.append(node)
                has_exprs = True
            self.constant = min(self.constant, const)
        if has_values:
            self.positional_values = values
        else:
            self.positional_values = None
        if has_exprs:
            t = ast.Tuple(elts=exprs, ctx=ast.Load())
            ast.copy_location(t, exprs[0])
            self.positional_exprs = compile_expr(self.location, t)
        else:
            self.positional_exprs = None
        self.has_keyword = False
        if self.keyword_exist('id'):
            self.constant = NOT_CONST
        if self.variable is not None:
            self.constant = NOT_CONST

    def keywords(self, context):
        if False:
            while True:
                i = 10
        return

    def execute(self, context):
        if False:
            while True:
                i = 10
        debug = context.debug
        screen = renpy.ui.screen
        cache = context.old_cache.get(self.serial, None) or context.miss_cache.get(self.serial, None)
        if not isinstance(cache, SLCache):
            cache = SLCache()
        context.new_cache[self.serial] = cache
        copy_on_change = cache.copy_on_change
        if debug:
            self.debug_line()
        if cache.constant and cache.style_prefix == context.style_prefix:
            for (i, local_scope, context_scope) in cache.constant_uses_scope:
                if context_scope is None:
                    context_scope = context.root_scope
                if local_scope:
                    scope = dict(context_scope)
                    scope.update(local_scope)
                else:
                    scope = context_scope
                if copy_on_change:
                    if i._scope(scope, False):
                        cache.constant = None
                        break
                else:
                    i._scope(scope, True)
            else:
                d = cache.constant
                if d is not NO_DISPLAYABLE:
                    if context.showif is not None:
                        d = self.wrap_in_showif(d, context, cache)
                    context.children.append(d)
                if context.uses_scope is not None:
                    context.uses_scope.extend(cache.constant_uses_scope)
                if debug:
                    profile_log.write('    reused constant displayable')
                return
        ctx = SLContext(context)
        fail = False
        main = None
        imagemap = False
        reused = False
        try:
            positional_values = self.positional_values
            positional_exprs = self.positional_exprs
            if positional_values and positional_exprs:
                values = eval(positional_exprs, context.globals, context.scope)
                positional = [b if a is use_expression else a for (a, b) in zip(positional_values, values)]
            elif positional_values:
                positional = positional_values
            elif positional_exprs:
                positional = eval(positional_exprs, context.globals, context.scope)
            else:
                positional = []
            keywords = ctx.keywords = self.default_keywords.copy()
            if self.constant:
                ctx.uses_scope = []
            SLBlock.keywords(self, ctx)
            arguments = keywords.pop('arguments', None)
            if arguments:
                positional += arguments
            properties = keywords.pop('properties', None)
            if properties:
                keywords.update(properties)
            widget_id = keywords.pop('id', None)
            transform = keywords.pop('at', None)
            prefer_screen_to_id = keywords.pop('prefer_screen_to_id', False)
            if widget_id and widget_id in screen.widget_properties:
                if prefer_screen_to_id:
                    new_keywords = screen.widget_properties[widget_id].copy()
                    new_keywords.update(keywords)
                    keywords = new_keywords
                else:
                    keywords.update(screen.widget_properties[widget_id])
            style_suffix = keywords.pop('style_suffix', None) or self.style
            if 'style' not in keywords and style_suffix:
                if ctx.style_prefix is None:
                    keywords['style'] = style_suffix
                else:
                    keywords['style'] = ctx.style_prefix + '_' + style_suffix
            old_d = cache.displayable
            if old_d:
                old_main = old_d._main or old_d
            else:
                old_main = None
            if debug:
                self.report_arguments(cache, positional, keywords, transform)
            can_reuse = old_d is not None and positional == cache.positional and (keywords == cache.keywords) and (context.style_prefix == cache.style_prefix)
            if self.variable is not None and copy_on_change:
                can_reuse = False
            if self.hotspot:
                imc = renpy.ui.imagemap_stack[-1]
                if cache.imagemap is not imc:
                    can_reuse = False
                cache.imagemap = imc
            if can_reuse:
                reused = True
                d = old_d
                main = old_main
                if widget_id and (not ctx.unlikely):
                    screen.widgets[widget_id] = main
                    screen.base_widgets[widget_id] = d
                if self.scope and main._uses_scope:
                    if copy_on_change:
                        if main._scope(ctx.scope, False):
                            reused = False
                    else:
                        main._scope(ctx.scope, True)
            if reused and self.imagemap:
                imagemap = True
                cache.imagemap.reuse()
                renpy.ui.imagemap_stack.append(cache.imagemap)
            if not reused:
                cache.positional = positional
                cache.keywords = keywords.copy()
                if self.scope:
                    keywords['scope'] = ctx.scope
                if self.replaces and ctx.updating:
                    keywords['replaces'] = old_main
                if self.pass_context:
                    keywords['context'] = ctx
                d = self.displayable(*positional, **keywords)
                d._unique()
                main = d._main or d
                main._location = self.location
                if widget_id and (not ctx.unlikely):
                    screen.widgets[widget_id] = main
                    screen.base_widgets[widget_id] = d
                imagemap = self.imagemap
                cache.copy_on_change = False
                cache.children = None
            if debug:
                if reused:
                    profile_log.write('    reused displayable')
                elif self.constant:
                    profile_log.write('    created constant displayable')
                else:
                    profile_log.write('    created displayable')
        except Exception:
            if not context.predicting:
                raise
            fail = True
        if self.variable is not None:
            context.scope[self.variable] = main
        ctx.children = []
        ctx.showif = None
        stack = renpy.ui.stack
        stack.append(ctx)
        try:
            for i in self.children:
                try:
                    i.execute(ctx)
                except Exception:
                    if not context.predicting:
                        raise
                    fail = True
        finally:
            ctx.keywords = None
            stack.pop()
            if imagemap:
                cache.imagemap = renpy.ui.imagemap_stack.pop()
                cache.imagemap.cache.finish()
        if fail:
            predict_displayable(main)
            for i in ctx.children:
                predict_displayable(i)
            context.fail = True
            return
        if ctx.children != cache.children:
            if reused and copy_on_change:
                keywords = keywords
                if self.scope:
                    keywords['scope'] = ctx.scope
                if self.replaces and context.updating:
                    keywords['replaces'] = old_main
                if self.pass_context:
                    keywords['context'] = ctx
                d = self.displayable(*positional, **keywords)
                main = d._main or d
                main._location = self.location
                if widget_id:
                    screen.widgets[widget_id] = main
                    screen.base_widgets[widget_id] = d
                cache.copy_on_change = False
                reused = False
            if reused:
                main._clear()
            if self.child_or_fixed and len(ctx.children) != 1:
                f = Fixed()
                for i in ctx.children:
                    f.add(i)
                main.add(f)
            else:
                for i in ctx.children:
                    main.add(i)
        d = d
        old_d = old_d
        if not context.predicting and old_d is not None:
            replaced_by = renpy.display.focus.replaced_by
            replaced_by[id(old_d)] = d
            if d is not main:
                for (old_part, new_part) in zip(old_d._composite_parts, d._composite_parts):
                    replaced_by[id(old_part)] = new_part
        cache.displayable = d
        cache.children = ctx.children
        cache.style_prefix = context.style_prefix
        if not transform:
            transform = None
        if transform is not None and d is not NO_DISPLAYABLE:
            if reused and transform == cache.raw_transform:
                if isinstance(cache.inner_transform, renpy.display.transform.Transform):
                    if cache.inner_transform.child is not d:
                        cache.inner_transform.set_child(d, duplicate=False)
                d = cache.outer_transform
            else:
                old_outer_transform = cache.outer_transform
                cache.raw_transform = transform
                cache.inner_transform = None
                cache.outer_transform = None
                if isinstance(transform, Transform):
                    d = transform(child=d)
                    d._unique()
                    cache.inner_transform = d
                    cache.outer_transform = d
                elif isinstance(transform, list_or_tuple):
                    for t in transform:
                        if isinstance(t, Transform):
                            d = t(child=d)
                            cache.outer_transform = d
                            if cache.inner_transform is None:
                                cache.inner_transform = d
                        else:
                            d = t(d)
                            cache.raw_transform = None
                            cache.outer_transform = None
                            cache.inner_transform = None
                        d._unique()
                else:
                    d = transform(d)
                    d._unique()
                    cache.raw_transform = None
                    cache.outer_transform = None
                    cache.inner_transform = None
                if isinstance(d, Transform):
                    if not context.updating:
                        old_outer_transform = None
                    d.take_state(old_outer_transform)
                    d.take_execution_state(old_outer_transform)
        else:
            cache.inner_transform = None
            cache.outer_transform = None
            cache.raw_transform = None
        if ctx.fail:
            context.fail = True
        elif self.constant:
            cache.constant = d
            if self.scope and main._uses_scope:
                local_scope = {}
                for i in self.local_constant:
                    if i in ctx.scope:
                        local_scope[i] = ctx.scope[i]
                if ctx.scope is context.root_scope:
                    ctx.uses_scope.append((main, local_scope, None))
                else:
                    ctx.uses_scope.append((main, local_scope, ctx.scope))
            cache.constant_uses_scope = ctx.uses_scope
            if context.uses_scope is not None:
                context.uses_scope.extend(ctx.uses_scope)
        if d is not NO_DISPLAYABLE:
            if context.showif is not None:
                d = self.wrap_in_showif(d, context, cache)
            context.children.append(d)

    def wrap_in_showif(self, d, context, cache):
        if False:
            while True:
                i = 10
        '\n        Wraps `d` in a ShowIf displayable.\n        '
        rv = renpy.sl2.sldisplayables.ShowIf(context.showif, cache.old_showif)
        rv.add(d)
        if not context.predicting:
            cache.old_showif = rv
        return rv

    def report_arguments(self, cache, positional, keywords, transform):
        if False:
            return 10
        if positional:
            report = []
            values = self.positional_values or [use_expression] * len(positional)
            for i in range(len(positional)):
                if values[i] is not use_expression:
                    report.append('const')
                elif cache.positional is None:
                    report.append('new')
                elif cache.positional[i] == positional[i]:
                    report.append('equal')
                else:
                    report.append('not-equal')
            profile_log.write('    args: %s', ' '.join(report))
        values = self.keyword_values or {}
        if keywords:
            report = {}
            if cache.keywords is None:
                for k in keywords:
                    if k in values:
                        report[k] = 'const'
                        continue
                    report[k] = 'new'
            else:
                for k in keywords:
                    k = str(k)
                    if k in values:
                        report[k] = 'const'
                        continue
                    if k not in cache.keywords:
                        report[k] = 'new-only'
                        continue
                    if keywords[k] == cache.keywords[k]:
                        report[k] = 'equal'
                    else:
                        report[k] = 'not-equal'
                for k in cache.keywords:
                    if k not in keywords:
                        report[k] = 'old-only'
            profile_log.write('    kwargs: %r', report)
        if transform is not None:
            if 'at' in values:
                profile_log.write('    at: const')
            elif cache.raw_transform is None:
                profile_log.write('    at: new')
            elif cache.raw_transform == transform:
                profile_log.write('    at: equal')
            else:
                profile_log.write('    at: not-equal')

    def copy_on_change(self, cache):
        if False:
            i = 10
            return i + 15
        c = cache.get(self.serial, None)
        if isinstance(c, SLCache):
            c.copy_on_change = True
        for i in self.children:
            i.copy_on_change(cache)

    def dump_const(self, prefix):
        if False:
            for i in range(10):
                print('nop')
        self.dc(prefix, self.name)
        for i in self.children:
            i.dump_const(prefix + '  ')

class SLIf(SLNode):
    """
    A screen language AST node that corresponds to an If/Elif/Else statement.
    """

    def __init__(self, loc):
        if False:
            while True:
                i = 10
        '\n        An AST node that represents an if statement.\n        '
        SLNode.__init__(self, loc)
        self.entries = []

    def copy(self, transclude):
        if False:
            print('Hello World!')
        rv = self.instantiate(transclude)
        rv.entries = [(expr, block.copy(transclude)) for (expr, block) in self.entries]
        return rv

    def analyze(self, analysis):
        if False:
            print('Hello World!')
        const = GLOBAL_CONST
        for (cond, _block) in self.entries:
            if cond is not None:
                const = min(const, analysis.is_constant_expr(cond))
        analysis.push_control(const)
        for (_cond, block) in self.entries:
            block.analyze(analysis)
        analysis.pop_control()

    def prepare(self, analysis):
        if False:
            return 10
        self.prepared_entries = []
        for (cond, block) in self.entries:
            if cond is not None:
                node = ccache.ast_eval(cond)
                cond_const = analysis.is_constant(node)
                self.constant = min(self.constant, cond_const)
                cond = compile_expr(self.location, node)
            else:
                cond_const = True
            block.prepare(analysis)
            self.constant = min(self.constant, block.constant)
            self.prepared_entries.append((cond, block, cond_const))
            self.has_keyword |= block.has_keyword
            self.last_keyword |= block.last_keyword

    def execute(self, context):
        if False:
            i = 10
            return i + 15
        if context.predicting:
            self.execute_predicting(context)
            return
        for (cond, block, _cond_const) in self.prepared_entries:
            if cond is None or eval(cond, context.globals, context.scope):
                for i in block.children:
                    i.execute(context)
                return

    def execute_predicting(self, context):
        if False:
            print('Hello World!')
        first = True
        predict_false = self.serial not in context.predicted
        context.predicted.add(self.serial)
        for (cond, block, const_cond) in self.prepared_entries:
            try:
                cond_value = cond is None or eval(cond, context.globals, context.scope)
            except Exception:
                cond_value = False
            if first and cond_value:
                first = False
                for i in block.children:
                    try:
                        i.execute(context)
                    except Exception:
                        pass
                if const_cond:
                    break
            elif predict_false:
                ctx = SLContext(context)
                ctx.children = []
                ctx.unlikely = True
                for i in block.children:
                    try:
                        i.execute(ctx)
                    except Exception:
                        pass
                for i in ctx.children:
                    predict_displayable(i)

    def keywords(self, context):
        if False:
            i = 10
            return i + 15
        for (cond, block, _cond_const) in self.prepared_entries:
            if cond is None or eval(cond, context.globals, context.scope):
                block.keywords(context)
                return

    def copy_on_change(self, cache):
        if False:
            while True:
                i = 10
        for (_cond, block) in self.entries:
            block.copy_on_change(cache)

    def used_screens(self, callback):
        if False:
            for i in range(10):
                print('nop')
        for (_cond, block) in self.entries:
            block.used_screens(callback)

    def has_transclude(self):
        if False:
            for i in range(10):
                print('nop')
        for (_cond, block) in self.entries:
            if block.has_transclude():
                return True
        return False

    def has_python(self):
        if False:
            while True:
                i = 10
        return any((i[1].has_python() for i in self.entries))

    def keyword_exist(self, name):
        if False:
            return 10
        return any((i[1].keyword_exist(name) for i in self.entries))

    def dump_const(self, prefix):
        if False:
            i = 10
            return i + 15
        first = True
        for (cond, block) in self.entries:
            if first:
                self.dc(prefix, 'if {}', cond)
            else:
                self.dc(prefix, 'elif {}', cond)
            first = False
            for i in block.children:
                i.dump_const(prefix + '  ')

class SLShowIf(SLNode):
    """
    The AST node that corresponds to the showif statement.
    """

    def __init__(self, loc):
        if False:
            while True:
                i = 10
        '\n        An AST node that represents an if statement.\n        '
        SLNode.__init__(self, loc)
        self.entries = []

    def copy(self, transclude):
        if False:
            print('Hello World!')
        rv = self.instantiate(transclude)
        rv.entries = [(expr, block.copy(transclude)) for (expr, block) in self.entries]
        return rv

    def analyze(self, analysis):
        if False:
            return 10
        for (_cond, block) in self.entries:
            block.analyze(analysis)

    def prepare(self, analysis):
        if False:
            i = 10
            return i + 15
        self.prepared_entries = []
        for (cond, block) in self.entries:
            if cond is not None:
                node = ccache.ast_eval(cond)
                self.constant = min(self.constant, analysis.is_constant(node))
                cond = compile_expr(self.location, node)
            block.prepare(analysis)
            self.constant = min(self.constant, block.constant)
            self.prepared_entries.append((cond, block))
        self.last_keyword = True

    def execute(self, context):
        if False:
            while True:
                i = 10
        first_true = context.showif is not False
        for (cond, block) in self.prepared_entries:
            ctx = SLContext(context)
            if not first_true:
                ctx.showif = False
            elif cond is None or eval(cond, context.globals, context.scope):
                ctx.showif = True
                first_true = False
            else:
                ctx.showif = False
            for i in block.children:
                i.execute(ctx)
            if ctx.fail:
                context.fail = True

    def copy_on_change(self, cache):
        if False:
            while True:
                i = 10
        for (_cond, block) in self.entries:
            block.copy_on_change(cache)

    def used_screens(self, callback):
        if False:
            return 10
        for (_cond, block) in self.entries:
            block.used_screens(callback)

    def has_transclude(self):
        if False:
            for i in range(10):
                print('nop')
        for (_cond, block) in self.entries:
            if block.has_transclude():
                return True
        return False

    def has_python(self):
        if False:
            while True:
                i = 10
        return any((i[1].has_python() for i in self.entries))

    def dump_const(self, prefix):
        if False:
            for i in range(10):
                print('nop')
        first = True
        for (cond, block) in self.entries:
            if first:
                self.dc(prefix, 'showif {}', cond)
            else:
                self.dc(prefix, 'else {}', cond)
            first = False
            for i in block.children:
                i.dump_const(prefix + '  ')

class SLFor(SLBlock):
    """
    The AST node that corresponds to a for statement. This only supports
    simple for loops that assign a single variable.
    """
    index_expression = None

    def __init__(self, loc, variable, expression, index_expression):
        if False:
            for i in range(10):
                print('nop')
        SLBlock.__init__(self, loc)
        self.variable = variable
        self.expression = expression
        self.index_expression = index_expression

    def copy(self, transclude):
        if False:
            i = 10
            return i + 15
        rv = self.instantiate(transclude)
        rv.variable = self.variable
        rv.expression = self.expression
        rv.index_expression = self.index_expression
        return rv

    def analyze(self, analysis):
        if False:
            return 10
        const = analysis.is_constant_expr(self.expression) == GLOBAL_CONST
        while True:
            if const:
                analysis.push_control(True, loop=True)
                analysis.mark_constant(self.variable)
            else:
                analysis.push_control(False, loop=True)
                analysis.mark_not_constant(self.variable)
            SLBlock.analyze(self, analysis)
            new_const = analysis.control.const
            analysis.pop_control()
            if new_const == const:
                break
            const = new_const

    def prepare(self, analysis):
        if False:
            return 10
        node = ccache.ast_eval(self.expression)
        const = analysis.is_constant(node)
        if const == GLOBAL_CONST:
            self.expression_value = list(py_eval_bytecode(compile_expr(self.location, node)))
            self.expression_expr = None
        else:
            self.expression_value = None
            self.expression_expr = compile_expr(self.location, node)
        self.constant = min(self.constant, const)
        SLBlock.prepare(self, analysis)
        self.last_keyword = True

    def execute(self, context):
        if False:
            i = 10
            return i + 15
        variable = self.variable
        expr = self.expression_expr
        try:
            if expr is not None:
                value = eval(expr, context.globals, context.scope)
            else:
                value = self.expression_value
        except Exception:
            if not context.predicting:
                raise
            value = [0]
        newcaches = {}
        oldcaches = context.old_cache.get(self.serial, newcaches) or {}
        if not isinstance(oldcaches, dict):
            oldcaches = {}
        misscaches = context.miss_cache.get(self.serial, newcaches) or {}
        if not isinstance(misscaches, dict):
            misscaches = {}
        ctx = SLContext(context)
        for (index, v) in enumerate(value):
            ctx.scope[variable] = v
            children_i = iter(self.children)
            if variable == '_sl2_i':
                sl_python = next(children_i)
                try:
                    sl_python.execute(ctx)
                except Exception:
                    if not context.predicting:
                        raise
            if self.index_expression is not None:
                index = eval(self.index_expression, ctx.globals, ctx.scope)
            ctx.old_cache = oldcaches.get(index, None) or {}
            if not isinstance(ctx.old_cache, dict):
                ctx.old_cache = {}
            ctx.miss_cache = misscaches.get(index, None) or {}
            if not isinstance(ctx.miss_cache, dict):
                ctx.miss_cache = {}
            newcaches[index] = ctx.new_cache = {}
            try:
                for i in children_i:
                    try:
                        i.execute(ctx)
                    except SLForException:
                        raise
                    except Exception:
                        if not context.predicting:
                            raise
            except SLBreakException:
                break
            except SLContinueException:
                continue
            if context.unlikely:
                break
        context.new_cache[self.serial] = newcaches
        if ctx.fail:
            context.fail = True

    def keywords(self, context):
        if False:
            while True:
                i = 10
        return

    def copy_on_change(self, cache):
        if False:
            print('Hello World!')
        c = cache.get(self.serial, None)
        if not isinstance(c, dict):
            return
        for child_cache in c.values():
            for i in self.children:
                i.copy_on_change(child_cache)

    def dump_const(self, prefix):
        if False:
            i = 10
            return i + 15
        self.dc(prefix, 'for {} in {}', self.variable, self.expression)
        for i in self.children:
            i.dump_const(prefix + '  ')

class SLForException(Exception):
    pass

class SLBreakException(SLForException):
    pass

class SLContinueException(SLForException):
    pass

class SLBreak(SLNode):

    def analyze(self, analysis):
        if False:
            return 10
        analysis.exit_loop()

    def execute(self, context):
        if False:
            print('Hello World!')
        raise SLBreakException()

    def copy(self, transclude):
        if False:
            return 10
        rv = self.instantiate(transclude)
        return rv

    def dump_const(self, prefix):
        if False:
            for i in range(10):
                print('nop')
        self.dc(prefix, 'break')

class SLContinue(SLNode):

    def analyze(self, analysis):
        if False:
            i = 10
            return i + 15
        analysis.exit_loop()

    def execute(self, context):
        if False:
            return 10
        raise SLContinueException()

    def copy(self, transclude):
        if False:
            i = 10
            return i + 15
        rv = self.instantiate(transclude)
        return rv

    def dump_const(self, prefix):
        if False:
            print('Hello World!')
        self.dc(prefix, 'continue')

class SLPython(SLNode):

    def __init__(self, loc, code):
        if False:
            return 10
        SLNode.__init__(self, loc)
        self.code = code

    def copy(self, transclude):
        if False:
            return 10
        rv = self.instantiate(transclude)
        rv.code = self.code
        return rv

    def analyze(self, analysis):
        if False:
            print('Hello World!')
        analysis.python(self.code.source)

    def execute(self, context):
        if False:
            while True:
                i = 10
        exec(self.code.bytecode, context.globals, context.scope)

    def prepare(self, analysis):
        if False:
            for i in range(10):
                print('nop')
        self.constant = NOT_CONST
        self.last_keyword = True

    def has_python(self):
        if False:
            print('Hello World!')
        return True

    def dump_const(self, prefix):
        if False:
            i = 10
            return i + 15
        self.dc(prefix, 'python')

class SLPass(SLNode):

    def execute(self, context):
        if False:
            return 10
        return

    def copy(self, transclude):
        if False:
            return 10
        rv = self.instantiate(transclude)
        return rv

    def dump_const(self, prefix):
        if False:
            return 10
        self.dc(prefix, 'pass')

class SLDefault(SLNode):

    def __init__(self, loc, variable, expression):
        if False:
            for i in range(10):
                print('nop')
        SLNode.__init__(self, loc)
        self.variable = variable
        self.expression = expression

    def copy(self, transclude):
        if False:
            for i in range(10):
                print('nop')
        rv = self.instantiate(transclude)
        rv.variable = self.variable
        rv.expression = self.expression
        return rv

    def analyze(self, analysis):
        if False:
            return 10
        analysis.mark_not_constant(self.variable)

    def prepare(self, analysis):
        if False:
            return 10
        self.expr = compile_expr(self.location, ccache.ast_eval(self.expression))
        self.constant = NOT_CONST
        self.last_keyword = True

    def execute(self, context):
        if False:
            return 10
        scope = context.scope
        variable = self.variable
        if variable in scope:
            return
        scope[variable] = eval(self.expr, context.globals, scope)

    def has_python(self):
        if False:
            return 10
        return True

    def dump_const(self, prefix):
        if False:
            print('Hello World!')
        self.dc(prefix, 'default {} = {}', self.variable, self.expression)

class SLUse(SLNode):
    id = None
    block = None

    def __init__(self, loc, target, args, id_expr, block):
        if False:
            print('Hello World!')
        SLNode.__init__(self, loc)
        self.target = target
        self.ast = None
        self.args = args
        self.id = id_expr
        self.block = block

    def copy(self, transclude):
        if False:
            while True:
                i = 10
        rv = self.instantiate(transclude)
        rv.target = self.target
        rv.args = self.args
        rv.id = self.id
        if self.block is not None:
            rv.block = self.block.copy(transclude)
        else:
            rv.block = None
        rv.ast = None
        return rv

    def analyze(self, analysis):
        if False:
            return 10
        self.last_keyword = True
        if self.id:
            self.constant = NOT_CONST
        if self.block:
            self.block.analyze(analysis)

    def prepare(self, analysis):
        if False:
            while True:
                i = 10
        self.ast = None
        if self.block:
            self.block.prepare(analysis)
            if self.block.constant == GLOBAL_CONST:
                const = True
            else:
                const = False
        else:
            const = True
        if isinstance(self.target, renpy.ast.PyExpr):
            self.constant = NOT_CONST
            const = False
            self.ast = None
        else:
            target = renpy.display.screen.get_screen_variant(self.target)
            if target is None:
                self.constant = NOT_CONST
                if renpy.config.developer:
                    raise Exception('A screen named {} does not exist.'.format(self.target))
                else:
                    return
            if target.ast is None:
                self.constant = NOT_CONST
                return
            if const:
                self.ast = target.ast.const_ast
            else:
                self.ast = target.ast.not_const_ast
            self.constant = min(self.constant, self.ast.constant)

    def execute_use_screen(self, context):
        if False:
            for i in range(10):
                print('nop')
        serial = context.use_index[self.serial]
        context.use_index[self.serial] = serial + 1
        name = (context.scope.get('_name', ()), self.serial, serial)
        if self.args:
            (args, kwargs) = self.args.evaluate(context.scope)
        else:
            args = []
            kwargs = {}
        renpy.display.screen.use_screen(self.target, *args, _name=name, _scope=context.scope, **kwargs)

    def execute(self, context):
        if False:
            i = 10
            return i + 15
        if isinstance(self.target, renpy.ast.PyExpr):
            target_name = eval(self.target, context.globals, context.scope)
            target = renpy.display.screen.get_screen_variant(target_name)
            if target is None:
                raise Exception('A screen named {} does not exist.'.format(target_name))
            ast = target.ast.not_const_ast
            id_prefix = '_use_expression'
        else:
            id_prefix = self.target
            ast = self.ast
        if ast is None:
            self.execute_use_screen(context)
            return
        ctx = SLContext(context)
        ctx.new_cache = context.new_cache[self.serial] = {'ast': ast}
        ctx.miss_cache = context.miss_cache.get(self.serial, None) or {}
        if self.id:
            use_id = (id_prefix, eval(self.id, context.globals, context.scope))
            ctx.old_cache = context.old_use_cache.get(use_id, None) or context.old_cache.get(self.serial, None) or {}
            ctx.new_use_cache[use_id] = ctx.new_cache
        else:
            ctx.old_cache = context.old_cache.get(self.serial, None) or {}
        if not isinstance(ctx.old_cache, dict):
            ctx.old_cache = {}
        if not isinstance(ctx.miss_cache, dict):
            ctx.miss_cache = {}
        try:
            if self.args:
                (args, kwargs) = self.args.evaluate(context.scope)
            else:
                args = []
                kwargs = {}
        except Exception:
            if not context.predicting:
                raise
            args = []
            kwargs = {}
        scope = ctx.old_cache.get('scope', None) or ctx.miss_cache.get('scope', None) or {}
        if not ctx.updating:
            scope.clear()
        if ast.parameters is not None:
            new_scope = ast.parameters.apply(args, kwargs, ignore_errors=context.predicting)
            scope.update(new_scope)
        else:
            if args:
                raise Exception('Screen {} does not take positional arguments. ({} given)'.format(self.target, len(args)))
            scope.clear()
            scope.update(context.scope)
            scope.update(kwargs)
        scope['_scope'] = scope
        ctx.new_cache['scope'] = scope
        ctx.scope = scope
        ctx.parent = weakref.ref(context)
        ctx.transclude = self.block
        try:
            ast.execute(ctx)
        finally:
            del scope['_scope']
        if ctx.fail:
            context.fail = True

    def copy_on_change(self, cache):
        if False:
            return 10
        c = cache.get(self.serial, None)
        if c is None:
            return
        ast = c.get('ast', None)
        if ast is not None:
            ast.copy_on_change(c)

    def used_screens(self, callback):
        if False:
            return 10
        if not isinstance(self.target, renpy.ast.PyExpr):
            callback(self.target)
        if self.block is not None:
            self.block.used_screens(callback)

    def has_transclude(self):
        if False:
            i = 10
            return i + 15
        if self.block:
            return self.block.has_transclude()
        else:
            return False

    def dump_const(self, prefix):
        if False:
            return 10
        self.dc(prefix, 'use', self.target)
        for i in self.ast.children:
            if self.block:
                i.dump_const(prefix + '│ ')
            else:
                i.dump_const(prefix + '  ')
        if self.block:
            self.dc(prefix, '└ (transclude block)')
            for i in self.block.children:
                i.dump_const(prefix + '  ')

class SLTransclude(SLNode):

    def __init__(self, loc):
        if False:
            i = 10
            return i + 15
        SLNode.__init__(self, loc)

    def copy(self, transclude):
        if False:
            while True:
                i = 10
        rv = self.instantiate(transclude)
        rv.constant = transclude
        return rv

    def execute(self, context):
        if False:
            while True:
                i = 10
        if not context.transclude:
            return
        parent = context.parent
        if parent is not None:
            parent = parent()
        ctx = SLContext(parent)
        ctx.new_cache = context.new_cache[self.serial] = {}
        ctx.old_cache = context.old_cache.get(self.serial, None) or {}
        ctx.miss_cache = context.miss_cache.get(self.serial, None) or {}
        ctx.uses_scope = context.uses_scope
        if not isinstance(ctx.old_cache, dict):
            ctx.old_cache = {}
        if not isinstance(ctx.miss_cache, dict):
            ctx.miss_cache = {}
        ctx.new_cache['transclude'] = context.transclude
        ctx.children = context.children
        ctx.showif = context.showif
        try:
            renpy.ui.stack.append(ctx)
            context.transclude.keywords(ctx)
            context.transclude.execute(ctx)
        finally:
            renpy.ui.stack.pop()
        if ctx.fail:
            context.fail = True

    def copy_on_change(self, cache):
        if False:
            return 10
        c = cache.get(self.serial, None)
        if c is None or 'transclude' not in c:
            return
        SLBlock.copy_on_change(c['transclude'], c)

    def has_transclude(self):
        if False:
            i = 10
            return i + 15
        return True

    def dump_const(self, prefix):
        if False:
            i = 10
            return i + 15
        self.dc(prefix, 'transclude')

class SLCustomUse(SLNode):
    """This represents special use screen statement defined
    by renpy.register_sl_statement.
    """

    def __init__(self, loc, target, positional, block):
        if False:
            for i in range(10):
                print('nop')
        SLNode.__init__(self, loc)
        self.target = target
        self.ast = None
        self.positional = positional
        self.block = block

    def copy(self, transclude):
        if False:
            for i in range(10):
                print('nop')
        rv = self.instantiate(transclude)
        rv.target = self.target
        rv.ast = None
        rv.positional = self.positional
        rv.block = self.block.copy(transclude)
        return rv

    def analyze(self, analysis):
        if False:
            return 10
        self.last_keyword = True
        self.block.analyze(analysis)

    def prepare(self, analysis):
        if False:
            while True:
                i = 10
        block = self.block
        block.prepare(analysis)
        target = renpy.display.screen.get_screen_variant(self.target)
        if target is None:
            self.constant = NOT_CONST
            if renpy.config.developer:
                raise Exception('A screen named {} does not exist.'.format(self.target))
            else:
                return
        if target.ast is None:
            self.constant = NOT_CONST
            if renpy.config.developer:
                raise Exception('A screen used in CD SLS should be a SL-based screen.')
            else:
                return
        if block.keyword_exist('id'):
            self.constant = NOT_CONST
            self.ast = target.ast.not_const_ast
        elif block.constant == GLOBAL_CONST:
            self.ast = target.ast.const_ast
        else:
            self.ast = target.ast.not_const_ast
        self.constant = min(self.constant, self.ast.constant)

    def execute(self, context):
        if False:
            i = 10
            return i + 15
        ctx = SLContext(context)
        ctx.new_cache = context.new_cache[self.serial] = {}
        ctx.miss_cache = context.miss_cache.get(self.serial, None) or {}
        try:
            args = [eval(i, context.globals, context.scope) for i in self.positional]
            kwargs = ctx.keywords = {}
            self.block.keywords(ctx)
            arguments = kwargs.pop('arguments', None)
            if arguments:
                args += arguments
            properties = kwargs.pop('properties', None)
            if properties:
                kwargs.update(properties)
            style_suffix = kwargs.pop('style_suffix', None)
            if 'style' not in kwargs and style_suffix:
                if ctx.style_prefix is None:
                    kwargs['style'] = style_suffix
                else:
                    kwargs['style'] = ctx.style_prefix + '_' + style_suffix
        except Exception:
            if not context.predicting:
                raise
            args = []
            kwargs = {}
        id = kwargs.pop('id', None)
        if id is not None:
            use_id = (self.target, id)
            ctx.old_cache = context.old_use_cache.get(use_id, None) or context.old_cache.get(self.serial, None) or {}
            ctx.new_use_cache[use_id] = ctx.new_cache
        else:
            ctx.old_cache = context.old_cache.get(self.serial, None) or {}
        if not isinstance(ctx.old_cache, dict):
            ctx.old_cache = {}
        if not isinstance(ctx.miss_cache, dict):
            ctx.miss_cache = {}
        ast = self.ast
        scope = ctx.old_cache.get('scope', None) or ctx.miss_cache.get('scope', None) or {}
        if not ctx.updating:
            scope.clear()
        if ast.parameters is not None:
            new_scope = ast.parameters.apply(args, kwargs, ignore_errors=context.predicting)
            scope.update(new_scope)
        else:
            if args:
                raise Exception('Screen {} does not take positional arguments. ({} given)'.format(self.target, len(args)))
            scope.clear()
            scope.update(context.scope)
            scope.update(kwargs)
        scope['_scope'] = scope
        ctx.new_cache['scope'] = scope
        ctx.scope = scope
        ctx.parent = weakref.ref(context)
        if self.block.children:
            ctx.transclude = self.block
        else:
            ctx.transclude = None
        try:
            ast.execute(ctx)
        finally:
            del scope['_scope']
        if ctx.fail:
            context.fail = True

    def copy_on_change(self, cache):
        if False:
            for i in range(10):
                print('nop')
        c = cache.get(self.serial, None)
        if c is None:
            return
        self.ast.copy_on_change(c)

    def used_screens(self, callback):
        if False:
            while True:
                i = 10
        callback(self.target)
        if self.block is not None:
            self.block.used_screens(callback)

    def has_transclude(self):
        if False:
            while True:
                i = 10
        return self.block.has_transclude()

    def dump_const(self, prefix):
        if False:
            return 10
        self.dc(prefix, 'custom-use', self.target)
        for i in self.ast.children:
            if self.block:
                i.dump_const(prefix + '│ ')
            else:
                i.dump_const(prefix + '  ')
        if self.block:
            self.dc('prefix', '└ (transclude block)')
            for i in self.block.children:
                i.dump_const(prefix + '  ')

class SLScreen(SLBlock):
    """
    This represents a screen defined in the screen language 2.
    """
    version = 0
    const_ast = None
    not_const_ast = None
    analysis = None
    layer = "'screens'"
    sensitive = 'True'
    roll_forward = 'None'

    def __init__(self, loc):
        if False:
            return 10
        SLBlock.__init__(self, loc)
        self.name = None
        self.modal = 'False'
        self.zorder = '0'
        self.tag = None
        self.variant = 'None'
        self.predict = 'None'
        self.sensitive = 'True'
        self.parameters = None
        self.analysis = None
        self.prepared = False

    def copy(self, transclude):
        if False:
            for i in range(10):
                print('nop')
        rv = self.instantiate(transclude)
        rv.name = self.name
        rv.modal = self.modal
        rv.zorder = self.zorder
        rv.tag = self.tag
        rv.variant = self.variant
        rv.predict = self.predict
        rv.parameters = self.parameters
        rv.sensitive = self.sensitive
        rv.prepared = False
        rv.analysis = None
        return rv

    def define(self, location):
        if False:
            print('Hello World!')
        '\n        Defines a screen.\n        '
        renpy.display.screen.define_screen(self.name, self, modal=self.modal, zorder=self.zorder, tag=self.tag, variant=renpy.python.py_eval(self.variant), predict=renpy.python.py_eval(self.predict), parameters=self.parameters, location=self.location, layer=renpy.python.py_eval(self.layer), sensitive=self.sensitive, roll_forward=renpy.python.py_eval(self.roll_forward))

    def analyze(self, analysis):
        if False:
            for i in range(10):
                print('nop')
        SLBlock.analyze(self, analysis)

    def analyze_screen(self):
        if False:
            i = 10
            return i + 15
        if self.const_ast:
            return
        key = (self.name, self.variant, self.location)
        if key in scache.const_analyzed:
            self.const_ast = scache.const_analyzed[key]
            self.not_const_ast = scache.not_const_analyzed[key]
            return
        self.const_ast = self
        if self.has_transclude():
            self.not_const_ast = self.copy(NOT_CONST)
            self.not_const_ast.const_ast = self.not_const_ast
            targets = [self.const_ast, self.not_const_ast]
        else:
            self.not_const_ast = self.const_ast
            targets = [self.const_ast]
        for ast in targets:
            analysis = ast.analysis = Analysis(None)
            if ast.parameters:
                analysis.parameters(ast.parameters)
            ast.analyze(analysis)
            while not analysis.at_fixed_point():
                ast.analyze(analysis)
        scache.const_analyzed[key] = self.const_ast
        scache.not_const_analyzed[key] = self.not_const_ast
        scache.updated = True

    def unprepare_screen(self):
        if False:
            print('Hello World!')
        self.prepared = False

    def prepare_screen(self):
        if False:
            i = 10
            return i + 15
        if self.prepared:
            return
        self.analyze_screen()
        self.version += 1
        self.const_ast.prepare(self.const_ast.analysis)
        if self.not_const_ast is not self.const_ast:
            self.not_const_ast.prepare(self.not_const_ast.analysis)
        self.prepared = True
        if renpy.display.screen.get_profile(self.name).const:
            profile_log.write('CONST ANALYSIS %s', self.name)
            new_constants = [i for i in self.const_ast.analysis.global_constant if i not in renpy.pyanalysis.constants]
            new_constants.sort()
            profile_log.write('    global_const: %s', ' '.join(new_constants))
            local_constants = list(self.const_ast.analysis.local_constant)
            local_constants.sort()
            profile_log.write('    local_const: %s', ' '.join(local_constants))
            not_constants = list(self.const_ast.analysis.not_constant)
            not_constants.sort()
            profile_log.write('    not_const: %s', ' '.join(not_constants))
            profile_log.write('')
            self.const_ast.dump_const('')
            profile_log.write('')

    def execute(self, context):
        if False:
            return 10
        self.const_ast.keywords(context)
        SLBlock.execute(self.const_ast, context)

    def report_traceback(self, name, last):
        if False:
            for i in range(10):
                print('nop')
        if last:
            return None
        if name == '__call__':
            return []
        return SLBlock.report_traceback(self, name, last)

    def copy_on_change(self, cache):
        if False:
            while True:
                i = 10
        SLBlock.copy_on_change(self.const_ast, cache)

    def __call__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        scope = kwargs['_scope']
        debug = kwargs.get('_debug', False)
        if self.parameters:
            args = scope.get('_args', ())
            kwargs = scope.get('_kwargs', {})
            values = renpy.ast.apply_arguments(self.parameters, args, kwargs, ignore_errors=renpy.display.predict.predicting)
            scope.update(values)
        if not self.prepared:
            self.prepare_screen()
        current_screen = renpy.display.screen.current_screen()
        if current_screen.screen_name[0] in renpy.config.profile_screens:
            debug = True
        context = SLContext()
        context.scope = scope
        context.root_scope = scope
        context.globals = renpy.python.store_dicts['store']
        context.debug = debug
        context.predicting = renpy.display.predict.predicting
        context.updating = current_screen.phase == renpy.display.screen.UPDATE
        name = scope['_name']

        def get_cache(d):
            if False:
                for i in range(10):
                    print('nop')
            rv = d.get(name, None)
            if not isinstance(rv, dict) or rv.get('version', None) != self.version:
                rv = {'version': self.version}
                d[name] = rv
            return rv
        context.old_cache = get_cache(current_screen.cache)
        context.miss_cache = get_cache(current_screen.miss_cache)
        context.new_cache = {'version': self.version}
        context.old_use_cache = current_screen.use_cache
        context.new_use_cache = {}
        self.execute(context)
        for i in context.children:
            renpy.ui.implicit_add(i)
        current_screen.cache[name] = context.new_cache
        current_screen.use_cache = context.new_use_cache

class ScreenCache(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.version = 1
        self.const_analyzed = {}
        self.not_const_analyzed = {}
        self.updated = False
scache = ScreenCache()
CACHE_FILENAME = 'cache/screens.rpyb'

def load_cache():
    if False:
        i = 10
        return i + 15
    if renpy.game.args.compile:
        return
    try:
        with renpy.loader.load(CACHE_FILENAME) as f:
            digest = f.read(hashlib.md5().digest_size)
            if digest != renpy.game.script.digest.digest():
                return
            s = loads(zlib.decompress(f.read()))
        if s.version == scache.version:
            renpy.game.script.update_bytecode()
            scache.const_analyzed.update(s.const_analyzed)
            scache.not_const_analyzed.update(s.not_const_analyzed)
    except Exception:
        pass

def save_cache():
    if False:
        while True:
            i = 10
    if not scache.updated:
        return
    if renpy.macapp:
        return
    try:
        data = zlib.compress(dumps(scache), 3)
        with open(renpy.loader.get_path(CACHE_FILENAME), 'wb') as f:
            f.write(renpy.game.script.digest.digest())
            f.write(data)
    except Exception:
        pass