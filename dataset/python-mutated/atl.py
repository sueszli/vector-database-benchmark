from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
from renpy.compat import PY2, basestring, bchr, bord, chr, open, pystr, range, round, str, tobytes, unicode
import random
import renpy
from renpy.pyanalysis import Analysis, NOT_CONST, GLOBAL_CONST

def compiling(loc):
    if False:
        i = 10
        return i + 15
    (file, number) = loc
    renpy.game.exception_info = 'Compiling ATL code at %s:%d' % (file, number)

def executing(loc):
    if False:
        for i in range(10):
            print('nop')
    (file, number) = loc
    renpy.game.exception_info = 'Executing ATL code at %s:%d' % (file, number)
warpers = {}

def atl_warper(f):
    if False:
        return 10
    name = f.__name__
    warpers[name] = f
    return f

@atl_warper
def pause(t):
    if False:
        print('Hello World!')
    if t < 1.0:
        return 0.0
    else:
        return 1.0

@atl_warper
def instant(t):
    if False:
        while True:
            i = 10
    return 1.0
position = renpy.object.Sentinel('position')

def any_object(x):
    if False:
        i = 10
        return i + 15
    return x

def bool_or_none(x):
    if False:
        return 10
    if x is None:
        return x
    return bool(x)

def float_or_none(x):
    if False:
        for i in range(10):
            print('nop')
    if x is None:
        return x
    return float(x)

def matrix(x):
    if False:
        i = 10
        return i + 15
    if x is None:
        return None
    elif callable(x):
        return x
    else:
        return renpy.display.matrix.Matrix(x)

def mesh(x):
    if False:
        return 10
    if isinstance(x, (renpy.gl2.gl2mesh2.Mesh2, renpy.gl2.gl2mesh3.Mesh3, tuple)):
        return x
    return bool(x)
PROPERTIES = {}

def correct_type(v, b, ty):
    if False:
        return 10
    '\n    Corrects the type of v to match ty. b is used to inform the match.\n    '
    if ty is position:
        if v is None:
            return None
        else:
            return type(b)(v)
    else:
        return ty(v)

def interpolate(t, a, b, type):
    if False:
        return 10
    '\n    Linearly interpolate the arguments.\n    '
    if b is None or isinstance(b, (bool, basestring, renpy.display.matrix.Matrix, renpy.display.transform.Camera)):
        if t >= 1.0:
            return b
        else:
            return a
    elif isinstance(b, tuple):
        if a is None:
            a = [None] * len(b)
        if not isinstance(type, tuple):
            type = (type,) * len(b)
        return tuple((interpolate(t, i, j, ty) for (i, j, ty) in zip(a, b, type)))
    elif callable(b):
        a_origin = getattr(a, 'origin', None)
        rv = b(a_origin, t)
        rv.origin = b
        return rv
    else:
        if a is None:
            a = 0
        return correct_type(a + t * (b - a), b, type)

def interpolate_spline(t, spline):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(spline[-1], tuple):
        return tuple((interpolate_spline(t, i) for i in zip(*spline)))
    if spline[0] is None:
        return spline[-1]
    if len(spline) == 2:
        t_p = 1.0 - t
        rv = t_p * spline[0] + t * spline[-1]
    elif len(spline) == 3:
        t_pp = (1.0 - t) ** 2
        t_p = 2 * t * (1.0 - t)
        t2 = t ** 2
        rv = t_pp * spline[0] + t_p * spline[1] + t2 * spline[2]
    elif len(spline) == 4:
        t_ppp = (1.0 - t) ** 3
        t_pp = 3 * t * (1.0 - t) ** 2
        t_p = 3 * t ** 2 * (1.0 - t)
        t3 = t ** 3
        rv = t_ppp * spline[0] + t_pp * spline[1] + t_p * spline[2] + t3 * spline[3]
    elif t <= 0.0 or t >= 1.0:
        rv = spline[0 if t <= 0.0 else -1]
    else:
        spline = [spline[1], spline[0]] + list(spline[2:-2]) + [spline[-1], spline[-2]]
        inner_spline_count = float(len(spline) - 3)
        sector = int(t // (1.0 / inner_spline_count) + 1)
        t = t % (1.0 / inner_spline_count) * inner_spline_count
        rv = get_catmull_rom_value(t, *spline[sector - 1:sector + 3])
    return correct_type(rv, spline[-1], position)

def get_catmull_rom_value(t, p_1, p0, p1, p2):
    if False:
        print('Hello World!')
    '\n    Very basic Catmull-Rom calculation with no alpha or handling\n    of multi-dimensional points\n    '
    t = float(max(0.0, min(1.0, t)))
    return type(p0)((t * ((2 - t) * t - 1) * p_1 + (t * t * (3 * t - 5) + 2) * p0 + t * ((4 - 3 * t) * t + 1) * p1 + (t - 1) * t * t * p2) / 2)
compile_queue = []

def compile_all():
    if False:
        i = 10
        return i + 15
    '\n    Called after the init phase is finished and transforms are compiled,\n    to compile all constant transforms.\n    '
    global compile_queue
    for i in compile_queue:
        if i.atl.constant == GLOBAL_CONST:
            i.compile()
    compile_queue = []
NotInContext = renpy.object.Sentinel('NotInContext')

class Context(object):

    def __init__(self, context):
        if False:
            return 10
        self.context = context

    def eval(self, expr):
        if False:
            print('Hello World!')
        return renpy.python.py_eval(expr, locals=self.context)

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, Context):
            return False
        return self.context == other.context

    def __ne__(self, other):
        if False:
            i = 10
            return i + 15
        return not self == other

class ATLTransformBase(renpy.object.Object):
    parameters = renpy.ast.EMPTY_PARAMETERS
    parent_transform = None
    atl_st_offset = 0
    predict_block = None
    nosave = ['parent_transform']

    def __init__(self, atl, context, parameters):
        if False:
            print('Hello World!')
        if parameters is None:
            parameters = ATLTransformBase.parameters
        else:
            context = context.copy()
            for (k, v) in parameters.parameters:
                if v is not None:
                    context[k] = renpy.python.py_eval(v, locals=context)
        self.parameters = parameters
        self.atl = atl
        self.context = Context(context)
        self.block = None
        self.predict_block = None
        self.properties = None
        self.atl_state = None
        self.done = False
        self.transform_event = None
        self.last_transform_event = None
        self.last_child_transform_event = None
        self.raw_child = None
        self.parent_transform = None
        if renpy.config.atl_start_on_show:
            self.atl_st_offset = None
        else:
            self.atl_st_offset = 0
        if renpy.game.context().init_phase:
            compile_queue.append(self)

    @property
    def transition(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns true if this is likely to be an ATL transition.\n        '
        return 'new_widget' in self.context.context

    def _handles_event(self, event):
        if False:
            i = 10
            return i + 15
        if event == 'replaced' and self.atl_state is None:
            return True
        if self.block is not None and self.block._handles_event(event):
            return True
        if self.child is None:
            return False
        return self.child._handles_event(event)

    def get_block(self):
        if False:
            return 10
        '\n        Returns the compiled block to use.\n        '
        if self.block:
            return self.block
        elif self.predict_block and renpy.display.predict.predicting:
            return self.predict_block
        else:
            return None

    def take_execution_state(self, t):
        if False:
            while True:
                i = 10
        '\n        Updates self to begin executing from the same point as t. This\n        requires that t.atl is self.atl.\n        '
        super(ATLTransformBase, self).take_execution_state(t)
        self.atl_st_offset = None
        self.atl_state = None
        if self is t:
            return
        elif not isinstance(t, ATLTransformBase):
            return
        elif t.atl is not self.atl:
            return
        if t.atl.constant != GLOBAL_CONST:
            block = self.get_block()
            if block is None:
                block = self.compile()
            if not deep_compare(self.block, t.block):
                return
        self.done = t.done
        self.block = t.block
        self.atl_state = t.atl_state
        self.transform_event = t.transform_event
        self.last_transform_event = t.last_transform_event
        self.last_child_transform_event = t.last_child_transform_event
        self.st = t.st
        self.at = t.at
        self.st_offset = t.st_offset
        self.at_offset = t.at_offset
        self.atl_st_offset = t.atl_st_offset
        if self.child is renpy.display.motion.null:
            if t.child and t.child._duplicatable:
                self.child = t.child._duplicate(None)
            else:
                self.child = t.child
            self.raw_child = t.raw_child

    def __call__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        _args = kwargs.pop('_args', None)
        context = self.context.context.copy()
        positional = list(self.parameters.positional)
        args = list(args)
        child = None
        if not positional and args:
            child = args.pop(0)
        while positional and args:
            name = positional.pop(0)
            value = args.pop(0)
            if name in kwargs:
                raise Exception('Parameter %r is used as both a positional and keyword argument to a transition.' % name)
            if name == 'child' or name == 'old_widget':
                child = value
            context[name] = value
        if args:
            raise Exception('Too many arguments passed to ATL transform.')
        for (k, v) in kwargs.items():
            if k == 'old_widget':
                child = v
            if k in positional:
                positional.remove(k)
                context[k] = v
            elif k in context:
                context[k] = v
            elif k == 'child':
                child = v
            else:
                raise Exception('Parameter %r is not known by ATL Transform.' % k)
        if child is None:
            child = self.child
        if getattr(child, '_duplicatable', False):
            child = child._duplicate(_args)
        parameters = renpy.ast.ParameterInfo([], positional, None, None)
        rv = renpy.display.motion.ATLTransform(atl=self.atl, child=child, style=self.style_arg, context=context, parameters=parameters, _args=_args)
        rv.parent_transform = self
        rv.take_state(self)
        return rv

    def compile(self):
        if False:
            while True:
                i = 10
        '\n        Compiles the ATL code into a block. As necessary, updates the\n        properties.\n        '
        constant = self.atl.constant == GLOBAL_CONST
        if not constant:
            for p in self.parameters.positional:
                if p not in self.context.context:
                    raise Exception("Cannot compile ATL Transform at %s:%d, as it's missing positional parameter %s." % (self.atl.loc[0], self.atl.loc[1], p))
        if constant and self.parent_transform:
            if self.parent_transform.block:
                self.block = self.parent_transform.block
                self.properties = self.parent_transform.properties
                self.parent_transform = None
                return self.block
        old_exception_info = renpy.game.exception_info
        if constant and self.atl.compiled_block is not None:
            block = self.atl.compiled_block
        else:
            block = self.atl.compile(self.context)
        if all((isinstance(statement, Interpolation) and statement.duration == 0 for statement in block.statements)):
            self.properties = []
            for interp in block.statements:
                self.properties.extend(interp.properties)
        if not constant and renpy.display.predict.predicting:
            self.predict_block = block
        else:
            self.block = block
            self.predict_block = None
        renpy.game.exception_info = old_exception_info
        if constant and self.parent_transform:
            self.parent_transform.block = self.block
            self.parent_transform.properties = self.properties
            self.parent_transform = None
        return block

    def execute(self, trans, st, at):
        if False:
            while True:
                i = 10
        if self.done:
            return None
        block = self.get_block()
        if block is None:
            block = self.compile()
        events = []
        if trans.hide_request:
            self.transform_event = 'hide'
        if trans.replaced_request:
            self.transform_event = 'replaced'
        if renpy.config.atl_multiple_events:
            if self.transform_event != self.last_transform_event:
                events.append(self.transform_event)
                self.last_transform_event = self.transform_event
        if self.child is not None and self.child.transform_event != self.last_child_transform_event:
            self.last_child_transform_event = self.child.transform_event
            if self.child.transform_event is not None:
                self.transform_event = self.child.transform_event
        if self.transform_event != self.last_transform_event:
            events.append(self.transform_event)
            self.last_transform_event = self.transform_event
        if self.transform_event in renpy.config.repeat_transform_events:
            self.transform_event = None
            self.last_transform_event = None
        old_exception_info = renpy.game.exception_info
        if self.atl_st_offset is None or st - self.atl_st_offset < 0:
            self.atl_st_offset = st
        if self.atl.animation or self.transition:
            timebase = at
        else:
            timebase = st - self.atl_st_offset
        (action, arg, pause) = block.execute(trans, timebase, self.atl_state, events)
        renpy.game.exception_info = old_exception_info
        if action == 'continue' and (not renpy.display.predict.predicting):
            self.atl_state = arg
        else:
            self.done = True
        return pause

    def predict_one(self):
        if False:
            while True:
                i = 10
        self.atl.predict(self.context)

    def visit(self):
        if False:
            while True:
                i = 10
        block = self.get_block()
        if block is None:
            block = self.compile()
        return self.children + block.visit()

class RawStatement(object):
    constant = None

    def __init__(self, loc):
        if False:
            while True:
                i = 10
        super(RawStatement, self).__init__()
        self.loc = loc

    def compile(self, ctx):
        if False:
            for i in range(10):
                print('nop')
        raise Exception('Compile not implemented.')

    def predict(self, ctx):
        if False:
            return 10
        return

    def mark_constant(self, analysis):
        if False:
            return 10
        '\n        Sets self.constant to GLOBAL_CONST if all expressions used in\n        this statement and its children are constant.\n        `analysis`\n            A pyanalysis.Analysis object containing the analysis of this ATL.\n        '
        self.constant = NOT_CONST

class Statement(renpy.object.Object):

    def __init__(self, loc):
        if False:
            for i in range(10):
                print('nop')
        super(Statement, self).__init__()
        self.loc = loc

    def execute(self, trans, st, state, events):
        if False:
            while True:
                i = 10
        raise Exception('Not implemented.')

    def visit(self):
        if False:
            print('Hello World!')
        return []

    def _handles_event(self, event):
        if False:
            while True:
                i = 10
        return False

class RawBlock(RawStatement):
    animation = False
    compiled_block = None

    def __init__(self, loc, statements, animation):
        if False:
            while True:
                i = 10
        super(RawBlock, self).__init__(loc)
        self.statements = statements
        self.animation = animation

    def compile(self, ctx):
        if False:
            i = 10
            return i + 15
        compiling(self.loc)
        statements = [i.compile(ctx) for i in self.statements]
        return Block(self.loc, statements)

    def predict(self, ctx):
        if False:
            while True:
                i = 10
        for i in self.statements:
            i.predict(ctx)

    def analyze(self, parameters=None):
        if False:
            i = 10
            return i + 15
        analysis = Analysis(None)
        if parameters is not None:
            analysis.parameters(parameters)
        self.mark_constant(analysis)
        if self.constant == GLOBAL_CONST:
            self.compile_block()

    def compile_block(self):
        if False:
            while True:
                i = 10
        old_exception_info = renpy.game.exception_info
        try:
            block = self.compile(Context({}))
        except RuntimeError:
            raise Exception('This transform refers to itself in a cycle.')
        except Exception:
            self.constant = NOT_CONST
        else:
            self.compiled_block = block
        renpy.game.exception_info = old_exception_info

    def mark_constant(self, analysis):
        if False:
            return 10
        constant = GLOBAL_CONST
        for i in self.statements:
            i.mark_constant(analysis)
            constant = min(constant, i.constant)
        self.constant = constant

class Block(Statement):

    def __init__(self, loc, statements):
        if False:
            print('Hello World!')
        super(Block, self).__init__(loc)
        self.statements = statements
        self.times = []
        for (i, s) in enumerate(statements):
            if isinstance(s, Time):
                self.times.append((s.time, i + 1))
        self.times.sort()

    def _handles_event(self, event):
        if False:
            while True:
                i = 10
        for i in self.statements:
            if i._handles_event(event):
                return True
        return False

    def execute(self, trans, st, state, events):
        if False:
            while True:
                i = 10
        executing(self.loc)
        if state is not None:
            (index, start, loop_start, repeats, times, child_state) = state
        else:
            (index, start, loop_start, repeats, times, child_state) = (0, 0, 0, 0, self.times[:], None)
        action = 'continue'
        arg = None
        pause = None
        while action == 'continue':
            if times:
                (time, tindex) = times[0]
                target = min(time, st)
                max_pause = time - target
            else:
                target = st
                max_pause = 15
            while True:
                if index >= len(self.statements):
                    return ('next', target - start, None)
                stmt = self.statements[index]
                (action, arg, pause) = stmt.execute(trans, target - start, child_state, events)
                if action == 'continue':
                    if pause is None:
                        pause = max_pause
                    (action, arg, pause) = ('continue', (index, start, loop_start, repeats, times, arg), min(max_pause, pause))
                    break
                elif action == 'event':
                    return (action, arg, pause)
                elif action == 'next':
                    index += 1
                    start = target - arg
                    child_state = None
                elif action == 'repeat':
                    (count, arg) = arg
                    loop_end = target - arg
                    duration = loop_end - loop_start
                    if state is None and duration <= 0:
                        raise Exception('ATL appears to be in an infinite loop.')
                    if duration:
                        new_repeats = int((target - loop_start) / duration)
                    else:
                        new_repeats = 0
                    if count is not None:
                        if repeats + new_repeats >= count:
                            new_repeats = count - repeats
                            loop_start += new_repeats * duration
                            return ('next', target - loop_start, None)
                    repeats += new_repeats
                    loop_start = loop_start + new_repeats * duration
                    start = loop_start
                    index = 0
                    child_state = None
            if times:
                (time, tindex) = times[0]
                if time <= target:
                    times.pop(0)
                    index = tindex
                    start = time
                    child_state = None
                    continue
            return (action, arg, pause)

    def visit(self):
        if False:
            return 10
        return [j for i in self.statements for j in i.visit()]
incompatible_props = {'alignaround': {'xaround', 'yaround', 'xanchoraround', 'yanchoraround'}, 'align': {'xanchor', 'yanchor', 'xpos', 'ypos'}, 'anchor': {'xanchor', 'yanchor'}, 'angle': {'xpos', 'ypos'}, 'anchorangle': {'xangle', 'yangle'}, 'around': {'xaround', 'yaround', 'xanchoraround', 'yanchoraround'}, 'offset': {'xoffset', 'yoffset'}, 'pos': {'xpos', 'ypos'}, 'radius': {'xpos', 'ypos'}, 'anchorradius': {'xanchor', 'yanchor'}, 'size': {'xsize', 'ysize'}, 'xalign': {'xpos', 'xanchor'}, 'xcenter': {'xpos', 'xanchor'}, 'xycenter': {'xpos', 'ypos', 'xanchor', 'yanchor'}, 'xysize': {'xsize', 'ysize'}, 'yalign': {'ypos', 'yanchor'}, 'ycenter': {'ypos', 'yanchor'}}
compatible_pairs = [{'radius', 'angle'}, {'anchorradius', 'anchorangle'}]

class RawMultipurpose(RawStatement):
    warp_function = None

    def __init__(self, loc):
        if False:
            for i in range(10):
                print('nop')
        super(RawMultipurpose, self).__init__(loc)
        self.warper = None
        self.duration = None
        self.properties = []
        self.expressions = []
        self.splines = []
        self.revolution = None
        self.circles = '0'

    def add_warper(self, name, duration, warp_function):
        if False:
            return 10
        self.warper = name
        self.duration = duration
        self.warp_function = warp_function

    def add_property(self, name, exprs):
        if False:
            i = 10
            return i + 15
        '\n        Checks if the property is compatible with any previously included, and\n        sets it.\n        Either returns the previously-set property, if any, or None.\n        '
        newly_set = incompatible_props.get(name, set()) | {name}
        for (old, _e) in self.properties:
            if newly_set.intersection(incompatible_props.get(old, (old,))):
                break
        else:
            old = None
        self.properties.append((name, exprs))
        if old is not None:
            pair = {old, name}
            for i in compatible_pairs:
                if pair == i:
                    old = None
        return old

    def add_expression(self, expr, with_clause):
        if False:
            while True:
                i = 10
        self.expressions.append((expr, with_clause))

    def add_revolution(self, revolution):
        if False:
            i = 10
            return i + 15
        self.revolution = revolution

    def add_circles(self, circles):
        if False:
            while True:
                i = 10
        self.circles = circles

    def add_spline(self, name, exprs):
        if False:
            for i in range(10):
                print('nop')
        self.splines.append((name, exprs))

    def compile(self, ctx):
        if False:
            print('Hello World!')
        compiling(self.loc)
        if self.warper is None and self.warp_function is None and (not self.properties) and (not self.splines) and (len(self.expressions) == 1):
            (expr, withexpr) = self.expressions[0]
            child = ctx.eval(expr)
            if withexpr:
                transition = ctx.eval(withexpr)
            else:
                transition = None
            if isinstance(child, (int, float)):
                return Interpolation(self.loc, 'pause', child, [], None, 0, [])
            child = renpy.easy.displayable(child)
            if isinstance(child, ATLTransformBase) and child.child is None:
                child.compile()
                return child.get_block()
            else:
                return Child(self.loc, child, transition)
        compiling(self.loc)
        if self.warp_function:
            warper = ctx.eval(self.warp_function)
        else:
            warper = self.warper or 'instant'
            if warper not in warpers:
                raise Exception('ATL Warper %s is unknown at runtime.' % warper)
        properties = []
        for (name, expr) in self.properties:
            if name not in PROPERTIES:
                raise Exception('ATL Property %s is unknown at runtime.' % name)
            value = ctx.eval(expr)
            properties.append((name, value))
        splines = []
        for (name, exprs) in self.splines:
            if name not in PROPERTIES:
                raise Exception('ATL Property %s is unknown at runtime.' % name)
            values = [ctx.eval(i) for i in exprs]
            splines.append((name, values))
        for (expr, _with) in self.expressions:
            try:
                value = ctx.eval(expr)
            except Exception:
                raise Exception('Could not evaluate expression %r when compiling ATL.' % expr)
            if not isinstance(value, ATLTransformBase):
                raise Exception('Expression %r is not an ATL transform, and so cannot be included in an ATL interpolation.' % expr)
            value.compile()
            if value.properties is None:
                raise Exception('ATL transform %r is too complicated to be included in interpolation.' % expr)
            properties.extend(value.properties)
        duration = ctx.eval(self.duration)
        circles = ctx.eval(self.circles)
        return Interpolation(self.loc, warper, duration, properties, self.revolution, circles, splines)

    def mark_constant(self, analysis):
        if False:
            for i in range(10):
                print('nop')
        constant = GLOBAL_CONST
        is_constant_expr = analysis.is_constant_expr
        constant = min(constant, is_constant_expr(self.warp_function))
        constant = min(constant, is_constant_expr(self.duration))
        constant = min(constant, is_constant_expr(self.circles))
        for (_name, expr) in self.properties:
            constant = min(constant, is_constant_expr(expr))
        for (_name, exprs) in self.splines:
            for expr in exprs:
                constant = min(constant, is_constant_expr(expr))
        for (expr, withexpr) in self.expressions:
            constant = min(constant, is_constant_expr(expr))
            constant = min(constant, is_constant_expr(withexpr))
        self.constant = constant

    def predict(self, ctx):
        if False:
            print('Hello World!')
        for (i, _j) in self.expressions:
            try:
                i = ctx.eval(i)
            except Exception:
                continue
            if isinstance(i, ATLTransformBase):
                i.atl.predict(ctx)
                return
            try:
                renpy.easy.predict(i)
            except Exception:
                continue

class RawContainsExpr(RawStatement):

    def __init__(self, loc, expr):
        if False:
            for i in range(10):
                print('nop')
        super(RawContainsExpr, self).__init__(loc)
        self.expression = expr

    def compile(self, ctx):
        if False:
            for i in range(10):
                print('nop')
        compiling(self.loc)
        child = ctx.eval(self.expression)
        return Child(self.loc, child, None)

    def mark_constant(self, analysis):
        if False:
            i = 10
            return i + 15
        self.constant = analysis.is_constant_expr(self.expression)

class RawChild(RawStatement):

    def __init__(self, loc, child):
        if False:
            print('Hello World!')
        super(RawChild, self).__init__(loc)
        self.children = [child]

    def compile(self, ctx):
        if False:
            for i in range(10):
                print('nop')
        children = []
        for i in self.children:
            children.append(renpy.display.motion.ATLTransform(i, context=ctx.context))
        box = renpy.display.layout.MultiBox(layout='fixed')
        for i in children:
            box.add(i)
        return Child(self.loc, box, None)

    def mark_constant(self, analysis):
        if False:
            while True:
                i = 10
        constant = GLOBAL_CONST
        for i in self.children:
            i.mark_constant(analysis)
            constant = min(constant, i.constant)
        self.constant = constant

class Child(Statement):

    def __init__(self, loc, child, transition):
        if False:
            return 10
        super(Child, self).__init__(loc)
        self.child = child
        self.transition = transition

    def execute(self, trans, st, state, events):
        if False:
            return 10
        executing(self.loc)
        old_child = trans.raw_child
        child = self.child
        if child._duplicatable:
            child = self.child._duplicate(trans._args)
            child._unique()
        if old_child is not None and old_child is not renpy.display.motion.null and (self.transition is not None):
            child = self.transition(old_widget=old_child, new_widget=child)
            child._unique()
        trans.set_child(child, duplicate=False)
        trans.raw_child = self.child
        return ('next', st, None)

    def visit(self):
        if False:
            return 10
        return [self.child]

class Interpolation(Statement):

    def __init__(self, loc, warper, duration, properties, revolution, circles, splines):
        if False:
            while True:
                i = 10
        super(Interpolation, self).__init__(loc)
        self.warper = warper
        self.duration = duration
        self.properties = properties
        self.splines = splines
        self.revolution = revolution
        self.circles = circles

    def execute(self, trans, st, state, events):
        if False:
            for i in range(10):
                print('nop')
        executing(self.loc)
        warper = warpers.get(self.warper, self.warper)
        if state is None and self.warper == 'pause' and (self.duration == 0) and renpy.config.atl_one_frame:
            force_frame = True
        else:
            force_frame = False
        if self.duration:
            complete = min(1.0, st / self.duration)
        else:
            complete = 1.0
        if complete < 0.0:
            complete = 0.0
        elif complete > 1.0:
            complete = 1.0
        complete = warper(complete)
        if state is None or len(state) != 6:
            newts = renpy.display.motion.TransformState()
            newts.take_state(trans.state)
            has_angle = False
            has_radius = False
            has_anchorangle = False
            has_anchorradius = False
            for (k, v) in self.properties:
                setattr(newts, k, v)
                if k == 'angle':
                    has_angle = True
                elif k == 'radius':
                    has_radius = True
                elif k == 'anchorangle':
                    has_anchorangle = True
                elif k == 'anchorradius':
                    has_anchorradius = True
            linear = trans.state.diff(newts)
            angle = None
            radius = None
            anchorangle = None
            anchorradius = None
            splines = []
            revdir = self.revolution
            circles = self.circles
            if (revdir or ((has_angle or has_radius) and renpy.config.automatic_polar_motion)) and newts.xaround is not None:
                for i in ['xpos', 'ypos', 'xanchor', 'yanchor', 'xaround', 'yaround', 'xanchoraround', 'yanchoraround']:
                    linear.pop(i, None)
                if revdir is not None:
                    trans.state.xaround = newts.xaround
                    trans.state.yaround = newts.yaround
                    trans.state.xanchoraround = newts.xanchoraround
                    trans.state.yanchoraround = newts.yanchoraround
                    startangle = trans.state.angle
                    endangle = newts.angle
                    startradius = trans.state.radius
                    endradius = newts.radius
                    startanchorangle = trans.state.anchorangle
                    endanchorangle = newts.anchorangle
                    startanchorradius = trans.state.anchorradius
                    endanchorradius = newts.anchorradius
                    if revdir == 'clockwise':
                        if endangle < startangle:
                            startangle -= 360
                        if endanchorangle < startanchorangle:
                            startanchorangle -= 360
                        startangle -= circles * 360
                        startanchorangle -= circles * 360
                    elif revdir == 'counterclockwise':
                        if endangle > startangle:
                            startangle += 360
                        if endanchorangle > startanchorangle:
                            startanchorangle += 360
                        startangle += circles * 360
                        startanchorangle += circles * 360
                    has_radius = True
                    has_angle = True
                    has_anchorangle = True
                    has_anchorradius = True
                    radius = (startradius, endradius)
                    angle = (startangle, endangle)
                    anchorradius = (startanchorradius, endanchorradius)
                    anchorangle = (startanchorangle, endanchorangle)
                else:
                    if has_angle:
                        start = trans.state.angle
                        end = newts.last_angle
                        if end - start > 180:
                            start += 360
                        if end - start < -180:
                            start -= 360
                        angle = (start, end)
                    if has_radius:
                        radius = (trans.state.radius, newts.radius)
                    if has_anchorangle:
                        start = trans.state.anchorangle
                        end = newts.last_anchorangle
                        if end - start > 180:
                            start += 360
                        if end - start < -180:
                            start -= 360
                        anchorangle = (start, end)
                    if has_anchorradius:
                        anchorradius = (trans.state.anchorradius, newts.anchorradius)
            for (name, values) in self.splines:
                splines.append((name, [getattr(trans.state, name)] + values))
            state = (linear, angle, radius, anchorangle, anchorradius, splines)
            for (k, v) in self.properties:
                if k not in linear:
                    setattr(trans.state, k, v)
        else:
            (linear, angle, radius, anchorangle, anchorradius, splines) = state
        for (k, (old, new)) in linear.items():
            if k == 'orientation':
                if old is None:
                    old = (0.0, 0.0, 0.0)
                if new is not None:
                    value = renpy.display.quaternion.euler_slerp(complete, old, new)
                elif complete >= 1:
                    value = None
                else:
                    value = old
            else:
                value = interpolate(complete, old, new, PROPERTIES[k])
            setattr(trans.state, k, value)
        if angle is not None:
            (startangle, endangle) = angle[:2]
            angle = interpolate(complete, startangle, endangle, float)
            trans.state.angle = angle
        if radius is not None:
            (startradius, endradius) = radius
            trans.state.radius = interpolate(complete, startradius, endradius, position)
        if anchorangle is not None:
            (startangle, endangle) = anchorangle[:2]
            angle = interpolate(complete, startangle, endangle, float)
            trans.state.anchorangle = angle
        if anchorradius is not None:
            (startradius, endradius) = anchorradius
            trans.state.anchorradius = interpolate(complete, startradius, endradius, position)
        for (name, values) in splines:
            value = interpolate_spline(complete, values)
            setattr(trans.state, name, value)
        if st >= self.duration and (not force_frame):
            return ('next', st - self.duration, None)
        elif not self.properties and (not self.revolution) and (not self.splines):
            return ('continue', state, max(0, self.duration - st))
        else:
            return ('continue', state, 0)

class RawRepeat(RawStatement):

    def __init__(self, loc, repeats):
        if False:
            for i in range(10):
                print('nop')
        super(RawRepeat, self).__init__(loc)
        self.repeats = repeats

    def compile(self, ctx):
        if False:
            print('Hello World!')
        compiling(self.loc)
        repeats = self.repeats
        if repeats is not None:
            repeats = ctx.eval(repeats)
        return Repeat(self.loc, repeats)

    def mark_constant(self, analysis):
        if False:
            print('Hello World!')
        self.constant = analysis.is_constant_expr(self.repeats)

class Repeat(Statement):

    def __init__(self, loc, repeats):
        if False:
            while True:
                i = 10
        super(Repeat, self).__init__(loc)
        self.repeats = repeats

    def execute(self, trans, st, state, events):
        if False:
            return 10
        return ('repeat', (self.repeats, st), 0)

class RawParallel(RawStatement):

    def __init__(self, loc, block):
        if False:
            i = 10
            return i + 15
        super(RawParallel, self).__init__(loc)
        self.blocks = [block]

    def compile(self, ctx):
        if False:
            return 10
        return Parallel(self.loc, [i.compile(ctx) for i in self.blocks])

    def predict(self, ctx):
        if False:
            print('Hello World!')
        for i in self.blocks:
            i.predict(ctx)

    def mark_constant(self, analysis):
        if False:
            i = 10
            return i + 15
        constant = GLOBAL_CONST
        for i in self.blocks:
            i.mark_constant(analysis)
            constant = min(constant, i.constant)
        self.constant = constant

class Parallel(Statement):

    def __init__(self, loc, blocks):
        if False:
            while True:
                i = 10
        super(Parallel, self).__init__(loc)
        self.blocks = blocks

    def _handles_event(self, event):
        if False:
            print('Hello World!')
        for i in self.blocks:
            if i._handles_event(event):
                return True
        return False

    def execute(self, trans, st, state, events):
        if False:
            for i in range(10):
                print('nop')
        executing(self.loc)
        if state is None:
            state = [(i, None) for i in self.blocks]
        left = []
        pauses = []
        newstate = []
        for (i, istate) in state:
            (action, arg, pause) = i.execute(trans, st, istate, events)
            if pause is not None:
                pauses.append(pause)
            if action == 'continue':
                newstate.append((i, arg))
            elif action == 'next':
                left.append(arg)
            elif action == 'event':
                return (action, arg, pause)
        if newstate:
            return ('continue', newstate, min(pauses))
        else:
            return ('next', min(left), None)

    def visit(self):
        if False:
            print('Hello World!')
        return [j for i in self.blocks for j in i.visit()]

class RawChoice(RawStatement):

    def __init__(self, loc, chance, block):
        if False:
            print('Hello World!')
        super(RawChoice, self).__init__(loc)
        self.choices = [(chance, block)]

    def compile(self, ctx):
        if False:
            for i in range(10):
                print('nop')
        compiling(self.loc)
        return Choice(self.loc, [(ctx.eval(chance), block.compile(ctx)) for (chance, block) in self.choices])

    def predict(self, ctx):
        if False:
            while True:
                i = 10
        for (_i, j) in self.choices:
            j.predict(ctx)

    def mark_constant(self, analysis):
        if False:
            while True:
                i = 10
        constant = GLOBAL_CONST
        for (_chance, block) in self.choices:
            block.mark_constant(analysis)
            constant = min(constant, block.constant)
        self.constant = constant

class Choice(Statement):

    def __init__(self, loc, choices):
        if False:
            while True:
                i = 10
        super(Choice, self).__init__(loc)
        self.choices = choices

    def _handles_event(self, event):
        if False:
            for i in range(10):
                print('nop')
        for i in self.choices:
            if i[1]._handles_event(event):
                return True
        return False

    def execute(self, trans, st, state, events):
        if False:
            return 10
        executing(self.loc)
        choice = None
        if state is None:
            total = 0
            for (chance, choice) in self.choices:
                total += chance
            n = random.uniform(0, total)
            for (chance, choice) in self.choices:
                if n < chance:
                    break
                n -= chance
            cstate = None
        else:
            (choice, cstate) = state
        (action, arg, pause) = choice.execute(trans, st, cstate, events)
        if action == 'continue':
            return ('continue', (choice, arg), pause)
        else:
            return (action, arg, None)

    def visit(self):
        if False:
            return 10
        return [j for i in self.choices for j in i[1].visit()]

class RawTime(RawStatement):

    def __init__(self, loc, time):
        if False:
            i = 10
            return i + 15
        super(RawTime, self).__init__(loc)
        self.time = time

    def compile(self, ctx):
        if False:
            return 10
        compiling(self.loc)
        return Time(self.loc, ctx.eval(self.time))

    def mark_constant(self, analysis):
        if False:
            i = 10
            return i + 15
        self.constant = analysis.is_constant_expr(self.time)

class Time(Statement):

    def __init__(self, loc, time):
        if False:
            while True:
                i = 10
        super(Time, self).__init__(loc)
        self.time = time

    def execute(self, trans, st, state, events):
        if False:
            for i in range(10):
                print('nop')
        return ('continue', None, None)

class RawOn(RawStatement):

    def __init__(self, loc, names, block):
        if False:
            for i in range(10):
                print('nop')
        super(RawOn, self).__init__(loc)
        self.handlers = {}
        for i in names:
            self.handlers[i] = block

    def compile(self, ctx):
        if False:
            return 10
        compiling(self.loc)
        handlers = {}
        for (k, v) in self.handlers.items():
            handlers[k] = v.compile(ctx)
        return On(self.loc, handlers)

    def predict(self, ctx):
        if False:
            i = 10
            return i + 15
        for i in self.handlers.values():
            i.predict(ctx)

    def mark_constant(self, analysis):
        if False:
            return 10
        constant = GLOBAL_CONST
        for block in self.handlers.values():
            block.mark_constant(analysis)
            constant = min(constant, block.constant)
        self.constant = constant

class On(Statement):

    def __init__(self, loc, handlers):
        if False:
            return 10
        super(On, self).__init__(loc)
        self.handlers = handlers

    def _handles_event(self, event):
        if False:
            return 10
        if event in self.handlers:
            return True
        else:
            return False

    def execute(self, trans, st, state, events):
        if False:
            i = 10
            return i + 15
        executing(self.loc)
        if state is None:
            (name, start, cstate) = ('start', st, None)
        else:
            (name, start, cstate) = state
        for event in events:
            while event:
                if event in self.handlers:
                    break
                event = event.partition('_')[2]
            if not event:
                continue
            lock_event = name == 'hide' and trans.hide_request or (name == 'replaced' and trans.replaced_request)
            if not lock_event:
                name = event
                start = st
                cstate = None
        while True:
            if name not in self.handlers:
                return ('continue', (name, start, cstate), None)
            (action, arg, pause) = self.handlers[name].execute(trans, st - start, cstate, events)
            if action == 'continue':
                if name == 'hide' or name == 'replaced':
                    trans.hide_response = False
                    trans.replaced_response = False
                return ('continue', (name, start, arg), pause)
            elif action == 'next':
                if name == 'default' or name == 'hide' or name == 'replaced':
                    name = None
                else:
                    name = 'default'
                start = st - arg
                cstate = None
                continue
            elif action == 'event':
                (name, arg) = arg
                if name in self.handlers:
                    start = max(st - arg, st - 30)
                    cstate = None
                    continue
                return ('event', (name, arg), None)

    def visit(self):
        if False:
            return 10
        return [j for i in self.handlers.values() for j in i.visit()]

class RawEvent(RawStatement):

    def __init__(self, loc, name):
        if False:
            return 10
        super(RawEvent, self).__init__(loc)
        self.name = name

    def compile(self, ctx):
        if False:
            for i in range(10):
                print('nop')
        return Event(self.loc, self.name)

    def mark_constant(self, analysis):
        if False:
            return 10
        self.constant = GLOBAL_CONST

class Event(Statement):

    def __init__(self, loc, name):
        if False:
            for i in range(10):
                print('nop')
        super(Event, self).__init__(loc)
        self.name = name

    def execute(self, trans, st, state, events):
        if False:
            while True:
                i = 10
        return ('event', (self.name, st), None)

class RawFunction(RawStatement):

    def __init__(self, loc, expr):
        if False:
            i = 10
            return i + 15
        super(RawFunction, self).__init__(loc)
        self.expr = expr

    def compile(self, ctx):
        if False:
            i = 10
            return i + 15
        compiling(self.loc)
        return Function(self.loc, ctx.eval(self.expr))

    def mark_constant(self, analysis):
        if False:
            print('Hello World!')
        self.constant = analysis.is_constant_expr(self.expr)

class Function(Statement):

    def __init__(self, loc, function):
        if False:
            print('Hello World!')
        super(Function, self).__init__(loc)
        self.function = function

    def _handles_event(self, event):
        if False:
            print('Hello World!')
        return True

    def execute(self, trans, st, state, events):
        if False:
            print('Hello World!')
        block = state or renpy.config.atl_function_always_blocks
        fr = self.function(trans, st if block else 0, trans.at)
        if not block and fr is not None:
            block = True
            fr = self.function(trans, st, trans.at)
        if fr is not None:
            return ('continue', True, fr)
        else:
            return ('next', 0 if block else st, None)

def parse_atl(l):
    if False:
        print('Hello World!')
    l.advance()
    block_loc = l.get_location()
    statements = []
    animation = False
    while not l.eob:
        loc = l.get_location()
        if l.keyword('repeat'):
            repeats = l.simple_expression()
            statements.append(RawRepeat(loc, repeats))
        elif l.keyword('block'):
            l.require(':')
            l.expect_eol()
            l.expect_block('block')
            block = parse_atl(l.subblock_lexer())
            statements.append(block)
        elif l.keyword('contains'):
            expr = l.simple_expression()
            if expr:
                l.expect_noblock('contains expression')
                statements.append(RawContainsExpr(loc, expr))
            else:
                l.require(':')
                l.expect_eol()
                l.expect_block('contains')
                block = parse_atl(l.subblock_lexer())
                statements.append(RawChild(loc, block))
        elif l.keyword('parallel'):
            l.require(':')
            l.expect_eol()
            l.expect_block('parallel')
            block = parse_atl(l.subblock_lexer())
            statements.append(RawParallel(loc, block))
        elif l.keyword('choice'):
            chance = l.simple_expression()
            if not chance:
                chance = '1.0'
            l.require(':')
            l.expect_eol()
            l.expect_block('choice')
            block = parse_atl(l.subblock_lexer())
            statements.append(RawChoice(loc, chance, block))
        elif l.keyword('on'):
            names = [l.require(l.word)]
            while l.match(','):
                name = l.word()
                if name is None:
                    break
                names.append(name)
            l.require(':')
            l.expect_eol()
            l.expect_block('on')
            block = parse_atl(l.subblock_lexer())
            statements.append(RawOn(loc, names, block))
        elif l.keyword('time'):
            time = l.require(l.simple_expression)
            l.expect_noblock('time')
            statements.append(RawTime(loc, time))
        elif l.keyword('function'):
            expr = l.require(l.simple_expression)
            l.expect_noblock('function')
            statements.append(RawFunction(loc, expr))
        elif l.keyword('event'):
            name = l.require(l.word)
            l.expect_noblock('event')
            statements.append(RawEvent(loc, name))
        elif l.keyword('pass'):
            l.expect_noblock('pass')
            statements.append(None)
        elif l.keyword('animation'):
            l.expect_noblock('animation')
            animation = True
        else:
            rm = renpy.atl.RawMultipurpose(loc)
            last_expression = False
            this_expression = False
            cp = l.checkpoint()
            warper = l.name()
            if warper in warpers:
                duration = l.require(l.simple_expression)
                warp_function = None
            elif warper == 'warp':
                warper = None
                warp_function = l.require(l.simple_expression)
                duration = l.require(l.simple_expression)
            else:
                l.revert(cp)
                warper = None
                warp_function = None
                duration = '0'
            rm.add_warper(warper, duration, warp_function)
            ll = l
            has_block = False
            while True:
                if warper is not None and (not has_block) and ll.match(':'):
                    ll.expect_eol()
                    ll.expect_block('ATL')
                    has_block = True
                    ll = l.subblock_lexer()
                    ll.advance()
                    ll.expect_noblock('ATL')
                if has_block and ll.eol():
                    ll.advance()
                    ll.expect_noblock('ATL')
                last_expression = this_expression
                this_expression = False
                if ll.keyword('pass'):
                    continue
                if ll.keyword('clockwise'):
                    rm.add_revolution('clockwise')
                    continue
                if ll.keyword('counterclockwise'):
                    rm.add_revolution('counterclockwise')
                    continue
                if ll.keyword('circles'):
                    expr = l.require(l.simple_expression)
                    rm.add_circles(expr)
                    continue
                cp = ll.checkpoint()
                prop = ll.name()
                if prop in PROPERTIES or (prop and prop.startswith('u_')):
                    expr = ll.require(ll.simple_expression)
                    knots = []
                    while ll.keyword('knot'):
                        knots.append(ll.require(ll.simple_expression))
                    if knots:
                        if prop == 'orientation':
                            raise Exception("Orientation doesn't support spline.")
                        knots.append(expr)
                        rm.add_spline(prop, knots)
                    else:
                        addprop_rv = rm.add_property(prop, expr)
                        if addprop_rv == prop:
                            ll.deferred_error('check_conflicting_properties', 'property {!r} is given a value more than once'.format(prop))
                        elif addprop_rv:
                            ll.deferred_error('check_conflicting_properties', 'properties {!r} and {!r} conflict with each other'.format(prop, addprop_rv))
                    continue
                ll.revert(cp)
                expr = ll.simple_expression()
                if not expr:
                    break
                if last_expression:
                    ll.error('ATL statement contains two expressions in a row; is one of them a misspelled property? If not, separate them with pass.')
                this_expression = True
                if ll.keyword('with'):
                    with_expr = ll.require(ll.simple_expression)
                else:
                    with_expr = None
                rm.add_expression(expr, with_expr)
            if not has_block:
                l.expect_noblock('ATL')
            statements.append(rm)
        if l.eol():
            l.advance()
            continue
        l.require(',', 'comma or end of line')
    merged = []
    old = None
    for new in statements:
        if isinstance(old, RawParallel) and isinstance(new, RawParallel):
            old.blocks.extend(new.blocks)
            continue
        elif isinstance(old, RawChoice) and isinstance(new, RawChoice):
            old.choices.extend(new.choices)
            continue
        elif isinstance(old, RawChild) and isinstance(new, RawChild):
            old.children.extend(new.children)
            continue
        elif isinstance(old, RawOn) and isinstance(new, RawOn):
            old.handlers.update(new.handlers)
            continue
        elif new is None:
            old = new
            continue
        merged.append(new)
        old = new
    return RawBlock(block_loc, merged, animation)

def deep_compare(a, b):
    if False:
        while True:
            i = 10
    '\n    Compares two trees of ATL statements for equality.\n    '
    if type(a) != type(b):
        return False
    if isinstance(a, (list, tuple)):
        return all((deep_compare(i, j) for (i, j) in zip(a, b)))
    if isinstance(a, dict):
        if len(a) != len(b):
            return False
        return all((k in b and deep_compare(a[k], b[k]) for k in a))
    if isinstance(a, Statement):
        return deep_compare(a.__dict__, b.__dict__)
    try:
        return a == b
    except Exception:
        return True