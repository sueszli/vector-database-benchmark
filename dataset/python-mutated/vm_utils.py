"""Utilities used in vm.py."""
import abc
import collections.abc
import dataclasses
import enum
import itertools
import logging
import re
import reprlib
from typing import Optional, Sequence, Tuple, Union
from pytype import overriding_checks
from pytype import state as frame_state
from pytype import utils
from pytype.abstract import abstract
from pytype.abstract import abstract_utils
from pytype.abstract import function
from pytype.abstract import mixin
from pytype.blocks import blocks
from pytype.overlays import metaclass
from pytype.pyc import opcodes
from pytype.pytd import mro
from pytype.pytd import pytd
from pytype.pytd import slots
from pytype.typegraph import cfg
log = logging.getLogger(__name__)
_TRUNCATE = 120
_TRUNCATE_STR = 72
_repr_obj = reprlib.Repr()
_repr_obj.maxother = _TRUNCATE
_repr_obj.maxstring = _TRUNCATE_STR
repper = _repr_obj.repr
_FUNCTION_TYPE_COMMENT_RE = re.compile('^\\((.*)\\)\\s*->\\s*(\\S.*?)\\s*$')
_BUILTIN_MATCHERS = ('bool', 'bytearray', 'bytes', 'dict', 'float', 'frozenset', 'int', 'list', 'set', 'str', 'tuple')

class PopBehavior(enum.Enum):
    """Ways in which a JUMP_IF opcode may pop a value off the stack."""
    NONE = enum.auto()
    OR = enum.auto()
    ALWAYS = enum.auto()

@dataclasses.dataclass(eq=True, frozen=True)
class _Block:
    type: str
    level: int
    op_index: int

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'Block({self.type}: {self.op_index} level={self.level})'

class FindIgnoredTypeComments:
    """A visitor that finds type comments that will be ignored."""

    def __init__(self, type_comments):
        if False:
            while True:
                i = 10
        self._type_comments = type_comments
        self._ignored_type_lines = set(type_comments)

    def visit_code(self, code):
        if False:
            print('Hello World!')
        'Interface for pyc.visit.'
        for op in code.code_iter:
            if isinstance(op, blocks.STORE_OPCODES):
                if op.annotation:
                    annot = op.annotation
                    if self._type_comments.get(op.line) == annot:
                        self._ignored_type_lines.discard(op.line)
            elif isinstance(op, opcodes.MAKE_FUNCTION):
                if op.annotation:
                    (_, line) = op.annotation
                    self._ignored_type_lines.discard(line)
        return code

    def ignored_lines(self):
        if False:
            print('Hello World!')
        'Returns a set of lines that contain ignored type comments.'
        return self._ignored_type_lines

class FinallyStateTracker:
    """Track return state for try/except/finally blocks."""
    RETURN_STATES = ('return', 'exception')

    def __init__(self):
        if False:
            return 10
        self.stack = []

    def process(self, op, state, ctx) -> Optional[str]:
        if False:
            print('Hello World!')
        'Store state.why, or return it from a stored state.'
        if ctx.vm.is_setup_except(op):
            self.stack.append([op, None])
        if isinstance(op, opcodes.END_FINALLY):
            if self.stack:
                (_, why) = self.stack.pop()
                if why:
                    return why
        elif self.stack and state.why in self.RETURN_STATES:
            self.stack[-1][-1] = state.why

    def check_early_exit(self, state) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Check if we are exiting the frame from within an except block.'
        return state.block_stack and any((x.type == 'finally' for x in state.block_stack)) and (state.why in self.RETURN_STATES)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return repr(self.stack)

class _NameErrorDetails(abc.ABC):
    """Base class for detailed name error messages."""

    @abc.abstractmethod
    def to_error_message(self) -> str:
        if False:
            print('Hello World!')
        ...

class _NameInInnerClassErrorDetails(_NameErrorDetails):

    def __init__(self, attr, class_name):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._attr = attr
        self._class_name = class_name

    def to_error_message(self):
        if False:
            while True:
                i = 10
        return f'Cannot reference {self._attr!r} from class {self._class_name!r} before the class is fully defined'

class _NameInOuterClassErrorDetails(_NameErrorDetails):
    """Name error details for a name defined in an outer class."""

    def __init__(self, attr, prefix, class_name):
        if False:
            return 10
        super().__init__()
        self._attr = attr
        self._prefix = prefix
        self._class_name = class_name

    def to_error_message(self):
        if False:
            i = 10
            return i + 15
        full_attr_name = f'{self._class_name}.{self._attr}'
        if self._prefix:
            full_class_name = f'{self._prefix}.{self._class_name}'
        else:
            full_class_name = self._class_name
        return f'Use {full_attr_name!r} to reference {self._attr!r} from class {full_class_name!r}'

class _NameInOuterFunctionErrorDetails(_NameErrorDetails):
    """Name error details for a name defined in an outer function."""

    def __init__(self, attr, outer_scope, inner_scope):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._attr = attr
        self._outer_scope = outer_scope
        self._inner_scope = inner_scope

    def to_error_message(self):
        if False:
            return 10
        keyword = 'global' if 'global' in self._outer_scope else 'nonlocal'
        return f'Add `{keyword} {self._attr}` in {self._inner_scope} to reference {self._attr!r} from {self._outer_scope}'

def _get_scopes(state, names: Sequence[str], ctx) -> Sequence[Union[abstract.InterpreterClass, abstract.InterpreterFunction]]:
    if False:
        while True:
            i = 10
    "Gets the class or function objects for a sequence of nested scope names.\n\n  For example, if the code under analysis is:\n    class Foo:\n      def f(self):\n        def g(): ...\n  then when called with ['Foo', 'f', 'g'], this method returns\n  [InterpreterClass(Foo), InterpreterFunction(f), InterpreterFunction(g)].\n\n  Arguments:\n    state: The current state.\n    names: A sequence of names for consecutive nested scopes in the module\n      under analysis. Must start with a module-level name.\n    ctx: The current context.\n\n  Returns:\n    The class or function object corresponding to each name in 'names'.\n  "
    scopes = []
    for name in names:
        prev = scopes[-1] if scopes else None
        if not prev:
            try:
                (_, var) = ctx.vm.load_global(state, name)
            except KeyError:
                break
        elif isinstance(prev, abstract.InterpreterClass):
            if name in prev.members:
                var = prev.members[name]
            else:
                break
        else:
            assert isinstance(prev, abstract.InterpreterFunction)
            if prev.last_frame and name in prev.last_frame.f_locals.pyval:
                var = prev.last_frame.f_locals.pyval[name]
            else:
                break
        try:
            scopes.append(abstract_utils.get_atomic_value(var, (abstract.InterpreterClass, abstract.InterpreterFunction)))
        except abstract_utils.ConversionError:
            break
    return scopes

def get_name_error_details(state, name: str, ctx) -> Optional[_NameErrorDetails]:
    if False:
        i = 10
        return i + 15
    'Gets a detailed error message for [name-error].'
    class_frames = []
    first_function_frame = None
    for frame in reversed(ctx.vm.frames):
        if not frame.func:
            break
        if frame.func.data.is_class_builder:
            class_frames.append(frame)
        else:
            first_function_frame = frame
            break
    clean = lambda func_name: func_name.replace('.<locals>', '')
    if first_function_frame:
        parts = clean(first_function_frame.func.data.name).split('.')
        if first_function_frame is ctx.vm.frame:
            parts = parts[:-1]
    else:
        parts = []
    (prefix, class_name_parts) = (None, [])
    for scope in itertools.chain(reversed(_get_scopes(state, parts, ctx)), [None]):
        if class_name_parts:
            if isinstance(scope, abstract.InterpreterClass):
                class_name_parts.append(scope.name)
            elif scope:
                prefix = clean(scope.name)
                break
        elif isinstance(scope, abstract.InterpreterClass):
            if name in scope.members:
                class_name_parts.append(scope.name)
        else:
            outer_scope = None
            if scope:
                if scope.last_frame and name in scope.last_frame.f_locals.pyval or (not scope.last_frame and name in ctx.vm.annotated_locals[scope.name.rsplit('.', 1)[-1]]):
                    outer_scope = f'function {clean(scope.name)!r}'
            else:
                try:
                    _ = ctx.vm.load_global(state, name)
                except KeyError:
                    pass
                else:
                    outer_scope = 'global scope'
            if outer_scope:
                if not ctx.vm.frame.func.data.is_class_builder:
                    inner_scope = f'function {clean(ctx.vm.frame.func.data.name)!r}'
                elif ctx.python_version >= (3, 11):
                    inner_scope = f'class {clean(class_frames[0].func.data.name)!r}'
                else:
                    class_name = '.'.join(parts + [class_frame.func.data.name for class_frame in reversed(class_frames)])
                    inner_scope = f'class {class_name!r}'
                return _NameInOuterFunctionErrorDetails(name, outer_scope, inner_scope)
    if class_name_parts:
        return _NameInOuterClassErrorDetails(name, prefix, '.'.join(reversed(class_name_parts)))
    if class_frames:
        for (i, frame) in enumerate(class_frames[1:]):
            if ctx.python_version >= (3, 11):
                short_name = frame.func.data.name.rsplit('.', 1)[-1]
            else:
                short_name = frame.func.data.name
            if name in ctx.vm.annotated_locals[short_name]:
                if ctx.python_version >= (3, 11):
                    class_name = clean(frame.func.data.name)
                else:
                    class_parts = [part.func.data.name for part in reversed(class_frames[i + 1:])]
                    class_name = '.'.join(parts + class_parts)
                return _NameInInnerClassErrorDetails(name, class_name)
    return None

def log_opcode(op, state, frame, stack_size):
    if False:
        i = 10
        return i + 15
    'Write a multi-line log message, including backtrace and stack.'
    if not log.isEnabledFor(logging.INFO):
        return
    if isinstance(op, (opcodes.CACHE, opcodes.PRECALL)):
        return
    indent = ' > ' * (stack_size - 1)
    stack_rep = repper(state.data_stack)
    block_stack_rep = repper(state.block_stack)
    if frame.module_name:
        name = frame.f_code.name
        log.info('%s | index: %d, %r, module: %s line: %d', indent, op.index, name, frame.module_name, op.line)
    else:
        log.info('%s | index: %d, line: %d', indent, op.index, op.line)
    log.info('%s | data_stack: %s', indent, stack_rep)
    log.info('%s | data_stack: %s', indent, [x.data for x in state.data_stack])
    log.info('%s | block_stack: %s', indent, block_stack_rep)
    log.info('%s | node: <%d>%s', indent, state.node.id, state.node.name)
    log.info('%s ## %s', indent, utils.maybe_truncate(str(op), _TRUNCATE))

def _process_base_class(node, base, ctx):
    if False:
        i = 10
        return i + 15
    'Process a base class for InterpreterClass creation.'
    new_base = ctx.program.NewVariable()
    for b in base.bindings:
        base_val = b.data
        if isinstance(b.data, abstract.AnnotationContainer):
            base_val = base_val.base_cls
        if isinstance(base_val, abstract.Union):
            for o in base_val.options:
                new_base.AddBinding(o, {b}, node)
        else:
            new_base.AddBinding(base_val, {b}, node)
    base = new_base
    if not any((isinstance(t, (abstract.Class, abstract.AMBIGUOUS_OR_EMPTY)) for t in base.data)):
        ctx.errorlog.base_class_error(ctx.vm.frames, base)
    return base

def _filter_out_metaclasses(bases, ctx):
    if False:
        for i in range(10):
            print('nop')
    'Process the temporary classes created by six.with_metaclass.\n\n  six.with_metaclass constructs an anonymous class holding a metaclass and a\n  list of base classes; if we find instances in `bases`, store the first\n  metaclass we find and remove all metaclasses from `bases`.\n\n  Args:\n    bases: The list of base classes for the class being constructed.\n    ctx: The current context.\n\n  Returns:\n    A tuple of (metaclass, base classes)\n  '
    non_meta = []
    meta = None
    for base in bases:
        with_metaclass = False
        for b in base.data:
            if isinstance(b, metaclass.WithMetaclassInstance):
                with_metaclass = True
                if not meta:
                    meta = b.cls.to_variable(ctx.root_node)
                non_meta.extend(b.bases)
        if not with_metaclass:
            non_meta.append(base)
    return (meta, non_meta)

def _expand_generic_protocols(node, bases, ctx):
    if False:
        i = 10
        return i + 15
    'Expand Protocol[T, ...] to Protocol, Generic[T, ...].'
    expanded_bases = []
    for base in bases:
        if any((abstract_utils.is_generic_protocol(b) for b in base.data)):
            protocol_base = ctx.program.NewVariable()
            generic_base = ctx.program.NewVariable()
            generic_cls = ctx.convert.lookup_value('typing', 'Generic')
            for b in base.bindings:
                if abstract_utils.is_generic_protocol(b.data):
                    protocol_base.AddBinding(b.data.base_cls, {b}, node)
                    generic_base.AddBinding(abstract.ParameterizedClass(generic_cls, b.data.formal_type_parameters, ctx, b.data.template), {b}, node)
                else:
                    protocol_base.PasteBinding(b)
            expanded_bases.append(generic_base)
            expanded_bases.append(protocol_base)
        else:
            expanded_bases.append(base)
    return expanded_bases

def _check_final_members(cls, class_dict, ctx):
    if False:
        for i in range(10):
            print('nop')
    'Check if the new class overrides a final attribute or method.'
    methods = set(class_dict)
    for base in cls.mro[1:]:
        if not isinstance(base, abstract.Class):
            continue
        if isinstance(base, abstract.PyTDClass):
            for m in methods:
                member = base.final_members.get(m)
                if isinstance(member, pytd.Function):
                    ctx.errorlog.overriding_final_method(ctx.vm.frames, cls, base, m)
                elif member:
                    ctx.errorlog.overriding_final_attribute(ctx.vm.frames, cls, base, m)
        else:
            for m in methods:
                if m in base.members:
                    if any((x.final for x in base.members[m].data)):
                        ctx.errorlog.overriding_final_method(ctx.vm.frames, cls, base, m)
                    ann = base.get_annotated_local(m)
                    if ann and ann.final:
                        ctx.errorlog.overriding_final_attribute(ctx.vm.frames, cls, base, m)
        methods.update(base.get_own_attributes())

def make_class(node, props, ctx):
    if False:
        return 10
    'Create a class with the name, bases and methods given.\n\n  Args:\n    node: The current CFG node.\n    props: class_mixin.ClassBuilderProperties required to build the class\n    ctx: The current context.\n\n  Returns:\n    A node and an instance of class_type.\n  '
    name = abstract_utils.get_atomic_python_constant(props.name_var)
    log.info('Declaring class %s', name)
    try:
        class_dict = abstract_utils.get_atomic_value(props.class_dict_var)
    except abstract_utils.ConversionError:
        log.error('Error initializing class %r', name)
        return ctx.convert.create_new_unknown(node)
    (metacls, bases) = _filter_out_metaclasses(props.bases, ctx)
    cls_var = metacls if metacls else props.metaclass_var
    bases = [_process_base_class(node, base, ctx) for base in bases]
    bases = _expand_generic_protocols(node, bases, ctx)
    if not bases:
        base = ctx.convert.object_type
        bases = [base.to_variable(ctx.root_node)]
    if isinstance(class_dict, abstract.Unsolvable) or not isinstance(class_dict, abstract.PythonConstant):
        var = ctx.new_unsolvable(node)
    else:
        if cls_var is None:
            cls_var = class_dict.members.get('__metaclass__')
            if cls_var:
                ctx.errorlog.ignored_metaclass(ctx.vm.frames, name, cls_var.data[0].full_name if cls_var.bindings else 'Any')
        if cls_var and all((v.data.full_name == 'builtins.type' for v in cls_var.bindings)):
            cls_var = None
        cls = abstract_utils.get_atomic_value(cls_var, default=ctx.convert.unsolvable) if cls_var else None
        if '__annotations__' not in class_dict.members and name in ctx.vm.annotated_locals:
            annotations_dict = ctx.vm.annotated_locals[name]
            if any((local.typ for local in annotations_dict.values())):
                annotations_member = abstract.AnnotationsDict(annotations_dict, ctx).to_variable(node)
                class_dict.members['__annotations__'] = annotations_member
                class_dict.pyval['__annotations__'] = annotations_member
        try:
            class_type = props.class_type or abstract.InterpreterClass
            assert issubclass(class_type, abstract.InterpreterClass)
            val = class_type(name, bases, class_dict.pyval, cls, ctx.vm.current_opcode, props.undecorated_methods, ctx)
            _check_final_members(val, class_dict.pyval, ctx)
            overriding_checks.check_overriding_members(val, bases, class_dict.pyval, ctx.matcher(node), ctx)
            val.decorators = props.decorators or []
        except mro.MROError as e:
            ctx.errorlog.mro_error(ctx.vm.frames, name, e.mro_seqs)
            var = ctx.new_unsolvable(node)
        except abstract_utils.GenericTypeError as e:
            ctx.errorlog.invalid_annotation(ctx.vm.frames, e.annot, e.error)
            var = ctx.new_unsolvable(node)
        else:
            var = props.new_class_var or ctx.program.NewVariable()
            var.AddBinding(val, props.class_dict_var.bindings, node)
            node = val.call_metaclass_init(node)
            node = val.call_init_subclass(node)
    ctx.vm.trace_opcode(None, name, var)
    return (node, var)

def _check_defaults(node, method, ctx):
    if False:
        print('Hello World!')
    'Check parameter defaults against annotations.'
    if not method.signature.has_param_annotations:
        return
    (_, args) = ctx.vm.create_method_arguments(node, method, use_defaults=True)
    try:
        (_, errors) = function.match_all_args(ctx, node, method, args)
    except function.FailedFunctionCall as e:
        raise AssertionError('Unexpected argument matching error: %s' % e.__class__.__name__) from e
    for (e, arg_name, value) in errors:
        bad_param = e.bad_call.bad_param
        expected_type = bad_param.typ
        if value == ctx.convert.ellipsis:
            should_report = not method.has_empty_body()
        else:
            should_report = True
        if should_report:
            ctx.errorlog.annotation_type_mismatch(ctx.vm.frames, expected_type, value.to_binding(node), arg_name, bad_param.error_details)

def make_function(name, node, code, globs, defaults, kw_defaults, closure, annotations, opcode, ctx):
    if False:
        i = 10
        return i + 15
    'Create a function or closure given the arguments.'
    if closure:
        closure = tuple((c for c in abstract_utils.get_atomic_python_constant(closure)))
        log.info('closure: %r', closure)
    if not name:
        name = abstract_utils.get_atomic_python_constant(code).qualname
    if not name:
        name = '<lambda>'
    val = abstract.InterpreterFunction.make(name, def_opcode=opcode, code=abstract_utils.get_atomic_python_constant(code), f_locals=ctx.vm.frame.f_locals, f_globals=globs, defaults=defaults, kw_defaults=kw_defaults, closure=closure, annotations=annotations, ctx=ctx)
    var = ctx.program.NewVariable()
    var.AddBinding(val, code.bindings, node)
    _check_defaults(node, val, ctx)
    if val.signature.annotations:
        ctx.vm.functions_type_params_check.append((val, ctx.vm.frame.current_opcode))
    return var

def update_excluded_types(node, ctx):
    if False:
        for i in range(10):
            print('nop')
    'Update the excluded_types attribute of functions in the current frame.'
    if not ctx.vm.frame.func:
        return
    func = ctx.vm.frame.func.data
    if isinstance(func, abstract.BoundFunction):
        func = func.underlying
    if not isinstance(func, abstract.InterpreterFunction):
        return
    for functions in ctx.vm.frame.functions_created_in_frame.values():
        for f in functions:
            f.signature.excluded_types |= func.signature.type_params
            func.signature.excluded_types |= f.signature.type_params
    for (name, local) in ctx.vm.current_annotated_locals.items():
        typ = local.get_type(node, name)
        if typ:
            func.signature.excluded_types.update((p.name for p in ctx.annotation_utils.get_type_parameters(typ)))

def push_block(state, t, level=None, *, index=0):
    if False:
        while True:
            i = 10
    if level is None:
        level = len(state.data_stack)
    return state.push_block(_Block(t, level, index))

def _base(cls):
    if False:
        i = 10
        return i + 15
    if isinstance(cls, abstract.ParameterizedClass):
        return cls.base_cls
    return cls

def _overrides(subcls, supercls, attr):
    if False:
        i = 10
        return i + 15
    'Check whether subcls_var overrides or newly defines the given attribute.\n\n  Args:\n    subcls: A potential subclass.\n    supercls: A potential superclass.\n    attr: An attribute name.\n\n  Returns:\n    True if subcls_var is a subclass of supercls_var and overrides or newly\n    defines the attribute. False otherwise.\n  '
    if subcls and supercls and (supercls in subcls.mro):
        subcls = _base(subcls)
        supercls = _base(supercls)
        for cls in subcls.mro:
            if cls == supercls:
                break
            if isinstance(cls, mixin.LazyMembers):
                cls.load_lazy_attribute(attr)
            if isinstance(cls, abstract.SimpleValue) and attr in cls.members and cls.members[attr].bindings:
                return True
    return False

def _call_binop_on_bindings(node, name, xval, yval, ctx):
    if False:
        for i in range(10):
            print('nop')
    'Call a binary operator on two cfg.Binding objects.'
    rname = slots.REVERSE_NAME_MAPPING.get(name)
    if rname and isinstance(xval.data, abstract.AMBIGUOUS_OR_EMPTY):
        return (node, ctx.program.NewVariable([ctx.convert.unsolvable], [xval, yval], node))
    options = [(xval, yval, name)]
    if rname:
        options.append((yval, xval, rname))
        if _overrides(yval.data.cls, xval.data.cls, rname):
            options.reverse()
    error = None
    for (left_val, right_val, attr_name) in options:
        if isinstance(left_val.data, abstract.Class) and attr_name == '__getitem__':
            valself = None
        else:
            valself = left_val
        (node, attr_var) = ctx.attribute_handler.get_attribute(node, left_val.data, attr_name, valself)
        if attr_var and attr_var.bindings:
            args = function.Args(posargs=(right_val.AssignToNewVariable(),))
            try:
                return function.call_function(ctx, node, attr_var, args, fallback_to_unsolvable=False, strict_filter=len(attr_var.bindings) > 1)
            except (function.DictKeyMissing, function.FailedFunctionCall) as e:
                if e > error:
                    error = e
    if error:
        raise error
    else:
        return (node, None)

def _get_annotation(node, var, ctx):
    if False:
        for i in range(10):
            print('nop')
    'Extract an annotation from terms in `a | b | ...`.'
    try:
        abstract_utils.get_atomic_python_constant(var, str)
    except abstract_utils.ConversionError:
        pass
    else:
        return None
    with ctx.errorlog.checkpoint() as record:
        annot = ctx.annotation_utils.extract_annotation(node, var, 'varname', ctx.vm.simple_stack())
    if record.errors:
        return None
    return annot

def _maybe_union(node, x, y, ctx):
    if False:
        print('Hello World!')
    "Attempt to evaluate a '|' operation as a typing.Union."
    x_annot = _get_annotation(node, x, ctx)
    y_annot = _get_annotation(node, y, ctx)
    opts = [x_annot, y_annot]
    if any((o is None for o in opts)):
        return None
    if all((isinstance(o, abstract.AMBIGUOUS) for o in opts)):
        return ctx.new_unsolvable(node)
    return abstract.Union(opts, ctx).to_variable(node)

def call_binary_operator(state, name, x, y, report_errors, ctx):
    if False:
        return 10
    'Map a binary operator to "magic methods" (__add__ etc.).'
    results = []
    log.debug('Calling binary operator %s', name)
    nodes = []
    error = None
    x = abstract_utils.simplify_variable(x, state.node, ctx)
    y = abstract_utils.simplify_variable(y, state.node, ctx)
    for xval in x.bindings:
        for yval in y.bindings:
            try:
                (node, ret) = _call_binop_on_bindings(state.node, name, xval, yval, ctx)
            except (function.DictKeyMissing, function.FailedFunctionCall) as e:
                if report_errors and e > error and state.node.HasCombination([xval, yval]):
                    error = e
            else:
                if ret:
                    nodes.append(node)
                    results.append(ret)
    if ctx.python_version >= (3, 10) and name == '__or__':
        fail = error or not results or isinstance(results[0].data[0], abstract.Unsolvable)
        if fail:
            ret = _maybe_union(state.node, x, y, ctx)
            if ret:
                return (state, ret)
    if nodes:
        state = state.change_cfg_node(ctx.join_cfg_nodes(nodes))
    result = ctx.join_variables(state.node, results)
    log.debug('Result: %r %r', result, result.data)
    log.debug('Error: %r', error)
    log.debug('Report Errors: %r', report_errors)
    if report_errors:
        if error is None:
            if not result.bindings:
                if ctx.options.report_errors:
                    ctx.errorlog.unsupported_operands(ctx.vm.frames, name, x, y)
                result = ctx.new_unsolvable(state.node)
        elif not result.bindings or ctx.options.strict_parameter_checks:
            if ctx.options.report_errors:
                ctx.errorlog.invalid_function_call(ctx.vm.frames, error)
            (state, result) = error.get_return(state)
    return (state, result)

def call_inplace_operator(state, iname, x, y, ctx):
    if False:
        return 10
    'Try to call a method like __iadd__, possibly fall back to __add__.'
    (state, attr) = ctx.vm.load_attr_noerror(state, x, iname)
    if attr is None:
        log.info('No inplace operator %s on %r', iname, x)
        name = iname.replace('i', '', 1)
        state = state.forward_cfg_node(f'BinOp:{name}')
        (state, ret) = call_binary_operator(state, name, x, y, report_errors=True, ctx=ctx)
    else:
        try:
            (state, ret) = ctx.vm.call_function_with_state(state, attr, (y,), fallback_to_unsolvable=False)
        except function.FailedFunctionCall as e:
            ctx.errorlog.invalid_function_call(ctx.vm.frames, e)
            (state, ret) = e.get_return(state)
    return (state, ret)

def check_for_deleted(state, name, var, ctx):
    if False:
        while True:
            i = 10
    for x in var.Data(state.node):
        if isinstance(x, abstract.Deleted):
            details = f'\nVariable {name} has been used after it has been deleted (line {x.line}).'
            ctx.errorlog.name_error(ctx.vm.frames, name, details=details)

def load_closure_cell(state, op, check_bindings, ctx):
    if False:
        print('Hello World!')
    'Retrieve the value out of a closure cell.\n\n  Used to generate the \'closure\' tuple for MAKE_CLOSURE.\n\n  Each entry in that tuple is typically retrieved using LOAD_CLOSURE.\n\n  Args:\n    state: The current VM state.\n    op: The opcode. op.arg is the index of a "cell variable": This corresponds\n      to an entry in co_cellvars or co_freevars and is a variable that\'s bound\n      into a closure.\n    check_bindings: Whether to check the retrieved value for bindings.\n    ctx: The current context.\n  Returns:\n    A new state.\n  '
    cell_index = ctx.vm.frame.f_code.get_cell_index(op.argval)
    cell = ctx.vm.frame.cells[cell_index]
    if check_bindings and (not cell.bindings):
        ctx.errorlog.name_error(ctx.vm.frames, op.argval)
        cell = ctx.new_unsolvable(state.node)
    visible_bindings = cell.Filter(state.node, strict=False)
    if len(visible_bindings) != len(cell.bindings):
        new_cell = ctx.program.NewVariable()
        if visible_bindings:
            for b in visible_bindings:
                new_cell.PasteBinding(b, state.node)
        else:
            new_cell.AddBinding(ctx.convert.unsolvable)
        ctx.vm.frame.cells[cell_index] = cell = new_cell
    name = op.argval
    ctx.vm.set_var_name(cell, name)
    check_for_deleted(state, name, cell, ctx)
    ctx.vm.trace_opcode(op, name, cell)
    return state.push(cell)

def jump_if(state, op, ctx, *, jump_if_val, pop=PopBehavior.NONE):
    if False:
        for i in range(10):
            print('nop')
    "Implementation of various _JUMP_IF bytecodes.\n\n  Args:\n    state: Initial FrameState.\n    op: An opcode.\n    ctx: The current context.\n    jump_if_val: Indicates what value leads to a jump. The non-jump state is\n      reached by the value's negation. Use frame_state.NOT_NONE for `not None`.\n    pop: Whether and how the opcode pops a value off the stack.\n  Returns:\n    The new FrameState.\n  "
    if pop is PopBehavior.ALWAYS:
        (state, value) = state.pop()
    else:
        value = state.top()
    if jump_if_val is None:
        normal_val = frame_state.NOT_NONE
    elif jump_if_val is frame_state.NOT_NONE:
        normal_val = None
    elif isinstance(jump_if_val, bool):
        normal_val = not jump_if_val
    else:
        raise NotImplementedError(f'Unsupported jump value: {jump_if_val!r}')
    jump = frame_state.restrict_condition(state.node, value, jump_if_val)
    normal = frame_state.restrict_condition(state.node, value, normal_val)
    if jump is not frame_state.UNSATISFIABLE:
        if jump:
            assert jump.binding
            else_state = state.forward_cfg_node('Jump', jump.binding).forward_cfg_node('Jump')
        else:
            else_state = state.forward_cfg_node('Jump')
        ctx.vm.store_jump(op.target, else_state)
    else:
        else_state = None
    if pop is PopBehavior.OR:
        state = state.pop_and_discard()
    if normal is frame_state.UNSATISFIABLE:
        return state.set_why('unsatisfiable')
    elif not else_state and (not normal):
        return state
    else:
        return state.forward_cfg_node('NoJump', normal.binding if normal else None)

def process_function_type_comment(node, op, func, ctx):
    if False:
        print('Hello World!')
    'Modifies annotations from a function type comment.\n\n  Checks if a type comment is present for the function.  If so, the type\n  comment is used to populate annotations.  It is an error to have\n  a type comment when annotations is not empty.\n\n  Args:\n    node: The current node.\n    op: An opcode (used to determine filename and line number).\n    func: An abstract.InterpreterFunction.\n    ctx: The current context.\n  '
    if not op.annotation:
        return
    (comment, lineno) = op.annotation
    if func.signature.annotations:
        ctx.errorlog.redundant_function_type_comment(op.code.filename, lineno)
        return
    fake_stack = ctx.vm.simple_stack(op.at_line(lineno))
    m = _FUNCTION_TYPE_COMMENT_RE.match(comment)
    if not m:
        ctx.errorlog.invalid_function_type_comment(fake_stack, comment)
        return
    (args, return_type) = m.groups()
    assert args is not None and return_type is not None
    if args != '...':
        annot = args.strip()
        try:
            ctx.annotation_utils.eval_multi_arg_annotation(node, func, annot, fake_stack)
        except abstract_utils.ConversionError:
            ctx.errorlog.invalid_function_type_comment(fake_stack, annot, details='Must be constant.')
    ret = ctx.convert.build_string(None, return_type)
    func.signature.set_annotation('return', ctx.annotation_utils.extract_annotation(node, ret, 'return', fake_stack))

def _merge_tuple_bindings(var, ctx):
    if False:
        i = 10
        return i + 15
    "Merge a set of heterogeneous tuples from var's bindings."
    if len(var.bindings) == 1:
        return var
    length = var.data[0].tuple_length
    seq = [ctx.program.NewVariable() for _ in range(length)]
    for tup in var.data:
        for i in range(length):
            seq[i].PasteVariable(tup.pyval[i])
    return seq

def _var_is_fixed_length_tuple(var: cfg.Variable) -> bool:
    if False:
        i = 10
        return i + 15
    return all((isinstance(d, abstract.Tuple) for d in var.data)) and all((d.tuple_length == var.data[0].tuple_length for d in var.data))

def _var_maybe_unknown(var: cfg.Variable) -> bool:
    if False:
        return 10
    return any((isinstance(x, abstract.Unsolvable) for x in var.data)) or all((isinstance(x, abstract.Unknown) for x in var.data))

def _convert_keys(keys_var: cfg.Variable):
    if False:
        for i in range(10):
            print('nop')
    keys = abstract_utils.get_atomic_python_constant(keys_var, tuple)
    return tuple(map(abstract_utils.get_atomic_python_constant, keys))

def match_sequence(obj_var: cfg.Variable) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'See if var is a sequence for pattern matching.'
    return abstract_utils.match_atomic_python_constant(obj_var, collections.abc.Iterable) or abstract_utils.is_var_indefinite_iterable(obj_var) or _var_is_fixed_length_tuple(obj_var) or _var_maybe_unknown(obj_var)

def match_mapping(node, obj_var: cfg.Variable, ctx) -> bool:
    if False:
        print('Hello World!')
    'See if var is a map for pattern matching.'
    mapping = ctx.convert.lookup_value('typing', 'Mapping')
    return abstract_utils.match_atomic_python_constant(obj_var, collections.abc.Mapping) or ctx.matcher(node).compute_one_match(obj_var, mapping).success or _var_maybe_unknown(obj_var)

def match_keys(node, obj_var: cfg.Variable, keys_var: cfg.Variable, ctx) -> Optional[cfg.Variable]:
    if False:
        while True:
            i = 10
    'Pick values out of a mapping for pattern matching.'
    keys = _convert_keys(keys_var)
    if _var_maybe_unknown(obj_var):
        return ctx.convert.build_tuple(node, [ctx.new_unsolvable(node) for _ in keys])
    try:
        mapping = abstract_utils.get_atomic_python_constant(obj_var, collections.abc.Mapping)
    except abstract_utils.ConversionError:
        ret = [ctx.program.NewVariable() for _ in keys]
        for d in obj_var.data:
            v = d.get_instance_type_parameter(abstract_utils.V)
            for x in ret:
                x.PasteVariable(v)
    else:
        try:
            ret = [mapping[k] for k in keys]
        except KeyError:
            return None
    return ctx.convert.build_tuple(node, ret)

@dataclasses.dataclass
class ClassMatch:
    success: Optional[bool]
    values: Optional[cfg.Variable]

    @property
    def matched(self):
        if False:
            for i in range(10):
                print('nop')
        return self.success is not False

def _match_builtin_class(node, success: Optional[bool], cls: abstract.Class, keys: Tuple[str], posarg_count: int, ctx) -> ClassMatch:
    if False:
        print('Hello World!')
    'Match a builtin class with a single posarg constructor.'
    assert success is not False
    if posarg_count > 1:
        ctx.errorlog.match_posargs_count(ctx.vm.frames, cls, posarg_count, 1)
        return ClassMatch(False, None)
    elif keys:
        return ClassMatch(False, None)
    else:
        ret = [cls.instantiate(node)]
        return ClassMatch(success, ctx.convert.build_tuple(node, ret))

def match_class(node, obj_var: cfg.Variable, cls_var: cfg.Variable, keys_var: cfg.Variable, posarg_count: int, ctx) -> ClassMatch:
    if False:
        i = 10
        return i + 15
    'Pick attributes out of a class instance for pattern matching.'
    keys = _convert_keys(keys_var)
    try:
        cls = abstract_utils.get_atomic_value(cls_var, (abstract.Class, abstract.AnnotationContainer))
    except abstract_utils.ConversionError:
        return ClassMatch(success=None, values=None)
    if isinstance(cls, abstract.AnnotationContainer):
        cls = cls.base_cls
    if _var_maybe_unknown(obj_var):
        instance_var = ctx.vm.init_class(node, cls)
        success = None
    else:
        m = ctx.matcher(node).compute_one_match(obj_var, cls, match_all_views=False)
        total = ctx.matcher(node).compute_one_match(obj_var, cls, match_all_views=True)
        if m.success:
            instance_var = obj_var
            success = True if total.success else None
        else:
            return ClassMatch(False, None)
    if posarg_count:
        if isinstance(cls, abstract.PyTDClass) and cls.name in _BUILTIN_MATCHERS:
            return _match_builtin_class(node, success, cls, keys, posarg_count, ctx)
        if posarg_count > len(cls.match_args):
            ctx.errorlog.match_posargs_count(ctx.vm.frames, cls, posarg_count, len(cls.match_args))
            return ClassMatch(False, None)
        keys = cls.match_args[:posarg_count] + keys
    ret = [ctx.program.NewVariable() for _ in keys]
    for (i, k) in enumerate(keys):
        for b in instance_var.bindings:
            (_, v) = ctx.attribute_handler.get_attribute(node, b.data, k, b)
            if not v:
                return ClassMatch(False, None)
            ret[i].PasteVariable(v)
    return ClassMatch(success, ctx.convert.build_tuple(node, ret))

def copy_dict_without_keys(node, obj_var: cfg.Variable, keys_var: cfg.Variable, ctx) -> cfg.Variable:
    if False:
        while True:
            i = 10
    'Create a copy of the input dict with some keys deleted.'
    if not all((abstract_utils.is_concrete_dict(x) for x in obj_var.data)):
        return obj_var
    keys = _convert_keys(keys_var)
    ret = abstract.Dict(ctx)
    for data in obj_var.data:
        for (k, v) in data.items():
            if k not in keys:
                ret.set_str_item(node, k, v)
    return ret.to_variable(node)

def unpack_iterable(node, var, ctx):
    if False:
        i = 10
        return i + 15
    'Unpack an iterable.'
    elements = []
    try:
        itr = abstract_utils.get_atomic_python_constant(var, collections.abc.Iterable)
    except abstract_utils.ConversionError:
        if abstract_utils.is_var_indefinite_iterable(var):
            elements.append(abstract.Splat(ctx, var).to_variable(node))
        elif _var_is_fixed_length_tuple(var):
            vs = _merge_tuple_bindings(var, ctx)
            elements.extend(vs)
        elif _var_maybe_unknown(var):
            v = ctx.convert.tuple_type.instantiate(node)
            elements.append(abstract.Splat(ctx, v).to_variable(node))
        else:
            elements.append(abstract.Splat(ctx, var).to_variable(node))
    else:
        for v in itr:
            if isinstance(v, cfg.Variable):
                elements.append(v)
            else:
                elements.append(ctx.convert.constant_to_var(v))
    return elements

def pop_and_unpack_list(state, count, ctx):
    if False:
        while True:
            i = 10
    'Pop count iterables off the stack and concatenate.'
    (state, iterables) = state.popn(count)
    elements = []
    for var in iterables:
        elements.extend(unpack_iterable(state.node, var, ctx))
    return (state, elements)

def merge_indefinite_iterables(node, target, iterables_to_merge):
    if False:
        i = 10
        return i + 15
    for var in iterables_to_merge:
        if abstract_utils.is_var_splat(var):
            for val in abstract_utils.unwrap_splat(var).data:
                p = val.get_instance_type_parameter(abstract_utils.T)
                target.merge_instance_type_parameter(node, abstract_utils.T, p)
        else:
            target.merge_instance_type_parameter(node, abstract_utils.T, var)

def unpack_and_build(state, count, build_concrete, container_type, ctx):
    if False:
        for i in range(10):
            print('nop')
    (state, seq) = pop_and_unpack_list(state, count, ctx)
    if any((abstract_utils.is_var_splat(x) for x in seq)):
        retval = abstract.Instance(container_type, ctx)
        merge_indefinite_iterables(state.node, retval, seq)
        ret = retval.to_variable(state.node)
    else:
        ret = build_concrete(state.node, seq)
    return state.push(ret)

def build_function_args_tuple(node, seq, ctx):
    if False:
        i = 10
        return i + 15
    tup = ctx.convert.tuple_to_value(seq)
    tup.is_unpacked_function_args = True
    return tup.to_variable(node)

def ensure_unpacked_starargs(node, starargs, ctx):
    if False:
        return 10
    'Unpack starargs if it has not been done already.'
    if not any((isinstance(x, abstract.Tuple) and x.is_unpacked_function_args for x in starargs.data)):
        seq = unpack_iterable(node, starargs, ctx)
        starargs = build_function_args_tuple(node, seq, ctx)
    return starargs

def build_map_unpack(state, arg_list, ctx):
    if False:
        print('Hello World!')
    'Merge a list of kw dicts into a single dict.'
    args = abstract.Dict(ctx)
    for arg in arg_list:
        for data in arg.data:
            args.update(state.node, data)
    args = args.to_variable(state.node)
    return args

def _binding_to_coroutine(state, b, bad_bindings, ret, top, ctx):
    if False:
        for i in range(10):
            print('nop')
    'Helper for _to_coroutine.\n\n  Args:\n    state: The current state.\n    b: A cfg.Binding.\n    bad_bindings: Bindings that are not coroutines.\n    ret: A return variable that this helper will add to.\n    top: Whether this is the top-level recursive call.\n    ctx: The current context.\n\n  Returns:\n    The state.\n  '
    if b not in bad_bindings:
        ret.PasteBinding(b)
        return state
    if ctx.matcher(state.node).match_var_against_type(b.variable, ctx.convert.generator_type, {}, {b.variable: b}) is not None:
        ret_param = b.data.get_instance_type_parameter(abstract_utils.V)
        coroutine = abstract.Coroutine(ctx, ret_param, state.node)
        ret.AddBinding(coroutine, [b], state.node)
        return state
    if not top:
        ret.PasteBinding(b)
        return state
    (_, await_method) = ctx.attribute_handler.get_attribute(state.node, b.data, '__await__', b)
    if await_method is None or not await_method.bindings:
        ret.PasteBinding(b)
        return state
    (state, await_obj) = ctx.vm.call_function_with_state(state, await_method, ())
    (state, subret) = to_coroutine(state, await_obj, False, ctx)
    ret.PasteVariable(subret)
    return state

def to_coroutine(state, obj, top, ctx):
    if False:
        return 10
    "Convert any awaitables and generators in obj to coroutines.\n\n  Implements the GET_AWAITABLE opcode, which returns obj unchanged if it is a\n  coroutine or generator and otherwise resolves obj.__await__\n  (https://docs.python.org/3/library/dis.html#opcode-GET_AWAITABLE). So that\n  we don't have to handle awaitable generators specially, our implementation\n  converts generators to coroutines.\n\n  Args:\n    state: The current state.\n    obj: The object, a cfg.Variable.\n    top: Whether this is the top-level recursive call, to prevent incorrectly\n      recursing into the result of obj.__await__.\n    ctx: The current context.\n\n  Returns:\n    A tuple of the state and a cfg.Variable of coroutines.\n  "
    bad_bindings = []
    for b in obj.bindings:
        if ctx.matcher(state.node).match_var_against_type(obj, ctx.convert.coroutine_type, {}, {obj: b}) is None:
            bad_bindings.append(b)
    if not bad_bindings:
        return (state, obj)
    ret = ctx.program.NewVariable()
    for b in obj.bindings:
        state = _binding_to_coroutine(state, b, bad_bindings, ret, top, ctx)
    return (state, ret)