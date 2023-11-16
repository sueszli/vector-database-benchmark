"""Code for checking and inferring types."""
import collections
import dataclasses
import enum
import logging
import re
from typing import Any, Dict, Sequence, Tuple, Union
import attrs
from pytype import state as frame_state
from pytype import vm
from pytype.abstract import abstract
from pytype.abstract import abstract_utils
from pytype.abstract import function
from pytype.overlays import special_builtins
from pytype.overlays import typing_overlay
from pytype.pytd import escape
from pytype.pytd import optimize
from pytype.pytd import pytd
from pytype.pytd import pytd_utils
from pytype.pytd import visitors
from pytype.typegraph import cfg
log = logging.getLogger(__name__)
_SKIP_FUNCTION_RE = re.compile('<(?!lambda)\\w+>$')
_InstanceCacheType = Dict[abstract.InterpreterClass, Dict[Any, Union['_InitClassState', cfg.Variable]]]

@dataclasses.dataclass(eq=True, frozen=True)
class _CallRecord:
    node: cfg.CFGNode
    function: cfg.Binding
    signatures: Sequence[abstract.PyTDSignature]
    positional_arguments: Tuple[Union[cfg.Binding, cfg.Variable], ...]
    keyword_arguments: Tuple[Tuple[str, Union[cfg.Binding, cfg.Variable]], ...]
    return_value: cfg.Variable

class _InitClassState(enum.Enum):
    INSTANTIATING = enum.auto()
    INITIALIZING = enum.auto()

class CallTracer(vm.VirtualMachine):
    """Virtual machine that records all function calls."""
    _CONSTRUCTORS = ('__new__', '__init__')

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)
        self._unknowns = {}
        self._calls = set()
        self._method_calls = set()
        self._instance_cache: _InstanceCacheType = collections.defaultdict(dict)
        self._initialized_instances = set()
        self._interpreter_functions = []
        self._interpreter_classes = []
        self._analyzed_functions = set()
        self._analyzed_classes = set()

    def create_varargs(self, node):
        if False:
            for i in range(10):
                print('nop')
        value = abstract.Instance(self.ctx.convert.tuple_type, self.ctx)
        value.merge_instance_type_parameter(node, abstract_utils.T, self.ctx.convert.create_new_unknown(node))
        return value.to_variable(node)

    def create_kwargs(self, node):
        if False:
            i = 10
            return i + 15
        key_type = self.ctx.convert.primitive_class_instances[str].to_variable(node)
        value_type = self.ctx.convert.create_new_unknown(node)
        kwargs = abstract.Instance(self.ctx.convert.dict_type, self.ctx)
        kwargs.merge_instance_type_parameter(node, abstract_utils.K, key_type)
        kwargs.merge_instance_type_parameter(node, abstract_utils.V, value_type)
        return kwargs.to_variable(node)

    def create_method_arguments(self, node, method, use_defaults=False):
        if False:
            i = 10
            return i + 15
        "Create arguments for the given method.\n\n    Creates Unknown objects as arguments for the given method. Note that we\n    don't need to take parameter annotations into account as\n    InterpreterFunction.call() will take care of that.\n\n    Args:\n      node: The current node.\n      method: An abstract.InterpreterFunction.\n      use_defaults: Whether to use parameter defaults for arguments. When True,\n        unknown arguments are created with force=False, as it is fine to use\n        Unsolvable rather than Unknown objects for type-checking defaults.\n\n    Returns:\n      A tuple of a node and a function.Args object.\n    "
        args = []
        num_posargs = method.argcount(node)
        num_posargs_no_default = num_posargs - len(method.defaults)
        for i in range(num_posargs):
            default_idx = i - num_posargs_no_default
            if use_defaults and default_idx >= 0:
                arg = method.defaults[default_idx]
            else:
                arg = self.ctx.convert.create_new_unknown(node, force=not use_defaults)
            args.append(arg)
        kws = {}
        for key in method.signature.kwonly_params:
            if use_defaults and key in method.kw_defaults:
                kws[key] = method.kw_defaults[key]
            else:
                kws[key] = self.ctx.convert.create_new_unknown(node, force=not use_defaults)
        starargs = self.create_varargs(node) if method.has_varargs() else None
        starstarargs = self.create_kwargs(node) if method.has_kwargs() else None
        return (node, function.Args(posargs=tuple(args), namedargs=kws, starargs=starargs, starstarargs=starstarargs))

    def call_function_with_args(self, node, val, args):
        if False:
            while True:
                i = 10
        'Call a function.\n\n    Args:\n      node: The given node.\n      val: A cfg.Binding containing the function.\n      args: A function.Args object.\n\n    Returns:\n      A tuple of (1) a node and (2) a cfg.Variable of the return value.\n    '
        assert isinstance(val.data, abstract.INTERPRETER_FUNCTION_TYPES)
        with val.data.record_calls():
            (new_node, ret) = self._call_function_in_frame(node, val, *attrs.astuple(args, recurse=False))
        return (new_node, ret)

    def _call_function_in_frame(self, node, val, args, kwargs, starargs, starstarargs):
        if False:
            print('Hello World!')
        fn = val.data
        if isinstance(fn, abstract.BoundInterpreterFunction):
            opcode = None
            f_globals = fn.underlying.f_globals
        else:
            assert isinstance(fn, abstract.InterpreterFunction)
            opcode = fn.def_opcode
            f_globals = fn.f_globals
        log.info('Analyzing function: %r', fn.name)
        state = frame_state.FrameState.init(node, self.ctx)
        frame = frame_state.SimpleFrame(node=node, opcode=opcode, f_globals=f_globals)
        frame.skip_in_tracebacks = True
        self.push_frame(frame)
        try:
            (state, ret) = self.call_function_with_state(state, val.AssignToNewVariable(node), args, kwargs, starargs, starstarargs)
        finally:
            self.pop_frame(frame)
        return (state.node, ret)

    def _maybe_fix_classmethod_cls_arg(self, node, cls, func, args):
        if False:
            return 10
        sig = func.signature
        if args.posargs and sig.param_names and (sig.param_names[0] not in sig.annotations):
            return args.replace(posargs=(cls.AssignToNewVariable(node),) + args.posargs[1:])
        else:
            return args

    def maybe_analyze_method(self, node, val, cls=None):
        if False:
            while True:
                i = 10
        method = val.data
        fname = val.data.name
        if isinstance(method, abstract.INTERPRETER_FUNCTION_TYPES):
            self._analyzed_functions.add(method.get_first_opcode())
            if not self.ctx.options.analyze_annotated and (method.signature.has_return_annotation or method.has_overloads) and (fname.rsplit('.', 1)[-1] not in self._CONSTRUCTORS):
                log.info('%r has annotations, not analyzing further.', fname)
            else:
                for f in method.iter_signature_functions():
                    (node, args) = self.create_method_arguments(node, f)
                    if f.is_classmethod and cls:
                        args = self._maybe_fix_classmethod_cls_arg(node, cls, f, args)
                    (node, _) = self.call_function_with_args(node, val, args)
        return node

    def call_with_fake_args(self, node0, funcv):
        if False:
            i = 10
            return i + 15
        'Attempt to call the given function with made-up arguments.'
        nodes = []
        rets = []
        for funcb in funcv.bindings:
            func = funcb.data
            log.info('Trying %s with fake arguments', func)
            if isinstance(func, abstract.INTERPRETER_FUNCTION_TYPES):
                (node1, args) = self.create_method_arguments(node0, func)
                (node2, ret) = function.call_function(self.ctx, node1, funcb.AssignToNewVariable(), args, fallback_to_unsolvable=False)
                nodes.append(node2)
                rets.append(ret)
        if nodes:
            ret = self.ctx.join_variables(node0, rets)
            node = self.ctx.join_cfg_nodes(nodes)
            if ret.bindings:
                return (node, ret)
        else:
            node = node0
        log.info('Unable to generate fake arguments for %s', funcv)
        return (node, self.ctx.new_unsolvable(node))

    def analyze_method_var(self, node0, name, var, cls):
        if False:
            i = 10
            return i + 15
        full_name = f'{cls.data.full_name}.{name}'
        if any((isinstance(v, abstract.INTERPRETER_FUNCTION_TYPES) for v in var.data)):
            log.info('Analyzing method: %r', full_name)
            node1 = self.ctx.connect_new_cfg_node(node0, f'Method:{full_name}')
        else:
            node1 = node0
        for val in var.bindings:
            node2 = self.maybe_analyze_method(node1, val, cls)
            node2.ConnectTo(node0)
        return node0

    def _bind_method(self, node, methodvar, instance_var):
        if False:
            while True:
                i = 10
        bound = self.ctx.program.NewVariable()
        for m in methodvar.Data(node):
            if isinstance(m, special_builtins.ClassMethodInstance):
                m = m.func.data[0]
                is_cls = True
            else:
                is_cls = isinstance(m, abstract.InterpreterFunction) and m.is_classmethod
            bound.AddBinding(m.property_get(instance_var, is_cls), [], node)
        return bound

    def _maybe_instantiate_binding_directly(self, node0, cls, container, instantiate_directly):
        if False:
            while True:
                i = 10
        (node1, new) = cls.data.get_own_new(node0, cls)
        if not new:
            instantiate_directly = True
        elif not instantiate_directly:
            instantiate_directly = any((not isinstance(f, abstract.InterpreterFunction) for f in new.data))
        if instantiate_directly:
            instance = cls.data.instantiate(node0, container=container)
        else:
            instance = None
        return (node1, new, instance)

    def _instantiate_binding(self, node0, cls, container, instantiate_directly):
        if False:
            while True:
                i = 10
        'Instantiate a class binding.'
        (node1, new, maybe_instance) = self._maybe_instantiate_binding_directly(node0, cls, container, instantiate_directly)
        if maybe_instance:
            return (node0, maybe_instance)
        instance = self.ctx.program.NewVariable()
        nodes = []
        for b in new.bindings:
            self._analyzed_functions.add(b.data.get_first_opcode())
            (node2, args) = self.create_method_arguments(node1, b.data)
            args = self._maybe_fix_classmethod_cls_arg(node0, cls, b.data, args)
            node3 = self.ctx.connect_new_cfg_node(node2, f'Call:{cls.data.name}.__new__')
            (node4, ret) = self.call_function_with_args(node3, b, args)
            instance.PasteVariable(ret)
            nodes.append(node4)
        return (self.ctx.join_cfg_nodes(nodes), instance)

    def _instantiate_var(self, node, clsv, container, instantiate_directly):
        if False:
            return 10
        'Build an (dummy) instance from a class, for analyzing it.'
        n = self.ctx.program.NewVariable()
        for cls in clsv.Bindings(node):
            (node, var) = self._instantiate_binding(node, cls, container, instantiate_directly)
            n.PasteVariable(var)
        return (node, n)

    def _mark_maybe_missing_members(self, values):
        if False:
            i = 10
            return i + 15
        'Set maybe_missing_members to True on these values and their type params.\n\n    Args:\n      values: A list of BaseValue objects. On every instance among the values,\n        recursively set maybe_missing_members to True on the instance and its\n        type parameters.\n    '
        values = list(values)
        seen = set()
        while values:
            v = values.pop(0)
            if v not in seen:
                seen.add(v)
                if isinstance(v, abstract.SimpleValue):
                    v.maybe_missing_members = True
                    for child in v.instance_type_parameters.values():
                        values.extend(child.data)

    def init_class_and_forward_node(self, node, cls, container=None, extra_key=None):
        if False:
            while True:
                i = 10
        "Instantiate a class, and also call __init__.\n\n    Calling __init__ can be expensive, so this method caches its created\n    instances. If you don't need __init__ called, use cls.instantiate instead.\n\n    Args:\n      node: The current node.\n      cls: The class to instantiate.\n      container: Optionally, a container to pass to the class's instantiate()\n        method, so that type parameters in the container's template are\n        instantiated to TypeParameterInstance.\n      extra_key: Optionally, extra information about the location at which the\n        instantion occurs. By default, this method keys on the current opcode\n        and the class, which sometimes isn't enough to disambiguate callers that\n        shouldn't get back the same cached instance.\n\n    Returns:\n      A tuple of node and instance variable.\n    "
        cls_key = cls.expr if cls.is_late_annotation() and (not cls.resolved) else cls
        cache = self._instance_cache[cls_key]
        key = (self.current_opcode, extra_key)
        status = instance = cache.get(key)
        if not instance or isinstance(instance, _InitClassState):
            clsvar = cls.to_variable(node)
            instantiate_directly = any((v is _InitClassState.INSTANTIATING for v in cache.values()))
            cache[key] = _InitClassState.INSTANTIATING
            (node, instance) = self._instantiate_var(node, clsvar, container, instantiate_directly)
            if instantiate_directly or status is _InitClassState.INITIALIZING:
                self._mark_maybe_missing_members(instance.data)
            else:
                cache[key] = _InitClassState.INITIALIZING
                node = self.call_init(node, instance)
            cache[key] = instance
        return (node, instance)

    def init_class(self, node, cls, container=None, extra_key=None):
        if False:
            i = 10
            return i + 15
        return self.init_class_and_forward_node(node, cls, container, extra_key)[-1]

    def get_bound_method(self, node, obj, method_name, valself):
        if False:
            i = 10
            return i + 15

        def bind(cur_node, m):
            if False:
                return 10
            return self._bind_method(cur_node, m, valself.AssignToNewVariable())
        (node, method) = self.ctx.attribute_handler.get_attribute(node, obj, method_name, valself)
        if not method:
            return (node, None)
        cls = valself.data.cls
        bound_method = bind(node, method) if obj == cls else method
        if not isinstance(cls, abstract.InterpreterClass) or any((isinstance(m, abstract.FUNCTION_TYPES) for m in bound_method.data)):
            return (node, bound_method)
        undecorated_method = cls.get_undecorated_method(method_name, node)
        if undecorated_method:
            return (node, bind(node, undecorated_method))
        else:
            return (node, bound_method)

    def _call_method(self, node, binding, method_name):
        if False:
            print('Hello World!')
        (node, bound_method) = self.get_bound_method(node, binding.data.cls, method_name, binding)
        if bound_method:
            return self.analyze_method_var(node, method_name, bound_method, binding.data.cls.to_binding(node))
        else:
            return node

    def _call_init_on_binding(self, node, b):
        if False:
            print('Hello World!')
        if isinstance(b.data, abstract.SimpleValue):
            for param in b.data.instance_type_parameters.values():
                node = self.call_init(node, param)
        node = self._call_method(node, b, '__init__')
        cls = b.data.cls
        if isinstance(cls, abstract.InterpreterClass):
            for method in cls.additional_init_methods:
                node = self._call_method(node, b, method)
        return node

    def call_init(self, node, instance):
        if False:
            while True:
                i = 10
        for b in instance.bindings:
            if b.data in self._initialized_instances:
                continue
            self._initialized_instances.add(b.data)
            node = self._call_init_on_binding(node, b)
        return node

    def reinitialize_if_initialized(self, node, instance):
        if False:
            print('Hello World!')
        if instance in self._initialized_instances:
            self._call_init_on_binding(node, instance.to_binding(node))

    def analyze_class(self, node, val):
        if False:
            return 10
        cls = val.data
        log.info('Analyzing class: %r', cls.full_name)
        self._analyzed_classes.add(cls)
        (node, instance) = self.init_class_and_forward_node(node, cls)
        good_instances = [b for b in instance.bindings if cls == b.data.cls]
        if not good_instances:
            instance = cls.instantiate(node)
            node = self.call_init(node, instance)
        elif len(good_instances) != len(instance.bindings):
            instance = self.ctx.join_bindings(node, good_instances)
        for instance_value in instance.data:
            cls.register_canonical_instance(instance_value)
        methods = sorted(cls.members.items())
        while methods:
            (name, methodvar) = methods.pop(0)
            if name in self._CONSTRUCTORS:
                continue
            for v in methodvar.data:
                if isinstance(v, special_builtins.PropertyInstance):
                    for m in (v.fget, v.fset, v.fdel):
                        if m:
                            methods.insert(0, (name, m))
                if not cls.is_abstract and isinstance(v, abstract.Function) and v.is_abstract:
                    unwrapped = abstract_utils.maybe_unwrap_decorated_function(v)
                    name = unwrapped.data[0].name if unwrapped else v.name
                    self.ctx.errorlog.ignored_abstractmethod(self.ctx.vm.simple_stack(cls.get_first_opcode()), cls.name, name)
            is_method_or_nested_class = any((isinstance(m, (abstract.FUNCTION_TYPES, abstract.InterpreterClass)) for m in methodvar.data))
            if self.ctx.options.bind_decorated_methods and (not is_method_or_nested_class) and (undecorated_method := cls.get_undecorated_method(name, node)):
                methodvar = undecorated_method
            b = self._bind_method(node, methodvar, instance)
            node = self.analyze_method_var(node, name, b, val)
        return node

    def analyze_function(self, node0, val):
        if False:
            return 10
        if val.data.is_attribute_of_class:
            log.info('Analyze functions: Skipping class method %s', val.data.name)
        else:
            node1 = self.ctx.connect_new_cfg_node(node0, f'Function:{val.data.name}')
            node2 = self.maybe_analyze_method(node1, val)
            node2.ConnectTo(node0)
        return node0

    def _should_analyze_as_interpreter_function(self, data):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(data, abstract.InterpreterFunction) and (not data.is_overload) and (not data.is_class_builder) and (data.get_first_opcode() not in self._analyzed_functions) and (not _SKIP_FUNCTION_RE.search(data.name))

    def analyze_toplevel(self, node, defs):
        if False:
            while True:
                i = 10
        for (name, var) in sorted(defs.items()):
            if not self._is_typing_member(name, var):
                for value in var.bindings:
                    if isinstance(value.data, abstract.InterpreterClass):
                        new_node = self.analyze_class(node, value)
                    elif isinstance(value.data, abstract.INTERPRETER_FUNCTION_TYPES) and (not value.data.is_overload):
                        new_node = self.analyze_function(node, value)
                    else:
                        continue
                    new_node.ConnectTo(node)
        for c in self._interpreter_classes:
            for value in c.bindings:
                if isinstance(value.data, abstract.InterpreterClass) and value.data not in self._analyzed_classes:
                    node = self.analyze_class(node, value)
        for f in self._interpreter_functions:
            for value in f.bindings:
                if self._should_analyze_as_interpreter_function(value.data):
                    node = self.analyze_function(node, value)
        for (func, opcode) in self.functions_type_params_check:
            func.signature.check_type_parameters(self.simple_stack(opcode), opcode, func.is_attribute_of_class)
        return node

    def analyze(self, node, defs, maximum_depth):
        if False:
            return 10
        assert not self.frame
        self._maximum_depth = maximum_depth
        self._analyzing = True
        node = node.ConnectNew(name='Analyze')
        return self.analyze_toplevel(node, defs)

    def trace_unknown(self, name, unknown_binding):
        if False:
            i = 10
            return i + 15
        self._unknowns[name] = unknown_binding

    def trace_call(self, node, func, sigs, posargs, namedargs, result):
        if False:
            print('Hello World!')
        'Add an entry into the call trace.\n\n    Args:\n      node: The CFG node right after this function call.\n      func: A cfg.Binding of a function that was called.\n      sigs: The signatures that the function might have been called with.\n      posargs: The positional arguments, an iterable over cfg.Variable.\n      namedargs: The keyword arguments, a dict mapping str to cfg.Variable.\n      result: A Variable of the possible result values.\n    '
        log.debug('Logging call to %r with %d args, return %r', func, len(posargs), result)
        args = tuple(posargs)
        kwargs = tuple((namedargs or {}).items())
        record = _CallRecord(node, func, sigs, args, kwargs, result)
        if isinstance(func.data, abstract.BoundPyTDFunction):
            self._method_calls.add(record)
        elif isinstance(func.data, abstract.PyTDFunction):
            self._calls.add(record)

    def trace_functiondef(self, f):
        if False:
            while True:
                i = 10
        self._interpreter_functions.append(f)

    def trace_classdef(self, c):
        if False:
            for i in range(10):
                print('nop')
        self._interpreter_classes.append(c)

    def pytd_classes_for_unknowns(self):
        if False:
            for i in range(10):
                print('nop')
        classes = []
        for (name, val) in self._unknowns.items():
            log.info('Generating structural definition for unknown: %r', name)
            if val in val.variable.Filter(self.ctx.exitpoint, strict=False):
                classes.append(val.data.to_structural_def(self.ctx.exitpoint, name))
        return classes

    def _skip_definition_export(self, name, var):
        if False:
            return 10
        return name in abstract_utils.TOP_LEVEL_IGNORE or self._is_typing_member(name, var) or self._is_future_feature(name, var)

    def pytd_for_types(self, defs):
        if False:
            while True:
                i = 10
        annotated_names = set()
        data = []
        annots = abstract_utils.get_annotations_dict(defs)
        for (name, t) in self.ctx.pytd_convert.annotations_to_instance_types(self.ctx.exitpoint, annots):
            annotated_names.add(name)
            data.append(pytd.Constant(name, t))
        for (name, var) in defs.items():
            if name in annotated_names or self._skip_definition_export(name, var):
                continue
            log.info('Generating pytd type for top-level definition: %r', name)
            if any((v == self.ctx.convert.unsolvable for v in var.Data(self.ctx.exitpoint))):
                options = [self.ctx.convert.unsolvable]
            else:
                options = var.FilteredData(self.ctx.exitpoint, strict=False)
            if len(options) > 1 and (not all((isinstance(o, abstract.FUNCTION_TYPES) for o in options))):
                if all((isinstance(o, abstract.TypeParameter) for o in options)):
                    pytd_def = pytd_utils.JoinTypes((t.to_pytd_def(self.ctx.exitpoint, name) for t in options))
                    if isinstance(pytd_def, pytd.TypeParameter):
                        data.append(pytd_def)
                    else:
                        data.append(pytd.Constant(name, pytd.AnythingType()))
                elif all((isinstance(o, (abstract.ParameterizedClass, abstract.Union)) for o in options)):
                    pytd_def = pytd_utils.JoinTypes((t.to_pytd_def(self.ctx.exitpoint, name).type for t in options))
                    data.append(pytd.Alias(name, pytd_def))
                else:
                    combined_types = pytd_utils.JoinTypes((t.to_type(self.ctx.exitpoint) for t in options))
                    data.append(pytd.Constant(name, combined_types))
            elif options:
                for option in options:
                    try:
                        should_optimize = not isinstance(option, abstract.FUNCTION_TYPES)
                        with self.ctx.pytd_convert.optimize_literals(should_optimize):
                            d = option.to_pytd_def(self.ctx.exitpoint, name)
                    except NotImplementedError:
                        with self.ctx.pytd_convert.optimize_literals():
                            d = option.to_type(self.ctx.exitpoint)
                        if isinstance(d, pytd.NothingType):
                            if isinstance(option, abstract.Empty):
                                d = pytd.AnythingType()
                            else:
                                assert isinstance(option, typing_overlay.Never)
                    if isinstance(d, pytd.Type) and (not isinstance(d, pytd.TypeParameter)):
                        data.append(pytd.Constant(name, d))
                    else:
                        data.append(d)
            else:
                log.error('No visible options for %s', name)
                data.append(pytd.Constant(name, pytd.AnythingType()))
        return pytd_utils.WrapTypeDeclUnit('inferred', data)

    @staticmethod
    def _call_traces_to_function(call_traces, name_transform=lambda x: x):
        if False:
            while True:
                i = 10
        funcs = collections.defaultdict(pytd_utils.OrderedSet)

        def to_type(node, arg):
            if False:
                for i in range(10):
                    print('nop')
            return pytd_utils.JoinTypes((a.to_type(node) for a in arg.data))
        for ct in call_traces:
            log.info('Generating pytd function for call trace: %r', ct.function.data.name)
            arg_names = max((sig.get_positional_names() for sig in ct.signatures), key=len)
            for i in range(len(arg_names)):
                if not isinstance(ct.function.data, abstract.BoundFunction) or i > 0:
                    arg_names[i] = function.argname(i)
            arg_types = []
            for arg in ct.positional_arguments:
                arg_types.append(to_type(ct.node, arg))
            kw_types = []
            for (name, arg) in ct.keyword_arguments:
                kw_types.append((name, to_type(ct.node, arg)))
            ret = pytd_utils.JoinTypes((t.to_type(ct.node) for t in ct.return_value.data))
            starargs = None
            starstarargs = None
            funcs[ct.function.data.name].add(pytd.Signature(tuple((pytd.Parameter(n, t, pytd.ParameterKind.REGULAR, False, None) for (n, t) in zip(arg_names, arg_types))) + tuple((pytd.Parameter(n, t, pytd.ParameterKind.REGULAR, False, None) for (n, t) in kw_types)), starargs, starstarargs, ret, exceptions=(), template=()))
        functions = []
        for (name, signatures) in funcs.items():
            functions.append(pytd.Function(name_transform(name), tuple(signatures), pytd.MethodKind.METHOD))
        return functions

    def _is_typing_member(self, name, var):
        if False:
            return 10
        for module_name in ('typing', 'typing_extensions'):
            if module_name not in self.loaded_overlays:
                continue
            overlay = self.loaded_overlays[module_name]
            if overlay:
                module = overlay.get_module(name)
                if name in module.members and module.members[name].data == var.data:
                    return True
        return False

    def _is_future_feature(self, name, var):
        if False:
            for i in range(10):
                print('nop')
        for v in var.data:
            if isinstance(v, abstract.Instance) and v.cls.module == '__future__':
                return True
        return False

    def pytd_functions_for_call_traces(self):
        if False:
            return 10
        return self._call_traces_to_function(self._calls, escape.pack_partial)

    def pytd_classes_for_call_traces(self):
        if False:
            for i in range(10):
                print('nop')
        class_to_records = collections.defaultdict(list)
        for call_record in self._method_calls:
            args = call_record.positional_arguments
            unknown = False
            for arg in args:
                if any((isinstance(a, abstract.Unknown) for a in arg.data)):
                    unknown = True
            if not unknown:
                continue
            classes = args[0].data
            for cls in classes:
                if isinstance(cls.cls, abstract.PyTDClass):
                    class_to_records[cls].append(call_record)
        classes = []
        for (cls, call_records) in class_to_records.items():
            full_name = cls.module + '.' + cls.name if cls.module else cls.name
            log.info('Generating pytd class for call trace: %r', full_name)
            classes.append(pytd.Class(name=escape.pack_partial(full_name), keywords=(), bases=(pytd.NamedType('builtins.object'),), methods=tuple(self._call_traces_to_function(call_records)), constants=(), classes=(), decorators=(), slots=None, template=()))
        return classes

    def compute_types(self, defs):
        if False:
            while True:
                i = 10
        classes = tuple(self.pytd_classes_for_unknowns()) + tuple(self.pytd_classes_for_call_traces())
        functions = tuple(self.pytd_functions_for_call_traces())
        aliases = ()
        ty = pytd_utils.Concat(self.pytd_for_types(defs), pytd_utils.CreateModule('unknowns', classes=classes, functions=functions, aliases=aliases))
        ty = ty.Visit(optimize.CombineReturnsAndExceptions())
        ty = ty.Visit(optimize.PullInMethodClasses())
        ty = ty.Visit(visitors.DefaceUnresolved([ty, self.ctx.loader.concat_all()], escape.UNKNOWN))
        return ty.Visit(visitors.AdjustTypeParameters())

    def _check_return(self, node, actual, formal):
        if False:
            print('Hello World!')
        if not self.ctx.options.report_errors:
            return True
        if formal.full_name == 'typing.TypeGuard':
            expected = self.ctx.convert.bool_type
        else:
            expected = formal
        match_result = self.ctx.matcher(node).compute_one_match(actual, expected)
        if not match_result.success:
            self.ctx.errorlog.bad_return_type(self.frames, node, match_result.bad_matches)
        return match_result.success