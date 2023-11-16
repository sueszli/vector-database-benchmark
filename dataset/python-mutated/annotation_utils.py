"""Utilities for inline type annotations."""
import collections
import dataclasses
import itertools
from typing import Any, Dict, Optional, Sequence, Set, Tuple
from pytype import state
from pytype import utils
from pytype.abstract import abstract
from pytype.abstract import abstract_utils
from pytype.abstract import function
from pytype.abstract import mixin
from pytype.overlays import typing_overlay
from pytype.pytd import pytd_utils
from pytype.typegraph import cfg

@dataclasses.dataclass
class AnnotatedValue:
    typ: Any
    value: Any
    final: bool = False

class AnnotationUtils(utils.ContextWeakrefMixin):
    """Utility class for inline type annotations."""

    def sub_annotations(self, node, annotations, substs, instantiate_unbound):
        if False:
            print('Hello World!')
        'Apply type parameter substitutions to a dictionary of annotations.'
        if substs and all(substs):
            return {name: self.sub_one_annotation(node, annot, substs, instantiate_unbound) for (name, annot) in annotations.items()}
        return annotations

    def _get_type_parameter_subst(self, node: cfg.CFGNode, annot: abstract.TypeParameter, substs: Sequence[Dict[str, cfg.Variable]], instantiate_unbound: bool) -> abstract.BaseValue:
        if False:
            i = 10
            return i + 15
        'Helper for sub_one_annotation.'
        if all((annot.full_name in subst and subst[annot.full_name].bindings for subst in substs)):
            vals = sum((subst[annot.full_name].data for subst in substs), [])
        else:
            vals = None
        if vals is None or any((isinstance(v, abstract.AMBIGUOUS) for v in vals)) or all((isinstance(v, abstract.Empty) for v in vals)):
            if instantiate_unbound:
                vals = annot.instantiate(node).data
            else:
                vals = [annot]
        return self.ctx.convert.merge_classes(vals)

    def sub_one_annotation(self, node, annot, substs, instantiate_unbound=True):
        if False:
            return 10

        def get_type_parameter_subst(annotation):
            if False:
                return 10
            return self._get_type_parameter_subst(node, annotation, substs, instantiate_unbound)
        return self._do_sub_one_annotation(node, annot, get_type_parameter_subst)

    def _do_sub_one_annotation(self, node, annot, get_type_parameter_subst_fn):
        if False:
            return 10
        'Apply type parameter substitutions to an annotation.'
        stack = [(annot, None)]
        late_annotations = {}
        done = []
        while stack:
            (cur, inner_type_keys) = stack.pop()
            if not cur.formal:
                done.append(cur)
            elif isinstance(cur, mixin.NestedAnnotation):
                if cur.is_late_annotation() and any((t[0] == cur for t in stack)):
                    if cur not in late_annotations:
                        param_strings = []
                        for t in utils.unique_list(self.get_type_parameters(cur)):
                            s = pytd_utils.Print(get_type_parameter_subst_fn(t).get_instance_type(node))
                            param_strings.append(s)
                        expr = f"{cur.expr}[{', '.join(param_strings)}]"
                        late_annot = abstract.LateAnnotation(expr, cur.stack, cur.ctx)
                        late_annotations[cur] = late_annot
                    done.append(late_annotations[cur])
                elif inner_type_keys is None:
                    (keys, vals) = zip(*cur.get_inner_types())
                    stack.append((cur, keys))
                    stack.extend(((val, None) for val in vals))
                else:
                    inner_types = []
                    for k in inner_type_keys:
                        inner_types.append((k, done.pop()))
                    done_annot = cur.replace(inner_types)
                    if cur in late_annotations:
                        late_annot = late_annotations.pop(cur)
                        late_annot.set_type(done_annot)
                        if '[' in late_annot.expr:
                            if self.ctx.vm.late_annotations is None:
                                self.ctx.vm.flatten_late_annotation(node, late_annot, self.ctx.vm.frame.f_globals)
                            else:
                                self.ctx.vm.late_annotations[late_annot.expr.split('[', 1)[0]].append(late_annot)
                    done.append(done_annot)
            else:
                done.append(get_type_parameter_subst_fn(cur))
        assert len(done) == 1
        return done[0]

    def sub_annotations_for_parameterized_class(self, cls: abstract.ParameterizedClass, annotations: Dict[str, abstract.BaseValue]) -> Dict[str, abstract.BaseValue]:
        if False:
            for i in range(10):
                print('nop')
        'Apply type parameter substitutions to a dictionary of annotations.\n\n    Args:\n      cls: ParameterizedClass that defines type parameter substitutions.\n      annotations: A dictionary of annotations to which type parameter\n        substition should be applied.\n\n    Returns:\n      Annotations with type parameters substituted.\n    '
        formal_type_parameters = cls.get_formal_type_parameters()

        def get_type_parameter_subst(annotation: abstract.TypeParameter) -> Optional[abstract.BaseValue]:
            if False:
                while True:
                    i = 10
            for name in (f'{cls.full_name}.{annotation.name}', f'{cls.name}.{annotation.name}'):
                if name in formal_type_parameters:
                    return formal_type_parameters[name]
            return annotation
        return {name: self._do_sub_one_annotation(self.ctx.root_node, annot, get_type_parameter_subst) for (name, annot) in annotations.items()}

    def get_late_annotations(self, annot):
        if False:
            while True:
                i = 10
        if annot.is_late_annotation() and (not annot.resolved):
            yield annot
        elif isinstance(annot, mixin.NestedAnnotation):
            for (_, typ) in annot.get_inner_types():
                yield from self.get_late_annotations(typ)

    def add_scope(self, annot, types, cls, seen=None):
        if False:
            i = 10
            return i + 15
        'Add scope for type parameters.\n\n    In original type class, all type parameters that should be added a scope\n    will be replaced with a new copy.\n\n    Args:\n      annot: The type class.\n      types: A type name list that should be added a scope.\n      cls: The class that type parameters should be scoped to.\n      seen: Already seen types.\n\n    Returns:\n      The type with fresh type parameters that have been added the scope.\n    '
        if seen is None:
            seen = {annot}
        elif annot in seen or not annot.formal:
            return annot
        else:
            seen.add(annot)
        if isinstance(annot, abstract.TypeParameter):
            if annot.name in types:
                return annot.with_scope(cls.full_name)
            elif annot.full_name == 'typing.Self':
                bound_annot = annot.copy()
                bound_annot.bound = cls
                return bound_annot
            else:
                return annot
        elif isinstance(annot, mixin.NestedAnnotation):
            inner_types = [(key, self.add_scope(typ, types, cls, seen)) for (key, typ) in annot.get_inner_types()]
            return annot.replace(inner_types)
        return annot

    def get_type_parameters(self, annot, seen=None):
        if False:
            i = 10
            return i + 15
        'Returns all the TypeParameter instances that appear in the annotation.\n\n    Note that if you just need to know whether or not the annotation contains\n    type parameters, you can check its `.formal` attribute.\n\n    Args:\n      annot: An annotation.\n      seen: A seen set.\n    '
        seen = seen or set()
        if annot in seen or not annot.formal:
            return []
        if isinstance(annot, mixin.NestedAnnotation):
            seen = seen | {annot}
        if isinstance(annot, abstract.TypeParameter):
            return [annot]
        elif isinstance(annot, abstract.TupleClass):
            annots = []
            for idx in range(annot.tuple_length):
                annots.extend(self.get_type_parameters(annot.formal_type_parameters[idx], seen))
            return annots
        elif isinstance(annot, mixin.NestedAnnotation):
            return sum((self.get_type_parameters(t, seen) for (_, t) in annot.get_inner_types()), [])
        return []

    def get_callable_type_parameter_names(self, val: abstract.BaseValue):
        if False:
            while True:
                i = 10
        "Gets all TypeParameter names that appear in a Callable in 'val'."
        type_params = set()
        seen = set()
        stack = [val]
        while stack:
            annot = stack.pop()
            if annot in seen or not annot.formal:
                continue
            seen.add(annot)
            if annot.full_name == 'typing.Callable':
                params = collections.Counter(self.get_type_parameters(annot))
                if isinstance(annot, abstract.CallableClass):
                    params -= collections.Counter(self.get_type_parameters(annot.formal_type_parameters[abstract_utils.ARGS]))
                type_params.update((p.name for (p, n) in params.items() if n > 1))
            elif isinstance(annot, mixin.NestedAnnotation):
                stack.extend((v for (_, v) in annot.get_inner_types()))
        return type_params

    def convert_function_type_annotation(self, name, typ):
        if False:
            i = 10
            return i + 15
        visible = typ.data
        if len(visible) > 1:
            self.ctx.errorlog.ambiguous_annotation(self.ctx.vm.frames, visible, name)
            return None
        else:
            return visible[0]

    def convert_function_annotations(self, node, raw_annotations):
        if False:
            print('Hello World!')
        'Convert raw annotations to a {name: annotation} dict.'
        if raw_annotations:
            names = abstract_utils.get_atomic_python_constant(raw_annotations[-1])
            type_list = raw_annotations[:-1]
            annotations_list = []
            for (name, t) in zip(names, type_list):
                name = abstract_utils.get_atomic_python_constant(name)
                t = self.convert_function_type_annotation(name, t)
                annotations_list.append((name, t))
            return self.convert_annotations_list(node, annotations_list)
        else:
            return {}

    def convert_annotations_list(self, node, annotations_list):
        if False:
            print('Hello World!')
        'Convert a (name, raw_annot) list to a {name: annotation} dict.'
        annotations = {}
        for (name, t) in annotations_list:
            if t is None or abstract_utils.is_ellipsis(t):
                continue
            annot = self._process_one_annotation(node, t, name, self.ctx.vm.simple_stack())
            if annot is not None:
                annotations[name] = annot
        return annotations

    def convert_class_annotations(self, node, raw_annotations):
        if False:
            i = 10
            return i + 15
        'Convert a name -> raw_annot dict to annotations.'
        annotations = {}
        raw_items = raw_annotations.items()
        for (name, t) in raw_items:
            annot = self._process_one_annotation(node, t, None, self.ctx.vm.simple_stack())
            annotations[name] = annot or self.ctx.convert.unsolvable
        return annotations

    def init_annotation(self, node, name, annot, container=None, extra_key=None):
        if False:
            while True:
                i = 10
        value = self.ctx.vm.init_class(node, annot, container=container, extra_key=extra_key)
        for d in value.data:
            d.from_annotation = name
        return (node, value)

    def _in_class_frame(self):
        if False:
            print('Hello World!')
        frame = self.ctx.vm.frame
        if not frame.func:
            return False
        return isinstance(frame.func.data, abstract.BoundFunction) or frame.func.data.is_attribute_of_class

    def extract_and_init_annotation(self, node, name, var):
        if False:
            for i in range(10):
                print('nop')
        'Extracts an annotation from var and instantiates it.'
        frame = self.ctx.vm.frame
        substs = frame.substs
        if self._in_class_frame():
            self_var = frame.first_arg
            if self_var:
                (defining_cls_name, _, _) = frame.func.data.name.rpartition('.')
                type_params = []
                defining_classes = []
                for v in self_var.data:
                    v_cls = v if isinstance(v, abstract.Class) else v.cls
                    for cls in v_cls.mro:
                        if cls.name == defining_cls_name:
                            type_params.extend((p.with_scope(None) for p in cls.template))
                            defining_classes.append(cls)
                            break
                self_substs = tuple((abstract_utils.get_type_parameter_substitutions(cls, type_params) for cls in defining_classes))
                substs = abstract_utils.combine_substs(substs, self_substs)
        typ = self.extract_annotation(node, var, name, self.ctx.vm.simple_stack(), allowed_type_params=set(itertools.chain(*substs)))
        if isinstance(typ, typing_overlay.Final):
            return (typ, self.ctx.new_unsolvable(node))
        return self._sub_and_instantiate(node, name, typ, substs)

    def _sub_and_instantiate(self, node, name, typ, substs):
        if False:
            print('Hello World!')
        if isinstance(typ, abstract.FinalAnnotation):
            (t, value) = self._sub_and_instantiate(node, name, typ.annotation, substs)
            return (abstract.FinalAnnotation(t, self.ctx), value)
        if typ.formal:
            substituted_type = self.sub_one_annotation(node, typ, substs, instantiate_unbound=False)
        else:
            substituted_type = typ
        if typ.formal and self._in_class_frame():
            class_substs = abstract_utils.combine_substs(substs, [{'typing.Self': self.ctx.vm.frame.first_arg}])
            type_for_value = self.sub_one_annotation(node, typ, class_substs, instantiate_unbound=False)
        else:
            type_for_value = substituted_type
        (_, value) = self.init_annotation(node, name, type_for_value)
        return (substituted_type, value)

    def apply_annotation(self, node, op, name, value):
        if False:
            return 10
        'If there is an annotation for the op, return its value.'
        assert op is self.ctx.vm.frame.current_opcode
        if op.code.filename != self.ctx.vm.filename:
            return AnnotatedValue(None, value)
        if not op.annotation:
            return AnnotatedValue(None, value)
        annot = op.annotation
        if annot == '...':
            return AnnotatedValue(None, value)
        frame = self.ctx.vm.frame
        stack = self.ctx.vm.simple_stack()
        with self.ctx.vm.generate_late_annotations(stack):
            (var, errorlog) = abstract_utils.eval_expr(self.ctx, node, frame.f_globals, frame.f_locals, annot)
        if errorlog:
            self.ctx.errorlog.invalid_annotation(self.ctx.vm.frames, annot, details=errorlog.details)
        (typ, annot_val) = self.extract_and_init_annotation(node, name, var)
        if isinstance(typ, typing_overlay.Final):
            return AnnotatedValue(None, value, final=True)
        elif isinstance(typ, abstract.FinalAnnotation):
            return AnnotatedValue(typ.annotation, annot_val, final=True)
        elif typ.full_name == 'typing.TypeAlias':
            annot = self.extract_annotation(node, value, name, stack)
            return AnnotatedValue(None, annot.to_variable(node))
        else:
            return AnnotatedValue(typ, annot_val)

    def extract_annotation(self, node, var, name, stack, allowed_type_params: Optional[Set[str]]=None):
        if False:
            print('Hello World!')
        "Returns an annotation extracted from 'var'.\n\n    Args:\n      node: The current node.\n      var: The variable to extract from.\n      name: The annotated name.\n      stack: The frame stack.\n      allowed_type_params: Type parameters that are allowed to appear in the\n        annotation. 'None' means all are allowed. If non-None, the result of\n        calling get_callable_type_parameter_names on the extracted annotation is\n        also added to the allowed set.\n    "
        try:
            typ = abstract_utils.get_atomic_value(var)
        except abstract_utils.ConversionError:
            self.ctx.errorlog.ambiguous_annotation(self.ctx.vm.frames, None, name)
            return self.ctx.convert.unsolvable
        typ = self._process_one_annotation(node, typ, name, stack)
        if not typ:
            return self.ctx.convert.unsolvable
        if typ.formal and allowed_type_params is not None:
            allowed_type_params = allowed_type_params | self.get_callable_type_parameter_names(typ)
            if self.ctx.vm.frame.func and (isinstance(self.ctx.vm.frame.func.data, abstract.BoundFunction) or self.ctx.vm.frame.func.data.is_class_builder):
                allowed_type_params.add('typing.Self')
            illegal_params = []
            for x in self.get_type_parameters(typ):
                if not allowed_type_params.intersection([x.name, x.full_name]):
                    illegal_params.append(x.name)
            if illegal_params:
                self._log_illegal_params(illegal_params, stack, typ, name)
                return self.ctx.convert.unsolvable
        return typ

    def _log_illegal_params(self, illegal_params, stack, typ, name):
        if False:
            print('Hello World!')
        out_of_scope_params = utils.unique_list(illegal_params)
        details = 'TypeVar(s) %s not in scope' % ', '.join((repr(p) for p in out_of_scope_params))
        if self.ctx.vm.frame.func:
            method = self.ctx.vm.frame.func.data
            if isinstance(method, abstract.BoundFunction):
                desc = 'class'
                frame_name = method.name.rsplit('.', 1)[0]
            else:
                desc = 'class' if method.is_class_builder else 'method'
                frame_name = method.name
            details += f' for {desc} {frame_name!r}'
        if 'AnyStr' in out_of_scope_params:
            str_type = 'Union[str, bytes]'
            details += f'\nNote: For all string types, use {str_type}.'
        self.ctx.errorlog.invalid_annotation(stack, typ, details, name)

    def eval_multi_arg_annotation(self, node, func, annot, stack):
        if False:
            print('Hello World!')
        'Evaluate annotation for multiple arguments (from a type comment).'
        (args, errorlog) = self._eval_expr_as_tuple(node, annot, stack)
        if errorlog:
            self.ctx.errorlog.invalid_function_type_comment(stack, annot, details=errorlog.details)
        code = func.code
        expected = code.get_arg_count()
        names = code.varnames
        if len(args) != expected:
            if expected and names[0] in ['self', 'cls']:
                expected -= 1
                names = names[1:]
        if len(args) != expected:
            self.ctx.errorlog.invalid_function_type_comment(stack, annot, details='Expected %d args, %d given' % (expected, len(args)))
            return
        for (name, arg) in zip(names, args):
            resolved = self._process_one_annotation(node, arg, name, stack)
            if resolved is not None:
                func.signature.set_annotation(name, resolved)

    def _process_one_annotation(self, node: cfg.CFGNode, annotation: abstract.BaseValue, name: Optional[str], stack: Tuple[state.FrameType, ...]) -> Optional[abstract.BaseValue]:
        if False:
            while True:
                i = 10
        'Change annotation / record errors where required.'
        if isinstance(annotation, abstract.AnnotationContainer):
            annotation = annotation.base_cls
        if isinstance(annotation, typing_overlay.Union):
            self.ctx.errorlog.invalid_annotation(stack, annotation, 'Needs options', name)
            return None
        elif name is not None and name != 'return' and (annotation.full_name == 'typing.TypeGuard'):
            self.ctx.errorlog.invalid_annotation(stack, annotation, f'{annotation.name} is only allowed as a return annotation', name)
            return None
        elif isinstance(annotation, abstract.Instance) and annotation.cls == self.ctx.convert.str_type:
            if isinstance(annotation, abstract.PythonConstant):
                expr = annotation.pyval
                if not expr:
                    self.ctx.errorlog.invalid_annotation(stack, annotation, 'Cannot be an empty string', name)
                    return None
                frame = self.ctx.vm.frame
                with self.ctx.vm.generate_late_annotations(stack):
                    (v, errorlog) = abstract_utils.eval_expr(self.ctx, node, frame.f_globals, frame.f_locals, expr)
                if errorlog:
                    self.ctx.errorlog.copy_from(errorlog.errors, stack)
                if len(v.data) == 1:
                    return self._process_one_annotation(node, v.data[0], name, stack)
            self.ctx.errorlog.ambiguous_annotation(stack, [annotation], name)
            return None
        elif annotation.cls == self.ctx.convert.none_type:
            return self.ctx.convert.none_type
        elif isinstance(annotation, mixin.NestedAnnotation):
            if annotation.processed:
                return annotation
            annotation.processed = True
            for (key, typ) in annotation.get_inner_types():
                if annotation.full_name == 'typing.Callable' and key == abstract_utils.RET:
                    inner_name = 'return'
                else:
                    inner_name = name
                processed = self._process_one_annotation(node, typ, inner_name, stack)
                if processed is None:
                    return None
                elif name == inner_name and processed.full_name == 'typing.TypeGuard':
                    self.ctx.errorlog.invalid_annotation(stack, typ, f'{processed.name} is not allowed as inner type', name)
                    return None
                annotation.update_inner_type(key, processed)
            return annotation
        elif isinstance(annotation, (abstract.Class, abstract.AMBIGUOUS_OR_EMPTY, abstract.TypeParameter, abstract.ParamSpec, abstract.ParamSpecArgs, abstract.ParamSpecKwargs, abstract.Concatenate, abstract.FinalAnnotation, function.ParamSpecMatch, typing_overlay.Final, typing_overlay.Never)):
            return annotation
        else:
            self.ctx.errorlog.invalid_annotation(stack, annotation, 'Not a type', name)
            return None

    def _eval_expr_as_tuple(self, node, expr, stack):
        if False:
            for i in range(10):
                print('nop')
        'Evaluate an expression as a tuple.'
        if not expr:
            return ((), None)
        f_globals = self.ctx.vm.frame.f_globals
        f_locals = self.ctx.vm.frame.f_locals
        with self.ctx.vm.generate_late_annotations(stack):
            (result_var, errorlog) = abstract_utils.eval_expr(self.ctx, node, f_globals, f_locals, expr)
        result = abstract_utils.get_atomic_value(result_var)
        if isinstance(result, abstract.PythonConstant) and isinstance(result.pyval, tuple):
            return (tuple((abstract_utils.get_atomic_value(x) for x in result.pyval)), errorlog)
        else:
            return ((result,), errorlog)

    def deformalize(self, value):
        if False:
            print('Hello World!')
        while value.formal:
            if isinstance(value, abstract.ParameterizedClass):
                value = value.base_cls
            else:
                value = self.ctx.convert.unsolvable
        return value