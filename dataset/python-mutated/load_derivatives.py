import re
from collections import defaultdict
from typing import Any, Counter, Dict, List, Match, Optional, Sequence, Set, Tuple
import yaml
from torchgen.api import cpp
from torchgen.api.autograd import Derivative, DifferentiabilityInfo, ForwardDerivative, SavedAttribute
from torchgen.api.types import BaseCType, Binding, boolT, CppSignatureGroup, layoutT, longT, NamedCType, OptionalCType, scalarTypeT, SpecialArgName, stringT, symIntArrayRefT, SymIntT, tensorGeometryT, tensorOptionsT, typeAndSizeT, VectorCType
from torchgen.context import with_native_function
from torchgen.gen import get_grouped_by_view_native_functions, parse_native_yaml
from torchgen.model import AUTOGRAD_KEYS, FunctionSchema, NativeFunction, NativeFunctionsViewGroup, OperatorName, SchemaKind, Type, Variant
from torchgen.utils import concatMap, IDENT_REGEX, split_name_params
from torchgen.yaml_utils import YamlLoader
_GLOBAL_LOAD_DERIVATIVE_CACHE = {}
_VALID_AUTOGRAD_KEYS = set(AUTOGRAD_KEYS)

def add_view_copy_derivatives(infos: Dict[FunctionSchema, Dict[str, DifferentiabilityInfo]], view_groups: List[NativeFunctionsViewGroup]) -> None:
    if False:
        while True:
            i = 10
    view_name_to_group: Dict[OperatorName, NativeFunctionsViewGroup] = {g.view.func.name: g for g in view_groups}
    view_infos = {}
    for info_dispatch_dict in infos.values():
        maybe_view_group = None
        view_copy_differentiability_infos = {}
        for (dispatch_key, info) in info_dispatch_dict.items():
            maybe_view_group = view_name_to_group.get(info.func.func.name, None)
            if maybe_view_group is not None and maybe_view_group.view_copy is not None:
                view_copy_info = info.create_view_copy_from_view_derivative(maybe_view_group)
                if view_copy_info is not None:
                    fn_schema = view_copy_info.func.func
                    view_copy_differentiability_infos[dispatch_key] = view_copy_info
            else:
                break
        if len(view_copy_differentiability_infos) > 0:
            assert fn_schema is not None
            view_infos[fn_schema] = view_copy_differentiability_infos
    infos.update(view_infos)

def load_derivatives(derivatives_yaml_path: str, native_yaml_path: str, tags_yaml_path: str) -> Tuple[Dict[FunctionSchema, Dict[str, DifferentiabilityInfo]], Set[str]]:
    if False:
        for i in range(10):
            print('nop')
    global _GLOBAL_LOAD_DERIVATIVE_CACHE
    key = (derivatives_yaml_path, native_yaml_path)
    if key not in _GLOBAL_LOAD_DERIVATIVE_CACHE:
        with open(derivatives_yaml_path) as f:
            definitions = yaml.load(f, Loader=YamlLoader)
        funcs = parse_native_yaml(native_yaml_path, tags_yaml_path).native_functions
        native_functions_with_view_groups = get_grouped_by_view_native_functions(funcs)
        native_functions_without_view_copies = concatMap(lambda g: [g] if isinstance(g, NativeFunction) else list(g.functions(include_copy=False)), native_functions_with_view_groups)
        view_groups = [g for g in native_functions_with_view_groups if isinstance(g, NativeFunctionsViewGroup)]
        functions_by_signature: Dict[FunctionSchema, List[NativeFunction]] = defaultdict(list)
        functions_by_schema: Dict[str, NativeFunction] = {}
        for function in native_functions_without_view_copies:
            functions_by_signature[function.func.signature()].append(function)
            assert str(function.func) not in functions_by_schema
            functions_by_schema[str(function.func)] = function
        op_counter = Counter[str]()
        infos: Dict[FunctionSchema, Dict[str, DifferentiabilityInfo]] = {}
        used_dispatch_keys: Set[str] = set()
        for defn_dict in definitions:
            if 'dispatch' not in defn_dict:
                specification = defn_dict.pop('name')
                output_differentiability = defn_dict.pop('output_differentiability', None)
                defn_dict = {'name': specification, 'dispatch': {'Default': defn_dict}}
                if output_differentiability:
                    defn_dict['output_differentiability'] = output_differentiability
            (name, per_dispatch_diffinfos) = create_differentiability_info(defn_dict, functions_by_signature, functions_by_schema, op_counter, used_dispatch_keys)
            infos[name] = per_dispatch_diffinfos
        add_view_copy_derivatives(infos, view_groups)
        _GLOBAL_LOAD_DERIVATIVE_CACHE[key] = (infos, used_dispatch_keys)
    return _GLOBAL_LOAD_DERIVATIVE_CACHE[key]

@with_native_function
def cpp_arguments(f: NativeFunction) -> Sequence[Binding]:
    if False:
        print('Hello World!')
    sigs = CppSignatureGroup.from_native_function(f, method=False)
    if sigs.symint_signature is not None:
        return sigs.symint_signature.arguments()
    else:
        return sigs.signature.arguments()

def create_derivative(f: NativeFunction, formula: str, var_names: Tuple[str, ...], available_named_gradients: Sequence[str]) -> Derivative:
    if False:
        while True:
            i = 10
    original_formula = formula
    arguments: List[NamedCType] = [a.nctype.remove_const_ref() for a in cpp_arguments(f)]
    return_names = tuple((n if n != 'self' else 'result' for n in cpp.return_names(f)))
    return_types = tuple((cpp.return_type(r, symint=True).remove_const_ref() for r in f.func.returns))
    named_returns = [NamedCType(name, type) for (name, type) in zip(return_names, return_types)]
    (formula, saved_inputs) = saved_variables(formula, arguments, var_names)
    (formula, saved_outputs) = saved_variables(formula, named_returns, var_names)
    used_named_gradients = {name for name in available_named_gradients if re.search(IDENT_REGEX.format(name), formula)}
    for i in used_gradient_indices(formula):
        if i >= len(f.func.returns):
            raise RuntimeError(f'Out of bounds grads access: derivative formula for {cpp.name(f.func)} used grads[{i}], but the forward only returns {len(f.func.returns)} outputs.')
    return Derivative(formula=formula, original_formula=original_formula, var_names=var_names, saved_inputs=saved_inputs, saved_outputs=saved_outputs, named_gradients=used_named_gradients)

def create_forward_derivative(f: NativeFunction, formula: str, names: Tuple[str, ...]) -> ForwardDerivative:
    if False:
        return 10
    var_names = names
    var_types: Optional[Tuple[Type, ...]] = None
    for r in f.func.returns:
        if r.name in var_names:
            if var_types is None:
                var_types = tuple()
            var_types = var_types + (r.type,)
    if var_types is None:
        if var_names == ('result',):
            assert len(f.func.returns) == 1
            var_types = (f.func.returns[0].type,)
        else:
            for var_name in var_names:
                res = re.findall('^result(\\d+)$', var_name)
                if len(res) == 1:
                    if var_types is None:
                        var_types = tuple()
                    arg_idx = int(res[0])
                    var_types = var_types + (f.func.returns[arg_idx].type,)
    assert var_types is not None, 'No matching output for forward derivative definition'
    return ForwardDerivative(formula=formula, var_names=var_names, var_types=var_types, required_inputs_fw_grad=None, required_inputs_primal=None, required_original_self_value=False, is_reusing_outplace_formula=False)

def postprocess_forward_derivatives(f: NativeFunction, defn_name: str, all_arg_names: List[str], derivatives: List[Derivative], forward_derivatives: List[ForwardDerivative], args_with_derivatives: Sequence[Binding]) -> List[ForwardDerivative]:
    if False:
        return 10

    def find_required_inputs(formula: str, postfix: str) -> Tuple[str, ...]:
        if False:
            print('Hello World!')
        is_foreach = f.func.name.name.base.startswith('_foreach_')
        required_inputs = set()
        for arg in args_with_derivatives:
            if arg.type in ('at::TensorList', 'const at::ITensorListRef &') and (not is_foreach):
                continue
            arg_name = arg.name
            found = re.search(IDENT_REGEX.format(arg_name), formula)
            if found:
                raise RuntimeError(f'The forward formula for {defn_name} is using the base name of the {arg_name} argument which is ambiguous. You should use {arg_name}_p to access the primal value and {arg_name}_t to access the tangent.')
            found = re.search(IDENT_REGEX.format(arg_name + postfix), formula)
            if found:
                required_inputs.add(arg_name)
        return tuple(required_inputs)
    updated_derivatives: List[ForwardDerivative] = []
    for defn in forward_derivatives:
        formula = defn.formula
        required_inputs_tangent = find_required_inputs(formula, '_t')
        if formula == 'auto_element_wise':
            assert f.func.kind() != SchemaKind.inplace, f'Cannot use auto_element_wise with {f.func.name} because it is an in-place variant'
            if not len(args_with_derivatives) == 1 or len(forward_derivatives) > 1 or len(forward_derivatives[0].var_names) > 1:
                raise RuntimeError(f'Derivative definition of {defn_name} in derivatives.yaml defines the forward definition of gradient as element_wise but this only works for functions with a single differentiable input and a single differentiable output.')
            if not len(derivatives) == 1:
                raise RuntimeError(f'Derivative definition of {defn_name} in derivatives.yaml defines the forward definition of gradient as element_wise but it does not defines the gradient formula for its argument which is required.')
            backward_formula = derivatives[0].original_formula
            input_name = args_with_derivatives[0].name

            def repl(m: Any) -> str:
                if False:
                    return 10
                return f'{m.group(1)}{input_name}_t.conj(){m.group(2)}'
            fw_formula = re.sub(IDENT_REGEX.format('grad'), repl, backward_formula)
            for arg in args_with_derivatives:
                arg_name = arg.name

                def repl(m: Any) -> str:
                    if False:
                        print('Hello World!')
                    return f'{m.group(1)}{arg_name}_p{m.group(2)}'
                fw_formula = re.sub(IDENT_REGEX.format(arg_name), repl, fw_formula)
            fw_formula = f'({fw_formula}).conj()'
            required_inputs_tangent = tuple(all_arg_names)
            formula = fw_formula
        elif formula == 'auto_linear':
            if len(forward_derivatives) > 1 or len(forward_derivatives[0].var_names) > 1:
                raise RuntimeError(f'Derivative definition of {defn_name} in derivatives.yaml defines the forward definition of gradient as linear but this only works for functions with a single differentiable output.')
            diff_arg_names = [arg.name for arg in args_with_derivatives]
            assert len(diff_arg_names) > 0
            new_args = []
            for arg_name in all_arg_names:
                if arg_name in diff_arg_names:
                    arg_name = arg_name + '_t'
                new_args.append(arg_name)
            if f.func.has_symint():
                defn_name += '_symint'
            if Variant.function in f.variants:
                fw_formula = f"at::{defn_name}({', '.join(new_args)})"
            else:
                assert Variant.method in f.variants
                fw_formula = f"{new_args[0]}.{defn_name}({', '.join(new_args[1:])})"
            required_inputs_tangent = tuple(diff_arg_names)
            formula = fw_formula
        required_inputs_primal = find_required_inputs(formula, '_p')
        updated_derivatives.append(ForwardDerivative(formula=formula, var_names=defn.var_names, var_types=defn.var_types, required_inputs_fw_grad=required_inputs_tangent, required_inputs_primal=required_inputs_primal, required_original_self_value=False, is_reusing_outplace_formula=False))
    return updated_derivatives

def is_forward_derivative_definition(all_arg_names: List[str], names: Tuple[str, ...]) -> bool:
    if False:
        for i in range(10):
            print('nop')
    for name in names:
        if name not in all_arg_names:
            return True
        else:
            return False
    raise RuntimeError('Expected `names` to be non-empty')

def create_differentiability_info(defn_dict: Dict[Any, Any], functions_by_signature: Dict[FunctionSchema, List[NativeFunction]], functions_by_schema: Dict[str, NativeFunction], op_counter: Counter[str], used_dispatch_keys: Set[str]) -> Tuple[FunctionSchema, Dict[str, DifferentiabilityInfo]]:
    if False:
        for i in range(10):
            print('nop')
    'Processes a single entry `defn` in derivatives.yaml'

    def canonical_function(functions: Sequence[NativeFunction], name: str) -> NativeFunction:
        if False:
            print('Hello World!')
        for f in functions:
            if not f.func.is_functional_fn() and (not f.func.is_out_fn()) and (name == str(f.func.name.name)):
                return f
        assert name + '_' == cpp.name(functions[0].func)
        return functions[0]

    def split_names(raw_names: str) -> Tuple[str, ...]:
        if False:
            return 10
        'Given "foo, bar", return ["foo", "bar"].'
        return tuple((x.strip() for x in raw_names.split(',')))

    def check_grad_usage(defn_name: str, derivatives: Sequence[Derivative]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Check for some subtle mistakes one might make when writing derivatives.\n        These mistakes will compile, but will be latent until a function is\n        used with double backwards.\n        '
        uses_grad = False
        num_grads_uses = 0
        uses_named_grads = False
        used_grads_indices: List[int] = []
        for d in derivatives:
            formula = d.formula
            uses_grad = uses_grad or bool(re.findall(IDENT_REGEX.format('grad'), formula))
            num_grads_uses += len(re.findall(IDENT_REGEX.format('grads'), formula))
            uses_named_grads = uses_named_grads or bool(d.named_gradients)
            used_grads_indices.extend(used_gradient_indices(formula))
        assert num_grads_uses >= len(used_grads_indices)
        only_used_grads_indices = num_grads_uses == len(used_grads_indices)
        if uses_grad and num_grads_uses > 0:
            raise RuntimeError(f"Derivative definition of {defn_name} in derivatives.yaml illegally mixes use of 'grad' and 'grads'. Consider replacing occurrences of 'grad' with 'grads[0]'")
        if only_used_grads_indices and set(used_grads_indices) == {0}:
            raise RuntimeError(f"Derivative definition of {defn_name} in derivatives.yaml solely refers to 'grads[0]'.  If the first output is indeed the only differentiable output, replace 'grads[0]' with 'grad'; otherwise, there is a likely error in your derivatives declaration.")
        if uses_named_grads and (uses_grad or num_grads_uses > 0):
            raise RuntimeError(f'Derivative definition of {defn_name} in derivatives.yaml illegally mixes use of "grad_RETURN_NAME" and "grad" or "grads[x]". Use only one method for identifying gradients.')

    @with_native_function
    def set_up_derivatives(f: NativeFunction) -> Tuple[Sequence[Derivative], Sequence[ForwardDerivative], Sequence[Binding], Sequence[str], Sequence[str]]:
        if False:
            return 10
        derivatives: List[Derivative] = []
        forward_derivatives: List[ForwardDerivative] = []
        non_differentiable_arg_names: List[str] = []
        args_with_derivatives_set: Set[str] = set()
        all_arg_names = [a.name for a in cpp_arguments(f)]
        all_ret_names = [r.name for r in f.func.returns]
        differentiability = output_differentiability or [True] * len(f.func.returns)
        available_named_gradients = [f'grad_{ret.name}' for (ret, differentiable) in zip(f.func.returns, differentiability) if differentiable and ret.name is not None and ret.type.is_tensor_like()]
        for raw_names in sorted(defn.keys()):
            formula = defn[raw_names]
            names = split_names(raw_names)
            for name in names:
                assert not (name in all_arg_names and name in all_ret_names), f"While processing the derivative formula for '{f.func.name}' wrt '{name}', expected '{name}' to not be both an input arg and named return. "
            if is_forward_derivative_definition(all_arg_names, names):
                forward_derivatives.append(create_forward_derivative(f, formula, names))
            elif formula.lower().strip() == 'non_differentiable':
                non_differentiable_arg_names += names
            else:
                derivative = create_derivative(f, formula, names, available_named_gradients)
                derivatives.append(derivative)
                args_with_derivatives_set |= set(names)
        overlap = args_with_derivatives_set.intersection(non_differentiable_arg_names)
        if overlap:
            raise RuntimeError(f'derivatives definition for {defn} have overlapped non_differentiable and differentiable variables: {overlap}')
        args_with_derivatives = [a for a in cpp_arguments(f) if a.name in args_with_derivatives_set]
        forward_derivatives = postprocess_forward_derivatives(f, defn_name, all_arg_names, derivatives, forward_derivatives, args_with_derivatives)
        check_grad_usage(defn_name, derivatives)
        return (derivatives, forward_derivatives, args_with_derivatives, non_differentiable_arg_names, available_named_gradients)
    specification = defn_dict.pop('name')
    (defn_name, _) = split_name_params(specification)
    output_differentiability = defn_dict.pop('output_differentiability', None)
    output_differentiability_conditions = None
    if output_differentiability and any((isinstance(diff, str) for diff in output_differentiability)):
        if len(output_differentiability) != 1:
            raise RuntimeError(f'Not supported: for {specification},output_differentiability must either be List[bool] or a List[str] where each str is a condition. In the case where it is a condition, we only support single-output functions. Please file us an issue. ')
        output_differentiability_conditions = output_differentiability
        output_differentiability = [True]
    schema_function = functions_by_schema.get(specification)
    if not schema_function:
        avail = '\n'.join((k for (k, v) in functions_by_schema.items() if cpp.name(v.func) == defn_name))
        raise RuntimeError(f'could not find ATen function for schema: {specification} .  Available signatures:\n{avail}')
    signature = schema_function.func.signature()
    functions = functions_by_signature[signature]
    if len(functions) == 0:
        avail = '\n'.join((str(k) for (k, v) in functions_by_signature.items() if cpp.name(k) == defn_name))
        raise RuntimeError(f'could not find ATen function for legacy signature: {signature} corresponding to schema {specification}.  Please report a bug to PyTorch. Available signatures:\n{avail}')
    canonical = canonical_function(functions, defn_name)
    if 'grad_input_mask' in (a.name for a in cpp_arguments(canonical)):
        raise RuntimeError(f'Schema for {defn_name} has an argument named grad_input_mask, but this name would be shadowed by our codegen. Please use a different name in native_functions.yaml.')
    if 'result' in (a.name for a in cpp_arguments(canonical)):
        raise RuntimeError(f'Schema for {defn_name} has an argument named result, but this is only allowed for outputs.Please use a different name in native_functions.yaml.')
    diffinfo_dict = {}
    for (key, defn) in defn_dict['dispatch'].items():
        if key != 'Default' and key not in _VALID_AUTOGRAD_KEYS:
            raise RuntimeError(f'Invalid dispatch key {key} in derivatives.yaml for {specification}, expected key to be one of {_VALID_AUTOGRAD_KEYS}')
        if key not in used_dispatch_keys:
            used_dispatch_keys.add(key)
        (derivatives, forward_derivatives, args_with_derivatives, non_differentiable_arg_names, available_named_gradients) = set_up_derivatives(canonical)
        used_named_gradients: Set[str] = set()
        for d in derivatives:
            used_named_gradients |= d.named_gradients
        op = None
        if args_with_derivatives:
            op_prefix = _create_op_prefix(defn_name)
            if key != 'Default':
                op_prefix = op_prefix + key
            op = f'{op_prefix}{op_counter[op_prefix]}'
            op_counter[op_prefix] += 1
        diffinfo_dict[key] = DifferentiabilityInfo(name=defn_name, func=canonical, op=op, derivatives=derivatives, forward_derivatives=forward_derivatives, all_saved_inputs=dedup_vars([v for d in derivatives for v in d.saved_inputs]), all_saved_outputs=dedup_vars([v for d in derivatives for v in d.saved_outputs]), available_named_gradients=available_named_gradients, used_named_gradients=used_named_gradients, args_with_derivatives=args_with_derivatives, non_differentiable_arg_names=non_differentiable_arg_names, output_differentiability=output_differentiability, output_differentiability_conditions=output_differentiability_conditions)
    return (canonical.func, diffinfo_dict)
GRAD_INDEX_REGEX = '(?:^|\\W)grads\\[(\\d+)\\]'

def used_gradient_indices(formula: str) -> List[int]:
    if False:
        return 10
    'Determine a list of gradient indices (the i in grads[i]) that\n    are used by the formula.\n\n    >>> used_gradient_indices("foo(grads[0], grads[1])")\n    [0, 1]\n    '
    return [int(i) for i in re.findall(GRAD_INDEX_REGEX, formula)]

def saved_variables(formula: str, nctypes: List[NamedCType], var_names: Tuple[str, ...]) -> Tuple[str, Tuple[SavedAttribute, ...]]:
    if False:
        return 10

    def stride_expr(name: str) -> str:
        if False:
            while True:
                i = 10
        assert var_names == (name,), 'Replacement for ".strides()" is currently only supported for single derivatives of the same tensor that ".strides()" is being called on.'
        return f'strides_or_error({name}, "{name}")'
    REPLACEMENTS: List[Tuple[str, Dict[str, Any]]] = [('{}.sym_sizes\\(\\)', {'suffix': '_sym_sizes', 'nctype': lambda name: NamedCType(name, BaseCType(symIntArrayRefT))}), ('{}->sym_sizes\\(\\)', {'suffix': '_sym_sizes_opt', 'nctype': lambda name: NamedCType(name, OptionalCType(BaseCType(symIntArrayRefT))), 'expr': lambda name: f'{name}.has_value() ? c10::optional<c10::SymIntArrayRef>({name}->sym_sizes()) : c10::nullopt'}), ('{}.sym_blocksize\\(\\)', {'suffix': '_self_sym_blocksize_opt', 'nctype': lambda name: NamedCType(name, OptionalCType(BaseCType(symIntArrayRefT))), 'expr': lambda name: f'at::sparse_csr::getSymIntBlockSize({name})'}), ('{}.options\\(\\)', {'suffix': '_options', 'nctype': lambda name: NamedCType(name, BaseCType(tensorOptionsT))}), ('zeros_like\\({}\\)', {'suffix': '_info', 'nctype': lambda name: NamedCType(name, BaseCType(typeAndSizeT)), 'expr': lambda name: name, 'res': lambda name: name + '_info.zeros()'}), ('{}.sym_size\\((-?\\w+)\\)', {'suffix': lambda m: f"_sym_argsize_{m.groups()[0].replace('-', 'minus_')}", 'nctype': lambda name: NamedCType(name, BaseCType(SymIntT))}), ('{}.numel\\(\\)', {'suffix': '_numel', 'nctype': lambda name: NamedCType(name, BaseCType(longT))}), ('{}.sym_numel\\(\\)', {'suffix': '_sym_numel', 'nctype': lambda name: NamedCType(name, BaseCType(SymIntT))}), ('to_args_sizes\\({}\\)', {'suffix': '_args_sizes', 'nctype': lambda name: NamedCType(name, VectorCType(VectorCType(BaseCType(longT))))}), ('to_args_sizes_symint\\({}\\)', {'suffix': '_args_sizes_symint', 'nctype': lambda name: NamedCType(name, VectorCType(VectorCType(BaseCType(SymIntT))))}), ('to_args_scalartypes\\({}\\)', {'suffix': '_args_scalartypes', 'nctype': lambda name: NamedCType(name, VectorCType(BaseCType(scalarTypeT)))}), ('TensorGeometry\\({}\\)', {'suffix': '_geometry', 'nctype': lambda name: NamedCType(name, BaseCType(tensorGeometryT))}), ('{}.scalar_type\\(\\)', {'suffix': '_scalar_type', 'nctype': lambda name: NamedCType(name, BaseCType(scalarTypeT))}), ('{}.dim\\(\\)', {'suffix': '_dim', 'nctype': lambda name: NamedCType(name, BaseCType(longT))}), ('{}.sym_strides\\(\\)', {'suffix': '_sym_strides', 'nctype': lambda name: NamedCType(name, BaseCType(symIntArrayRefT)), 'expr': stride_expr}), ('{}.layout\\(\\)', {'suffix': '_layout', 'nctype': lambda name: NamedCType(name, BaseCType(layoutT))}), ('{}.is_conj\\(\\)', {'suffix': '_conjugate', 'nctype': lambda name: NamedCType(name, BaseCType(boolT))})]
    saved: List[SavedAttribute] = []
    if '.sizes()' in formula or '->sizes()' in formula:
        raise RuntimeError('.sizes() is not supported in derivative formulas. Instead, please use the SymInt version,' + f'.sym_sizes(), which returned a c10::SymIntArrayRef. formula={formula}')
    if re.search('\\.size\\([-]?\\d+\\)', formula) or re.search('->size\\([-]?\\d+\\)', formula):
        raise RuntimeError('.size(int) is not supported in derivative formulas. Instead, please use the SymInt version,' + f'.sym_size(int), which returned a c10::SymIntArrayRef. formula={formula}')
    if '.strides()' in formula or '->strides()' in formula:
        raise RuntimeError('.strides() is not supported in derivative formulas. Instead, please use the SymInt version,' + f'.sym_strides(), which returned a c10::SymIntArrayRef. formula={formula}')
    for nctype in nctypes:
        name = nctype.name.name if isinstance(nctype.name, SpecialArgName) else nctype.name
        for (regex, info) in REPLACEMENTS:

            def repl(m: Match[str]) -> str:
                if False:
                    while True:
                        i = 10
                suffix: str = info['suffix'](m) if callable(info['suffix']) else info['suffix']
                expr: str = info['expr'](name) if 'expr' in info else m.group(0)
                saved.append(SavedAttribute(nctype=info['nctype'](name + suffix), expr=expr))
                if 'res' in info:
                    replacement: str = info['res'](name)
                    return replacement
                return name + suffix
            formula = re.sub(regex.format(name), repl, formula)
        if nctype.type == OptionalCType(BaseCType(stringT)):
            formula = re.sub(f'\\b{name}\\b', f'{name}.has_value() ? c10::optional<c10::string_view>({name}.value()) : c10::nullopt', formula)
        if re.search(IDENT_REGEX.format(name), formula):
            saved.append(SavedAttribute(nctype=nctype, expr=name))
    return (formula, tuple(saved))

def _create_op_prefix(name: str) -> str:
    if False:
        return 10
    'Takes a native function name converts to a op prefix name.\n\n    Note that the "name" parameter must be the native function name\n    without the optional variant suffix, so "add" instead of\n    "add.out".\n\n    OP names correspond to classes, hence the change to title case.\n\n    Example::\n    >>> _create_op_prefix(\'add\')\n    \'AddBackward\'\n    '
    camel_case = ''.join([p.title() for p in name.split('_')])
    return (camel_case + 'Backward').replace('ForwardBackward', 'Backward')

def dedup_vars(vars: Sequence[SavedAttribute]) -> Sequence[SavedAttribute]:
    if False:
        for i in range(10):
            print('nop')
    seen: Set[str] = set()
    saved: List[SavedAttribute] = []
    for var in vars:
        name = var.nctype.name.name if isinstance(var.nctype.name, SpecialArgName) else var.nctype.name
        if name in seen:
            continue
        seen.add(name)
        saved.append(var)
    return saved