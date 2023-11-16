import itertools
from typing import Dict, List, Sequence, Union
from torchgen.api import cpp
from torchgen.api.types import DispatcherSignature
from torchgen.code_template import CodeTemplate
from torchgen.context import with_native_function
from torchgen.model import Argument, NativeFunction, SchemaKind, TensorOptionsArguments
from torchgen.utils import FileManager
MANUAL_BACKEND = {'options', 'data', 'set_data', 'is_leaf', 'output_nr', '_version', 'retain_grad', '_backward', 'requires_grad_'}
MANUAL_AUTOGRAD_AND_TRACER = {'resize_', 'resize_as_', 'detach', 'detach_', 'copy_', '_fw_primal', '_make_dual'}
MANUAL_AUTOGRAD = MANUAL_TRACER = MANUAL_BACKEND | MANUAL_AUTOGRAD_AND_TRACER
DONT_RECORD_TRACE = {'convolution', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d', 'lstm_cell', 'gru_cell', 'rnn_tanh_cell', 'rnn_relu_cell', '_coalesced'}

def should_trace(f: NativeFunction) -> bool:
    if False:
        i = 10
        return i + 15
    if any((str(arg.type) in {'Storage', 'Type', 'ConstQuantizerPtr'} for arg in f.func.schema_order_arguments())):
        return False
    if not any((r.type.is_tensor_like() for r in f.func.returns)):
        return False
    return f.func.name.name.base not in DONT_RECORD_TRACE
SELECT = CodeTemplate('\nif (${cond}) {\n  ${true}\n} else {\n  ${false}\n}\n')
OP_NAME = CodeTemplate('op_name = c10::Symbol::fromQualString("aten::${trace_name}");\n')
RENAME_TRACE = {'zero': 'zeros_like', 'fill': 'full_like'}

def format_trace_op_name(f: NativeFunction) -> str:
    if False:
        return 10
    if f.func.kind() in (SchemaKind.functional, SchemaKind.out) or f.func.name.name.dunder_method:
        trace_name = str(f.func.name.name)
        trace_name = RENAME_TRACE.get(trace_name, trace_name)
        return OP_NAME.substitute(trace_name=trace_name)
    outplace_trace_name = f.func.name.name.base
    inplace_trace_name = cpp.name(f.func)
    outplace_trace_name = RENAME_TRACE.get(outplace_trace_name, outplace_trace_name)
    inplace_trace_name = RENAME_TRACE.get(inplace_trace_name, inplace_trace_name)
    return SELECT.substitute(cond='tracer_state->force_outplace', true=OP_NAME.substitute(trace_name=outplace_trace_name), false=OP_NAME.substitute(trace_name=inplace_trace_name))
ADD_TRACE_INPUT = CodeTemplate('jit::tracer::addInputs(node, "${name}", ${input});')

def format_trace_inputs(f: NativeFunction) -> str:
    if False:
        print('Hello World!')

    def dispatch_trace_input(arg: Union[Argument, TensorOptionsArguments]) -> Sequence[str]:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(arg, TensorOptionsArguments):
            name = 'options'
            return [ADD_TRACE_INPUT.substitute(name=name, input='optTypeMetaToScalarType(options.dtype_opt())'), ADD_TRACE_INPUT.substitute(name=name, input='options.layout()'), ADD_TRACE_INPUT.substitute(name=name, input='options.device()'), ADD_TRACE_INPUT.substitute(name=name, input='options.pinned_memory()')]
        else:
            name = arg.name
            if str(arg.type) == 'Tensor?[]':
                return [f'jit::tracer::addInputs(node, "{name}", {name});']
            else:
                return [ADD_TRACE_INPUT.substitute(name=name, input=name)]
    args: List[Union[Argument, TensorOptionsArguments]] = list(f.func.schema_order_arguments())
    if f.func.is_out_fn():
        num_out_args = len(f.func.arguments.out)
        args = args[:-num_out_args]
    trace_inputs = itertools.chain.from_iterable((dispatch_trace_input(arg) for arg in args))
    if f.func.is_out_fn():
        inplace = [ADD_TRACE_INPUT.substitute(name=f.func.arguments.out[i].name, input=f.func.arguments.out[i].name) for i in range(num_out_args)]
        has_tensor_return = any((r.type.is_tensor_like() for r in f.func.returns))
        has_tensor_input_arg = any((a.type.is_tensor_like() for a in f.func.arguments.flat_non_out))
        is_factory_method = f.category_override == 'factory' or (has_tensor_return and (not has_tensor_input_arg))
        if f.func.name.name.base == 'normal':
            is_factory_method = True
        if is_factory_method:
            outplace = [ADD_TRACE_INPUT.substitute(name='out', input='optTypeMetaToScalarType(out.options().dtype_opt())'), ADD_TRACE_INPUT.substitute(name='out', input='out.options().layout()'), ADD_TRACE_INPUT.substitute(name='out', input='out.options().device()'), ADD_TRACE_INPUT.substitute(name='out', input='out.options().pinned_memory()')]
        else:
            outplace = []
        trace_inputs = itertools.chain(trace_inputs, [SELECT.substitute(cond='tracer_state->force_outplace', true='\n'.join(outplace), false='\n'.join(inplace))])
    return '\n'.join(trace_inputs)
RENAME_TRACE_ADD_ARGS = {'fill': '    jit::tracer::addInputs(node, "options", c10::optional<ScalarType>());\n    jit::tracer::addInputs(node, "options", layout_or_default(c10::nullopt));\n    jit::tracer::addInputs(node, "options", device_or_default(c10::nullopt));\n    jit::tracer::addInputs(node, "options", pinned_memory_or_default(c10::nullopt));\n    c10::optional<MemoryFormat> memory_format = c10::MemoryFormat::Preserve;\n    jit::tracer::addInputs(node, "memory_format", memory_format);\n', 'zero': '    jit::tracer::addInputs(node, "options", c10::optional<ScalarType>());\n    jit::tracer::addInputs(node, "options", layout_or_default(c10::nullopt));\n    jit::tracer::addInputs(node, "options", device_or_default(c10::nullopt));\n    jit::tracer::addInputs(node, "options", pinned_memory_or_default(c10::nullopt));\n    c10::optional<MemoryFormat> memory_format = c10::MemoryFormat::Preserve;\n    jit::tracer::addInputs(node, "memory_format", memory_format);\n'}
INPLACE_GUARD = CodeTemplate('jit::tracer::ensureUniqueIfOutOfPlaced("${name}", ${mutable_input});\n')
PRE_RECORD_TRACE = CodeTemplate('torch::jit::Node* node = nullptr;\nstd::shared_ptr<jit::tracer::TracingState> tracer_state;\nif (jit::tracer::isTracing()) {\n  tracer_state = jit::tracer::getTracingState();\n  at::Symbol op_name;\n  ${set_op_name}\n  node = tracer_state->createNode(op_name, /*num_outputs=*/0);\n  jit::tracer::recordSourceLocation(node);\n  ${add_trace_inputs}\n  tracer_state->insertNode(node);\n  ${inplace_guard}\n  jit::tracer::setTracingState(nullptr);\n}\n')

def format_prerecord_trace(f: NativeFunction) -> str:
    if False:
        i = 10
        return i + 15
    if not should_trace(f):
        return ''
    is_inplace = f.func.kind() in (SchemaKind.inplace, SchemaKind.out) and (not f.func.name.name.dunder_method)
    add_args = RENAME_TRACE_ADD_ARGS.get(f.func.name.name.base, '') if is_inplace else ''
    additional_inputs = SELECT.substitute(cond='tracer_state->force_outplace', true=add_args, false='') if add_args else ''
    return PRE_RECORD_TRACE.substitute(set_op_name=format_trace_op_name(f), add_trace_inputs=format_trace_inputs(f) + additional_inputs, inplace_guard=INPLACE_GUARD.substitute(name=cpp.name(f.func), mutable_input=f.func.arguments.out[0].name if f.func.arguments.out else 'self') if is_inplace else '')
POST_RECORD_TRACE = CodeTemplate('if (tracer_state) {\n  jit::tracer::setTracingState(std::move(tracer_state));\n  ${add_trace_outputs}\n}\n')

def format_postrecord_trace(f: NativeFunction) -> str:
    if False:
        return 10
    if not should_trace(f):
        return ''
    if f.func.is_out_fn():
        output_names_outplace = [arg.name for arg in f.func.arguments.out]
        output_names_inplace = cpp.return_names(f)
        if output_names_outplace == output_names_inplace:
            outputs = [f'jit::tracer::addOutput(node, {n});' for n in output_names_outplace]
            return POST_RECORD_TRACE.substitute(add_trace_outputs=outputs)
        selection = SELECT.substitute(cond='force_outplace', true='\n'.join((f'jit::tracer::addOutput(node, {n});' for n in output_names_outplace)), false='\n'.join((f'jit::tracer::addOutput(node, {n});' for n in output_names_inplace)))
        return POST_RECORD_TRACE.substitute(add_trace_outputs=selection)
    else:
        output_names = cpp.return_names(f)
        outputs = [f'jit::tracer::addOutput(node, {n});' for n in output_names]
        return POST_RECORD_TRACE.substitute(add_trace_outputs=outputs)

def declare_returned_variables(f: NativeFunction) -> str:
    if False:
        while True:
            i = 10
    modifies_arguments = f.func.kind() in (SchemaKind.inplace, SchemaKind.out)
    if modifies_arguments:
        return ''
    if len(f.func.returns) == 1:
        return ''
    types = [cpp.return_type(r, symint=True) for r in f.func.returns]
    names = cpp.return_names(f)
    return '\n'.join((f'{type.cpp_type()} {name};' for (type, name) in zip(types, names)))

def tie_return_values(f: NativeFunction) -> str:
    if False:
        while True:
            i = 10
    if len(f.func.returns) == 1:
        return f"auto {f.func.returns[0].name or 'result'}"
    names = cpp.return_names(f)
    return f"std::tie({', '.join(names)})"

def get_return_value(f: NativeFunction) -> str:
    if False:
        return 10
    names = cpp.return_names(f)
    if len(f.func.returns) == 1:
        return names[0]
    if f.func.kind() == SchemaKind.out:
        return f"std::forward_as_tuple({', '.join(names)})"
    else:
        moved = ', '.join((f'std::move({name})' for name in names))
        return f'std::make_tuple({moved})'
TRACE_DISPATCH = CodeTemplate('${assign_return_values}at::_ops::${unambiguous_name}::redispatch(${unpacked_args});')

def emit_trace_body(f: NativeFunction) -> List[str]:
    if False:
        print('Hello World!')
    trace_body: List[str] = []
    trace_body.append(format_prerecord_trace(f))
    trace_body.append(declare_returned_variables(f))
    dispatcher_sig = DispatcherSignature.from_schema(f.func)
    dispatcher_exprs = dispatcher_sig.exprs()
    dispatch_key_set = 'ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer)'
    redispatch_args = ', '.join([dispatch_key_set] + [a.expr for a in dispatcher_exprs])
    assign_return_values = f'{tie_return_values(f)} = ' if f.func.kind() in [SchemaKind.functional, SchemaKind.mutable] and f.func.returns else ''
    trace_body.append(TRACE_DISPATCH.substitute(assign_return_values=assign_return_values, unambiguous_name=f.func.name.unambiguous_name(), unpacked_args=redispatch_args))
    trace_body.append(format_postrecord_trace(f))
    if f.func.returns:
        trace_body.append(f'return {get_return_value(f)};')
    return trace_body
METHOD_DEFINITION = CodeTemplate('${return_type} ${type_wrapper_name}(${formals}) {\n  ${type_definition_body}\n}\n')

def type_wrapper_name(f: NativeFunction, key: str='Default') -> str:
    if False:
        i = 10
        return i + 15
    if f.func.name.overload_name:
        name = f'{cpp.name(f.func)}_{f.func.name.overload_name}'
    else:
        name = cpp.name(f.func)
    if key != 'Default':
        name = name + f'_{key}'
    return name

@with_native_function
def method_definition(f: NativeFunction) -> str:
    if False:
        i = 10
        return i + 15
    assert cpp.name(f.func) not in MANUAL_TRACER
    formals = ', '.join(['c10::DispatchKeySet ks'] + [f"{cpp.argument_type(a, binds='__placeholder__', symint=True).cpp_type()} {a.name}" for a in f.func.schema_order_arguments()])
    return METHOD_DEFINITION.substitute(return_type=cpp.returns_type(f.func.returns, symint=True).cpp_type(), type_wrapper_name=type_wrapper_name(f), formals=formals, type_definition_body=emit_trace_body(f))
WRAPPER_REGISTRATION = CodeTemplate('m.impl("${name}",\n       TORCH_FN(${class_type}::${type_wrapper_name})\n);\n')

@with_native_function
def method_registration(f: NativeFunction) -> str:
    if False:
        while True:
            i = 10
    assert cpp.name(f.func) not in MANUAL_TRACER
    return WRAPPER_REGISTRATION.substitute(name=f.func.name, type_wrapper_name=type_wrapper_name(f), class_type='TraceType')

def gen_trace_type_func(fn: NativeFunction) -> Dict[str, List[str]]:
    if False:
        return 10
    return {'ops_headers': [f'#include <ATen/ops/{fn.root_name}_ops.h>'], 'trace_method_definitions': [method_definition(fn)], 'trace_wrapper_registrations': [method_registration(fn)]}

def gen_trace_type(out: str, native_functions: List[NativeFunction], template_path: str) -> None:
    if False:
        while True:
            i = 10
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    fm.write_sharded('TraceType.cpp', [fn for fn in native_functions if cpp.name(fn.func) not in MANUAL_TRACER], key_fn=lambda fn: fn.root_name, base_env={'generated_comment': '@' + f'generated from {fm.template_dir_for_comments()}/TraceType.cpp'}, env_callable=gen_trace_type_func, num_shards=5, sharded_keys={'ops_headers', 'trace_method_definitions', 'trace_wrapper_registrations'})