import itertools
from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torchgen.api.dispatcher as dispatcher
from torchgen.api.lazy import getValueT, isValueType, LazyArgument, LazyIrProperties, LazyIrSchema, tensorListValueT
from torchgen.api.translate import translate
from torchgen.api.types import BaseCType, Binding, deviceT, DispatcherSignature, kernel_signature, NativeSignature, OptionalCType, VectorCType
from torchgen.context import method_with_native_function
from torchgen.dest.lazy_ts_lowering import ts_lowering_body
from torchgen.model import Argument, BackendIndex, BackendMetadata, BaseTy, BaseType, FunctionSchema, ListType, NativeFunction, NativeFunctionsGroup

def node_ctor_arg_rvalue_string(arg: LazyArgument) -> str:
    if False:
        while True:
            i = 10
    '\n    Given a LazyArgument,\n    generate a c++ string for materializing an rvalue of that arg for passing into\n    a lazy Node constructor.\n    '
    if isValueType(arg.lazy_type):
        if isinstance(arg.lazy_type, BaseCType):
            if arg.is_wrapped_scalar:
                return f'node_{arg.name}'
            elif arg.lazy_type.type is tensorListValueT:
                return f'lazy_{arg.name}_tensorlist'
            elif arg.is_symint_or_list:
                return f'GetSymIntValue({arg.name})'
            return f'lazy_{arg.name}->GetIrValue()'
        elif isinstance(arg.lazy_type, OptionalCType):
            if arg.is_symint_or_list:
                return f'{arg.name} ? c10::make_optional(GetSymIntValue(*{arg.name})) : c10::nullopt'
            elif arg.is_wrapped_scalar:
                return f'node_{arg.name}'
            return f'lazy_{arg.name} ? c10::make_optional(lazy_{arg.name}->GetIrValue()) : c10::nullopt'
        else:
            raise AssertionError(f'TODO not sure if there are other valid types to handle here ({arg.lazy_type})')
    elif isinstance(arg.orig_type, ListType) and arg.orig_type.elem == BaseType(BaseTy.SymInt):
        if arg.symint:
            return f'GetSymIntArrayRefValue({arg.name})'
        else:
            return f'std::vector<int64_t>({arg.name}.begin(), {arg.name}.end())'
    elif isinstance(arg.lazy_type, VectorCType) and isinstance(arg.lazy_type.elem, BaseCType):
        return f'std::vector<{arg.lazy_type.elem.type}>({arg.name}.begin(), {arg.name}.end())'
    elif isinstance(arg.lazy_type, OptionalCType) and isinstance(arg.lazy_type.elem, VectorCType) and isinstance(arg.lazy_type.elem.elem, BaseCType):
        return f'torch::lazy::ToOptionalVector<{arg.lazy_type.elem.elem.type}>({arg.name})'
    else:
        return f'{arg.name}'

def node_ctor_inputs(schema: LazyIrSchema) -> str:
    if False:
        while True:
            i = 10
    '\n    Produce a formatted string with the arguments as passed into the constructor of a node class.\n    '
    node_ctor_values = [node_ctor_arg_rvalue_string(arg) for arg in schema.filtered_args()]
    return ', '.join(node_ctor_values)

def gen_fallback_code(schema: LazyIrSchema, sig: Union[DispatcherSignature, NativeSignature], overload_name: str) -> str:
    if False:
        return 10
    '\n    Generate code that falls back to eager conditioned on a predicate\n    '
    dispatcher_sig = DispatcherSignature.from_schema(schema.func)
    exprs = translate(sig.arguments(), dispatcher_sig.arguments())
    fallback_args = ',\n                '.join([a.expr for a in exprs])
    if len(overload_name):
        aten_op_str = f'ATEN_OP2({schema.aten_name}, {overload_name})'
    else:
        aten_op_str = f'ATEN_OP({schema.aten_name})'
    or_has_generator = ''
    if schema.generator_arg:
        or_has_generator = f' || ({schema.generator_arg.name}.has_value() && {schema.generator_arg.name}->defined())'
    return f'\n        if (force_eager_fallback({aten_symbol(schema)}){or_has_generator}) {{\n            return at::native::call_fallback_fn_symint<&ltc_eager_fallback, {aten_op_str}>::call(\n                {fallback_args}\n            );\n        }}\n'

def aten_symbol(schema: LazyIrSchema) -> str:
    if False:
        for i in range(10):
            print('nop')
    missing_interned_strings = {'sigmoid_backward'}
    if schema.aten_name in missing_interned_strings:
        return f'c10::Symbol::fromQualString("aten::{schema.aten_name}")'
    if not schema.aten_name.startswith('at::'):
        return f'at::aten::{schema.aten_name}'
    else:
        return schema.aten_name

def convert_to_meta_tensors(sig: DispatcherSignature) -> Tuple[str, List[Binding]]:
    if False:
        return 10
    context: List[Binding] = []
    unwrapped_tensor_args: List[str] = []
    for arg in sig.arguments():
        if isinstance(arg.argument, Argument) and arg.argument.type.is_tensor_like():
            unwrapped_name = f'{arg.name}_meta'
            unwrapped_tensor_args.append(f'auto {unwrapped_name} = to_meta({arg.name});')
            context.append(arg.with_name(unwrapped_name))
        else:
            context.append(arg)
    unwrap_tensor_args_str = '\n        '.join(unwrapped_tensor_args)
    return (unwrap_tensor_args_str, context)

@dataclass(frozen=True)
class GenLazyIR(ABC):
    backend_index: BackendIndex
    backend_name: str
    node_base: str
    use_lazy_shape: bool

    @method_with_native_function
    def __call__(self, f: Union[NativeFunctionsGroup, NativeFunction]) -> List[str]:
        if False:
            while True:
                i = 10
        func = f.functional.func if isinstance(f, NativeFunctionsGroup) else f.func
        metadata = self.backend_index.get_kernel(f.functional if isinstance(f, NativeFunctionsGroup) else f)
        schema = LazyIrSchema(func, symint=metadata is not None and metadata.supports_symint())
        return self.gen(schema)

    def lowering_function(self, schema: LazyIrSchema) -> str:
        if False:
            while True:
                i = 10
        return ''

    def create_function(self, schema: LazyIrSchema, node_ctor_args: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        return ''

    def can_be_reused_function(self, schema: LazyIrSchema, node_ctor_args: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'bool CanBeReused({node_ctor_args}) const {{\n    return false;\n    }}'

    def node_base_ctor_call(self, schema: LazyIrSchema) -> str:
        if False:
            print('Hello World!')
        value_args = schema.filtered_args(values=True, scalars=False)
        base_ctor_value_args_list = []
        for arg in value_args:
            if isinstance(arg.lazy_type, (BaseCType, VectorCType)):
                base_ctor_value_args_list.append(f'{arg.name}')
            elif isinstance(arg.lazy_type, OptionalCType):
                base_ctor_value_args_list.append(f'{arg.name}.value_or(kNullValue)')
            else:
                raise AssertionError(f'Unsupported type ({arg.lazy_type}) - add support if necessary')
        base_ctor_value_args = ', '.join(base_ctor_value_args_list)
        scalar_args = schema.filtered_args(values=False, scalars=True)
        if schema.properties.ShapePrecompute:
            shape_ctor_arg = 'std::move(shapes),'
        elif schema.properties.ShapeCompute:
            shape_args = [a.name for a in value_args]
            shape_args.extend((a.name for a in scalar_args))
            shape_ctor_arg = f"compute_shape_{schema.name}({', '.join(shape_args)}),"
        elif schema.properties.ShapeCache:
            shape_args = [f'operand({i})' for i in range(len(value_args))]
            shape_args.extend((a.name for a in scalar_args))
            shape_ctor_arg = f"[&](){{ return compute_shape_{schema.name}({', '.join(shape_args)})[0]; }},"
        else:
            shape_ctor_arg = ''
        scalar_hashes = ', '.join((f'{a.name}' for a in scalar_args))
        return f'{self.node_base}(\n              {schema.node_name}::ClassOpKind(),\n              OpList{{{base_ctor_value_args}}},\n              {shape_ctor_arg}\n              /* num_outputs */ {len(schema.returns)},\n              torch::lazy::MHash({scalar_hashes}))'

    def gen(self, schema: LazyIrSchema) -> List[str]:
        if False:
            return 10
        opkind = schema.opkind or aten_symbol(schema)
        all_args = schema.filtered_args()
        value_args = schema.filtered_args(values=True, scalars=False)
        scalar_args = schema.filtered_args(values=False, scalars=True)
        ctor_args = [f'const {i.lazy_type.cpp_type()}& {i.name}' for i in all_args]
        reuse_ctor_args = ', '.join(ctor_args)
        if self.use_lazy_shape and schema.properties.ShapePrecompute:
            ctor_args.append('std::vector<torch::lazy::Shape>&& shapes')
        node_ctor_args = ', '.join(ctor_args)
        scalar_initializers = ',\n        '.join([f'{a.name}({a.name}.has_value() ? c10::make_optional(std::string(*{a.name})) : c10::nullopt)' if a.lazy_type.cpp_type() == 'c10::optional<c10::string_view>' else f'{a.name}({a.name})' for a in scalar_args])
        if len(scalar_initializers):
            scalar_initializers = f',\n        {scalar_initializers}'
        scalar_decls = '\n  '.join([f'std::string {a.name};' if a.lazy_type.cpp_type() == 'c10::string_view' else f'c10::optional<std::string> {a.name};' if a.lazy_type.cpp_type() == 'c10::optional<c10::string_view>' else f'{a.lazy_type.cpp_type()} {a.name};' for a in scalar_args])
        optional_values = [arg.name for arg in schema.filtered_args(values=True, scalars=False) if isinstance(arg.lazy_type, OptionalCType)]
        has_optional_decls = '\n  '.join([f'bool has_{value}: 1;' for value in optional_values])
        has_optional_defs = '\n    '.join([f'has_{value} = !!{value};' for value in optional_values])
        members_to_string = []
        for arg in scalar_args:
            if isinstance(arg.lazy_type, OptionalCType):
                members_to_string.append(f'if ({arg.name}.has_value()) {{\n      ss << ", {arg.name}=" << {arg.name}.value();\n    }} else {{\n      ss << ", {arg.name}=null";\n    }}')
            else:
                members_to_string.append(f'ss << ", {arg.name}=" << {arg.name};')
        members_to_string_str = '\n    '.join(members_to_string)
        return [f'class {schema.node_name} : public {self.node_base} {{\n public:\n  static torch::lazy::OpKind ClassOpKind() {{\n    return torch::lazy::OpKind({opkind});\n  }}\n\n  {schema.node_name}({node_ctor_args})\n      : {self.node_base_ctor_call(schema)}{scalar_initializers}\n  {{\n    {has_optional_defs}\n  }}\n\n  std::string ToString() const override {{\n    std::stringstream ss;\n    ss << {self.node_base}::ToString();\n    {members_to_string_str}\n    return ss.str();\n  }}\n\n  {self.create_function(schema, reuse_ctor_args)}\n\n  {self.can_be_reused_function(schema, reuse_ctor_args)}\n\n  {self.lowering_function(schema)}\n\n  {scalar_decls}\n  {has_optional_decls}\n\n}};\n\n']

@dataclass(frozen=True)
class GenTSLazyIR(GenLazyIR):

    def lowering_function(self, schema: LazyIrSchema) -> str:
        if False:
            i = 10
            return i + 15
        signature = '\n  torch::lazy::TSOpVector Lower(\n      std::shared_ptr<torch::jit::GraphFunction> function,\n      torch::lazy::TSLoweringContext* loctx) const override'
        if schema.properties.LowerDeclOnly:
            return f'{signature};'
        elif schema.properties.Lower:
            return f'{signature} {{\n    {ts_lowering_body(schema)}\n  }}\n            '
        else:
            return ''

    def create_function(self, schema: LazyIrSchema, node_ctor_args: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        signature = f'static NodePtr Create({node_ctor_args})'
        if schema.properties.CreateFnDeclOnly:
            return f'{signature};'
        elif not schema.properties.CreateFn:
            return ''
        return f'{signature} {{\n    return ReuseOrMakeNode<{schema.node_name}>(data);\n  }}'

    def can_be_reused_function(self, schema: LazyIrSchema, node_ctor_args: str) -> str:
        if False:
            i = 10
            return i + 15
        signature = f'bool CanBeReused({node_ctor_args}) const'
        if schema.properties.CanBeReusedDeclOnly:
            return f'{signature};'
        elif not schema.properties.CanBeReused:
            return ''
        value_comparison = []
        for arg in itertools.chain(schema.positional_values, schema.keyword_values):
            if isinstance(arg.lazy_type, OptionalCType):
                value_comparison.append(f'nullable_operand(i++) == {arg.name}.value_or(kNullValue)')
            else:
                value_comparison.append(f'operand(i++) == {arg.name}')
        for arg in itertools.chain(schema.positional_scalars, schema.keyword_scalars):
            if isinstance(arg.lazy_type, OptionalCType):
                value_comparison.append(f'((!this->{arg.name}&&!{arg.name}) || (this->{arg.name}&&{arg.name} && *(this->{arg.name}) == *{arg.name}))')
            else:
                value_comparison.append(f'this->{arg.name} == {arg.name}')
        value_comparison_str = ' &&\n        '.join(value_comparison)
        return f'{signature} {{\n    size_t i = 0;\n    return ({value_comparison_str});\n  }}'

@dataclass(frozen=True)
class GenLazyNativeFuncDefinition:
    class_method_name: str
    backend_index: BackendIndex
    tensor_class: str
    gen_forced_fallback_code: bool
    backend_namespace: str
    get_tensorlist: str
    get_tensor_or_wrap_number: str
    try_get_tensor: str
    metrics_counter: str
    create_tensor: str
    create_from_first_tensor: bool
    create_aten_from_ltc_tensor: str
    tuple_aten_from_ltc_tensors: str
    lazy_tensor_ptr: str
    get_device_fn: str

    def lazy_tensor_decls(self, func: NativeFunction, schema: LazyIrSchema) -> str:
        if False:
            for i in range(10):
                print('nop')
        value_args = schema.filtered_args(values=True, scalars=False)
        lazy_tensor_decls: List[str] = []
        for arg in value_args:
            if arg.is_wrapped_scalar:
                if isinstance(arg.lazy_type, OptionalCType):
                    lazy_tensor_decls.append(f'auto node_{arg.name} = {arg.name} ?\n                c10::make_optional(torch::lazy::LazyGraphExecutor::Get()->\n                    GetIrValueForScalarFromCodegen(*{arg.name}, *common_device)):\n                c10::nullopt;')
                else:
                    lazy_tensor_decls.append(f'auto node_{arg.name} = torch::lazy::LazyGraphExecutor::Get()->\n                            GetIrValueForScalarFromCodegen({arg.name}, *common_device);')
            elif arg.is_symint_or_list:
                continue
            elif isinstance(arg.lazy_type, BaseCType):
                if arg.lazy_type.type is tensorListValueT:
                    lazy_tensor_decls.append(f'auto lazy_{arg.name}_tensorlist = {self.backend_namespace}::{self.get_tensorlist}({arg.name});')
                else:
                    lazy_tensor_decls.append(f'{self.lazy_tensor_ptr} lazy_{arg.name} = {self.backend_namespace}::{self.get_tensor_or_wrap_number}({arg.name}, *common_device);')
            elif isinstance(arg.lazy_type, OptionalCType):
                assert arg.lazy_type.elem == BaseCType(getValueT()), arg.lazy_type.elem
                lazy_tensor_decls.append(f'{self.lazy_tensor_ptr} lazy_{arg.name} = {self.backend_namespace}::{self.try_get_tensor}({arg.name}.value_or(at::Tensor()));')
            else:
                raise AssertionError(f'TODO not sure if there are other valid types to handle here ({arg.lazy_type})')
        return '\n        '.join(lazy_tensor_decls)

    def force_eager_fallback(self, func: NativeFunction, schema: LazyIrSchema, metadata: BackendMetadata, sig: Union[DispatcherSignature, NativeSignature]) -> str:
        if False:
            i = 10
            return i + 15
        if self.gen_forced_fallback_code:
            return gen_fallback_code(schema, sig, overload_name=func.func.name.overload_name)
        return ''

    def metrics(self, func: NativeFunction, schema: LazyIrSchema) -> str:
        if False:
            print('Hello World!')
        return f'{self.metrics_counter};'

    def get_device(self, func: NativeFunction, schema: LazyIrSchema) -> str:
        if False:
            return 10
        value_args = schema.filtered_args(values=True, scalars=False)
        scalar_args = schema.filtered_args(values=False, scalars=True)
        value_types_names = [f'{a.name}' for a in value_args if not a.is_wrapped_scalar]
        optional_device = OptionalCType(BaseCType(deviceT))
        optional_devices = [a.name for a in scalar_args if a.lazy_type == optional_device]
        assert len(value_types_names) > 0 or len(optional_devices) > 0, 'Expected at least one Value or Device type'
        get_device_str = f"{self.get_device_fn}({', '.join(value_types_names + optional_devices)})"
        return f'auto common_device = {get_device_str};\n        TORCH_INTERNAL_ASSERT(common_device);\n        '

    def shape_inference(self, func: NativeFunction, schema: LazyIrSchema) -> str:
        if False:
            i = 10
            return i + 15
        metadata = self.backend_index.get_kernel(func)
        assert metadata is not None
        all_args = schema.filtered_args()
        returns_length = len(schema.returns)
        is_view_copy_op = 'view_copy' in func.tags
        is_structured = func.structured or func.structured_delegate is not None
        if is_structured or is_view_copy_op:
            meta_out = '\nstd::vector<torch::lazy::Shape> shapes{torch::lazy::Shape(out_meta.scalar_type(), out_meta.sizes().vec())};'
            if returns_length > 1:

                def this_shape(i: int) -> str:
                    if False:
                        return 10
                    return f'torch::lazy::Shape(std::get<{i}>(out_meta).scalar_type(), std::get<{i}>(out_meta).sizes().vec())'
                shapes_str = ','.join([this_shape(i) for i in range(returns_length)])
                meta_out = 'std::vector<torch::lazy::Shape> shapes{' + shapes_str + '};'
            dispatcher_sig = DispatcherSignature.from_schema(func.func)
            (meta_conversion_str, meta_call_ctx) = convert_to_meta_tensors(dispatcher_sig)
            meta_call_args = [e.expr for e in translate(meta_call_ctx, dispatcher_sig.arguments(), method=False)]
            if is_view_copy_op:
                assert func.has_composite_explicit_autograd_non_functional_kernel
                dispatch_ns = 'compositeexplicitautogradnonfunctional'
            else:
                dispatch_ns = 'meta'
            aten_name = schema.aten_name
            if func.func.has_symint() and metadata.supports_symint():
                aten_name += '_symint'
            shape_str = f"        {meta_conversion_str}\n        auto out_meta = at::{dispatch_ns}::{aten_name}({', '.join(meta_call_args)});\n        {meta_out}"
        else:
            shape_sig = ComputeShapeSignature(metadata.kernel, func, symint=metadata.supports_symint())
            shape_str = f'\n            auto shapes = {shape_sig.shape_call};'
        shape_str += f'\n            TORCH_INTERNAL_ASSERT(shapes.size() == {returns_length});'
        func_schema_str = 'aten::' + str(func.func)
        shape_str += f'''\n            if(torch::lazy::symbolicShapeEnabled()){{\n                std::vector<torch::jit::IValue> inputs = {{ {', '.join((str(a.name) for a in all_args))} }};\n                const char* schema_str = "{func_schema_str}";\n                applySymbolicShapesOnLT(schema_str, inputs, shapes);\n            }}\n        '''
        return shape_str

    def build_ir_node(self, func: NativeFunction, schema: LazyIrSchema) -> str:
        if False:
            print('Hello World!')
        node_ctor_input_str = node_ctor_inputs(schema)
        return f'torch::lazy::NodePtr node = torch::lazy::ReuseNode<{schema.node_name}>({node_ctor_input_str});\n        if (!node) {{\n            {self.shape_inference(func, schema)}\n            node = torch::lazy::MakeNode<{schema.node_name}>({node_ctor_input_str}, std::move(shapes));\n            CacheNode(node);\n        }}\n        '

    def create_lazy_tensor(self, first_tensor_name: Optional[str]=None) -> str:
        if False:
            for i in range(10):
                print('nop')
        if self.create_from_first_tensor:
            assert first_tensor_name is not None, 'Requires first tensor to create lazy tensor'
            return f'{first_tensor_name}.{self.create_tensor}'
        return f'{self.backend_namespace}::{self.create_tensor}'

    def return_aten_tensor(self, func: NativeFunction, schema: LazyIrSchema) -> str:
        if False:
            while True:
                i = 10
        returns_length = len(schema.returns)
        value_args = schema.filtered_args(values=True, scalars=False)
        value_types_names = [f'{a.name}' for a in value_args if not a.is_wrapped_scalar]
        first_tensor_name = value_types_names[0] if len(value_types_names) > 0 else None
        bridge_str = f'auto result = {self.create_aten_from_ltc_tensor}(\n                {self.create_lazy_tensor(first_tensor_name)}(std::move(node), *common_device));'
        if returns_length > 1:
            assert len(value_types_names) > 0, 'Code below assumes there is at least one tensor arg'
            bridge_str = f'std::vector<{self.lazy_tensor_ptr}> lazy_tensors;\n        for (int i = 0; i < {returns_length}; i++) {{\n            lazy_tensors.push_back({self.create_lazy_tensor(first_tensor_name)}({getValueT()}(node, i), *common_device));\n        }}\n        auto result = {self.tuple_aten_from_ltc_tensors}<{returns_length}>(lazy_tensors);'
        if schema.name.name.inplace or func.func.is_out_fn():
            assert returns_length == 1, f'We assumed there was no such case where an op is an in-place variant and has tuple outputs, but got tuple of len {returns_length}.'
            bridge_str = f'lazy_{first_tensor_name}->SetInPlaceIrValue(node);\n        auto& result = {first_tensor_name};'
        bridge_str += '\n        return result;'
        return bridge_str

    @method_with_native_function
    def __call__(self, func: NativeFunction) -> List[str]:
        if False:
            print('Hello World!')
        sig = kernel_signature(func, self.backend_index)
        metadata = self.backend_index.get_kernel(func)
        assert metadata is not None
        schema = LazyIrSchema(func.func, symint=metadata.supports_symint())
        return [f"    {sig.decl(name=f'{self.class_method_name}::{metadata.kernel}')} {{\n        {self.force_eager_fallback(func, schema, metadata, sig)}\n        {self.metrics(func, schema)}\n        {self.get_device(func, schema)}\n        {self.lazy_tensor_decls(func, schema)}\n        {self.build_ir_node(func, schema)}\n        {self.return_aten_tensor(func, schema)}\n    }}\n\n    "]

class ComputeShapeSignature:
    """
    Here we use the base name as the suffix of the signature to avoid generating for in-place variants.
    """

    def __init__(self, kernel_name: str, f: NativeFunction, *, symint: bool):
        if False:
            while True:
                i = 10
        self.__schema = LazyIrSchema(f.func, symint=symint)
        self.__dispatch_args = ', '.join([a.decl() for a in dispatcher.arguments(f.func, symint=symint)])
        self.__call_args = ', '.join([f'{arg.name}' for arg in self.__schema.filtered_args(generator=True)])
        self.__kernel_name = kernel_name

    def __decl_suffix(self) -> str:
        if False:
            return 10
        return f'{self.__kernel_name}({self.__dispatch_args})'

    def __call_suffix(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'{self.__kernel_name}({self.__call_args})'

    @property
    def shape_decl(self) -> str:
        if False:
            print('Hello World!')
        return f'TORCH_API std::vector<torch::lazy::Shape> compute_shape_{self.__decl_suffix()}'

    @property
    def shape_call(self) -> str:
        if False:
            while True:
                i = 10
        return f'torch::lazy::compute_shape_{self.__call_suffix()}'

@dataclass(frozen=True)
class GenLazyShapeInferenceDefinition:
    backend_index: BackendIndex
    tensor_class: str

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> List[str]:
        if False:
            i = 10
            return i + 15
        sig = kernel_signature(f, self.backend_index)
        metadata = self.backend_index.get_kernel(f)
        assert metadata is not None
        is_view_copy_op = 'view_copy' in f.tags
        is_structured = f.structured or f.structured_delegate is not None
        if is_structured or is_view_copy_op:
            return []
        else:
            shape_sig = ComputeShapeSignature(metadata.kernel, f, symint=metadata.supports_symint())
            return ['\n'.join([f'{shape_sig.shape_decl};'])]

def generate_non_native_lazy_ir_nodes(non_native: List[Dict[str, Any]], gen_lazy_ir: GenLazyIR) -> List[str]:
    if False:
        i = 10
        return i + 15
    'Generate the non-native lazy IR node classes'
    nodes = []
    for op in non_native:
        properties = LazyIrProperties('ShapeCache', 'CanBeReused', 'LowerDeclOnly')
        for p in op.get('properties', []):
            setattr(properties, p, True)
        schema = LazyIrSchema(FunctionSchema.parse(op['func']), properties, symint=True)
        schema.opkind = op.get('opkind')
        nodes.append(gen_lazy_ir.gen(schema)[0])
    return nodes