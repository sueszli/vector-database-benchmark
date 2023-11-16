import itertools
import textwrap
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union
import torchgen.api.cpp as cpp
import torchgen.api.meta as meta
import torchgen.api.structured as structured
from torchgen.api.translate import translate
from torchgen.api.types import BaseCType, Binding, ConstRefCType, CppSignature, CppSignatureGroup, DispatcherSignature, Expr, kernel_signature, MutRefCType, NamedCType, NativeSignature, tensorT
from torchgen.context import method_with_native_function, native_function_manager
from torchgen.model import Argument, BackendIndex, DeviceCheckType, DispatchKey, gets_generated_out_inplace_wrapper, is_cuda_dispatch_key, NativeFunction, NativeFunctionsGroup, SchemaKind, TensorOptionsArguments
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import assert_never, mapMaybe, Target

def gen_registration_headers(backend_index: BackendIndex, per_operator_headers: bool, rocm: bool) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    if per_operator_headers:
        headers = ['#include <ATen/ops/as_strided_native.h>']
    else:
        headers = ['#include <ATen/NativeFunctions.h>']
    if backend_index.dispatch_key in (DispatchKey.CPU, DispatchKey.Meta):
        headers.append('#include <ATen/EmptyTensor.h>')
    elif backend_index.dispatch_key == DispatchKey.CUDA:
        if rocm:
            headers.append('#include <ATen/hip/EmptyTensor.h>')
        else:
            headers.append('#include <ATen/cuda/EmptyTensor.h>')
    elif backend_index.dispatch_key == DispatchKey.MPS:
        headers.append('#include <ATen/mps/EmptyTensor.h>')
    elif per_operator_headers:
        headers += ['#include <ATen/ops/empty.h>', '#include <ATen/ops/empty_strided.h>', '#include <ATen/ops/_copy_from_and_resize.h>', '#include <ATen/ops/_copy_from.h>']
    else:
        headers.append('#include <ATen/Functions.h>')
    return headers

def gen_empty_impl_names(backend_index: BackendIndex) -> Tuple[Optional[str], Optional[str]]:
    if False:
        while True:
            i = 10
    empty_impl = None
    empty_strided_impl = None
    if backend_index.dispatch_key in (DispatchKey.Meta, DispatchKey.CPU, DispatchKey.CUDA, DispatchKey.MPS):
        dispatch = str(backend_index.dispatch_key).lower()
        empty_impl = f'at::detail::empty_{dispatch}'
        empty_strided_impl = f'at::detail::empty_strided_{dispatch}'
    elif backend_index.dispatch_key in (DispatchKey.CompositeExplicitAutogradNonFunctional, DispatchKey.QuantizedCPU, DispatchKey.QuantizedCUDA):
        empty_impl = 'at::empty'
        empty_strided_impl = 'at::empty_strided'
    return (empty_impl, empty_strided_impl)

def gen_create_out_helper(backend_index: BackendIndex) -> List[str]:
    if False:
        while True:
            i = 10
    if backend_index.dispatch_key == DispatchKey.Meta:
        empty_options = 'options.device(at::kMeta)'
    else:
        empty_options = 'options'
    (empty_impl, empty_strided_impl) = gen_empty_impl_names(backend_index)
    if empty_impl is None:
        return []
    return [f'\nTensor create_out(IntArrayRef sizes, IntArrayRef strides, const TensorOptions &options) {{\n  if (strides.empty()) {{\n      return {empty_impl}(sizes, {empty_options});\n  }} else {{\n      return {empty_strided_impl}(sizes, strides, {empty_options});\n  }}\n}}\n']

def gen_maybe_create_proxy_helper(backend_index: BackendIndex) -> List[str]:
    if False:
        i = 10
        return i + 15
    (_, empty_strided_impl) = gen_empty_impl_names(backend_index)
    return [] if empty_strided_impl is None else [f'\nc10::optional<Tensor> maybe_create_proxy(const Tensor &out, IntArrayRef sizes, IntArrayRef strides, const TensorOptions &options) {{\n  if (out.strides() != strides) {{\n    return {empty_strided_impl}(sizes, strides, options);\n  }}\n  return c10::nullopt;\n}}\n']

def gen_resize_out_helper(backend_index: BackendIndex) -> List[str]:
    if False:
        i = 10
        return i + 15
    if backend_index.dispatch_key == DispatchKey.CompositeExplicitAutogradNonFunctional:
        return []
    return ['\nvoid resize_out(const Tensor &out, IntArrayRef sizes, IntArrayRef strides, const TensorOptions &options) {\n  TORCH_CHECK(options.dtype() == out.dtype(),\n      "Expected out tensor to have dtype ", options.dtype(), ", but got ", out.dtype(), " instead");\n  TORCH_CHECK(options.device() == out.device(),\n      "Expected out tensor to have device ", options.device(), ", but got ", out.device(), " instead");\n  const bool resized = at::native::resize_output(out, sizes);\n  // Only restride if a resize occurred; otherwise we ignore the (advisory)\n  // strides from the meta function and directly use the output tensor\'s\n  // preexisting strides\n  if (resized) {\n    if (!strides.empty()) {\n      TORCH_INTERNAL_ASSERT(!options.memory_format_opt().has_value());\n      // TODO: avoid the redispatch here\n      out.as_strided_(sizes, strides);\n    } else if (options.memory_format_opt().has_value()) {\n      out.unsafeGetTensorImpl()->empty_tensor_restride(*options.memory_format_opt());\n    }\n  }\n}\n']

def gen_check_inplace_helper(backend_index: BackendIndex) -> List[str]:
    if False:
        i = 10
        return i + 15
    return ['\nvoid check_inplace(const Tensor &self, IntArrayRef sizes, const TensorOptions &options) {\n  // These checks are needed on those operators that:\n  //   1) don\'t use \'TensorIterator\' (e.g. \'addmm\' and \'baddbmm\')\n  //   2) have particular typing rules (e.g. \'cumsum\' and \'cumprod\')\n  // For other operators (e.g. \'add\'), \'TensorIterator\' already checks\n  // these things separately.\n  TORCH_CHECK(options.dtype() == self.dtype(),\n      "Bad in-place call: ",\n      "input tensor dtype ", self.dtype(), " and output tensor dtype ", options.dtype(), " should match");\n  TORCH_CHECK(options.device() == self.device(),\n      "Bad in-place call: ",\n      "input tensor device ", self.device(), " and output tensor device ", options.device(), " should match");\n  TORCH_CHECK(sizes == self.sizes(),\n      "Bad in-place call: ",\n      "input tensor size ", self.sizes(), " and output tensor size ", sizes, " should match");\n}\n']

def gen_registration_helpers(backend_index: BackendIndex) -> List[str]:
    if False:
        i = 10
        return i + 15
    return [*gen_create_out_helper(backend_index), *gen_resize_out_helper(backend_index), *gen_check_inplace_helper(backend_index), *gen_maybe_create_proxy_helper(backend_index)]

@dataclass(frozen=True)
class RegisterDispatchKey:
    backend_index: BackendIndex
    target: Literal[Target.ANONYMOUS_DEFINITION, Target.NAMESPACED_DEFINITION, Target.NAMESPACED_DECLARATION, Target.REGISTRATION]
    selector: SelectiveBuilder
    rocm: bool
    symint: bool
    class_method_name: Optional[str]
    skip_dispatcher_op_registration: bool

    @staticmethod
    def gen_device_check(type: DeviceCheckType, args: List[Argument], method_name: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        if type == DeviceCheckType.NoCheck:
            return '  // No device check\n'
        device_check = 'c10::optional<Device> common_device = nullopt;\n'
        device_check += '(void)common_device; // Suppress unused variable warning\n'
        for arg in args:
            if arg.type.is_tensor_like():
                device_check += f'\n  c10::impl::check_and_update_common_device(common_device, {arg.name}, "{method_name}", "{arg.name}");'
        return device_check

    @method_with_native_function
    def __call__(self, f: Union[NativeFunctionsGroup, NativeFunction]) -> List[str]:
        if False:
            while True:
                i = 10
        if isinstance(f, NativeFunctionsGroup):
            g: NativeFunctionsGroup = f
            if g.structured:
                return self.gen_structured(g)
            else:
                return list(mapMaybe(lambda f: self.gen_unstructured(f, g), g.functions()))
        elif isinstance(f, NativeFunction):
            r = self.gen_unstructured(f)
            return [] if r is None else [r]
        else:
            assert_never(f)

    def wrapper_kernel_sig(self, f: NativeFunction) -> Union[NativeSignature, DispatcherSignature]:
        if False:
            for i in range(10):
                print('nop')
        return DispatcherSignature.from_schema(f.func, prefix=f'wrapper_{self.backend_index.dispatch_key}_{f.func.name.overload_name}_', symint=self.symint)

    def gen_out_inplace_wrapper(self, f: NativeFunction, g: Optional[NativeFunctionsGroup]) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        if g is None:
            return None
        k = f.func.kind()
        if k is SchemaKind.inplace:
            copy_op = 'at::_copy_from'
        elif k is SchemaKind.out:
            copy_op = 'at::_copy_from_and_resize'
        else:
            raise AssertionError('gen_out_inplace_wrapper called on a functional op')
        sig = self.wrapper_kernel_sig(f)
        name = sig.name()
        func_res = f'{name}_tmp'
        return_names = cpp.return_names(f)
        if len(return_names) > 1:
            updates = '\n  '.join((f'{copy_op}(std::get<{i}>({func_res}), {ret_name});' for (i, ret_name) in enumerate(return_names)))
            returns = f"{sig.returns_type().cpp_type()}({', '.join(return_names)})"
        elif len(return_names) == 1:
            ret_name = return_names[0]
            updates = f'{copy_op}({func_res}, {ret_name});'
            returns = ret_name
        else:
            assert len(f.func.arguments.out) == 1
            returns = ''
            out_arg = f.func.arguments.out[0]
            if out_arg.type.is_list_like():
                updates = f'    for (int64_t i = 0; i < {func_res}.size(); ++i) {{\n        {copy_op}({func_res}[i], {out_arg.name}[i]);\n    }}'
            else:
                updates = f'{copy_op}({func_res}, {out_arg.name});'
        functional_sig = self.wrapper_kernel_sig(g.functional)
        wrapper_name = sig.name()
        return f"{sig.defn(name=wrapper_name)} {{\n  auto {func_res} = {functional_sig.name()}({', '.join((e.expr for e in translate(sig.arguments(), functional_sig.arguments())))});\n  {updates}\n  return {returns};\n}}\n"

    def gen_structured(self, g: NativeFunctionsGroup) -> List[str]:
        if False:
            while True:
                i = 10
        metadata = self.backend_index.get_kernel(g)
        if self.backend_index.dispatch_key == DispatchKey.Meta:
            assert not self.backend_index.has_kernel(g.out), 'Do not explicitly specify Meta dispatch key on structured functions, they will be automatically generated for you'
        elif self.backend_index.dispatch_key == DispatchKey.CompositeExplicitAutogradNonFunctional:
            assert not self.backend_index.has_kernel(g.out), 'Do not explicitly specify CompositeExplicitAutograd dispatch key on structured functions, they will be automatically generated for you'
        elif metadata is None or not metadata.structured:
            return list(mapMaybe(lambda f: self.gen_unstructured(f, g), g.functions()))
        structured_gen = StructuredRegisterDispatchKey(self.backend_index, self.target, self.selector, self.rocm, self.symint, self.class_method_name, self.skip_dispatcher_op_registration, g)
        return list(mapMaybe(structured_gen.gen_one, g.functions()))

    def gen_unstructured(self, f: NativeFunction, g: Optional[NativeFunctionsGroup]=None) -> Optional[str]:
        if False:
            return 10
        with native_function_manager(f):
            inplace_meta = False
            gets_out_inplace_wrapper = False
            if not self.backend_index.has_kernel(f):
                if self.backend_index.dispatch_key == DispatchKey.Meta and f.func.kind() is SchemaKind.inplace and (not f.has_composite_kernel) and (len(f.func.returns) == 1):
                    inplace_meta = True
                elif not self.backend_index.use_out_as_primary and g is not None and gets_generated_out_inplace_wrapper(f, g, self.backend_index):
                    gets_out_inplace_wrapper = True
                else:
                    return None
            if f.manual_kernel_registration:
                return None
            if self.target is Target.REGISTRATION and (not self.selector.is_native_function_selected(f)):
                return None
            sig = self.wrapper_kernel_sig(f)
            name = sig.name()
            returns_type = sig.returns_type().cpp_type()
            args = sig.arguments()
            args_str = ', '.join((a.defn() for a in args))
            cpp_sig_group = CppSignatureGroup.from_native_function(f, method=False, fallback_binding=False)
            if self.target is Target.NAMESPACED_DECLARATION:
                result = ''
                for cpp_sig in cpp_sig_group.signatures(symint=self.symint):
                    result += f'TORCH_API {cpp_sig.decl()};\n'
                return result
            elif self.target is Target.NAMESPACED_DEFINITION:

                def generate_defn(cpp_sig: CppSignature) -> str:
                    if False:
                        for i in range(10):
                            print('nop')
                    return f"\n{cpp_sig.defn()} {{\nreturn {sig.name()}({', '.join((e.expr for e in translate(cpp_sig.arguments(), sig.arguments())))});\n}}\n"
                result = ''
                for cpp_sig in cpp_sig_group.signatures(symint=self.symint):
                    result += generate_defn(cpp_sig)
                return result
            elif self.target is Target.ANONYMOUS_DEFINITION:
                if inplace_meta:
                    assert f.func.arguments.self_arg is not None
                    self_arg_name = f.func.arguments.self_arg.argument.name
                    return f'\n{returns_type} {name}({args_str}) {{\n  TORCH_CHECK_NOT_IMPLEMENTED({self_arg_name}.is_meta(),\n    "Cannot inplace into non-meta tensor with meta tensor argument");\n  return {self_arg_name};\n}}\n'
                if gets_out_inplace_wrapper:
                    return self.gen_out_inplace_wrapper(f, g)
                metadata = self.backend_index.get_kernel(f)
                if metadata is None:
                    return None
                if self.class_method_name is None:
                    impl_name = f'{metadata.cpp_namespace}::{metadata.kernel}'
                else:
                    impl_name = f'{metadata.cpp_namespace}::{self.class_method_name}::{metadata.kernel}'
                kernel_sig = kernel_signature(f, self.backend_index)
                args_exprs_str = ', '.join((e.expr for e in translate(sig.arguments(), kernel_sig.arguments(), method=False)))
                device_check = '  // No device check\n'
                if self.backend_index.device_guard:
                    device_check_args = itertools.chain(f.func.arguments.out, f.func.arguments.flat_positional)
                    device_check = RegisterDispatchKey.gen_device_check(f.device_check, list(device_check_args), name)
                device_guard = '// DeviceGuard omitted'
                if f.device_guard and self.backend_index.device_guard:
                    has_tensor_options = any((isinstance(a, TensorOptionsArguments) for a in f.func.arguments.non_out))
                    if has_tensor_options:
                        device_guard = '\n  const DeviceGuard device_guard(device_or_default(device));'
                        if is_cuda_dispatch_key(self.backend_index.dispatch_key):
                            device_guard = f'globalContext().lazyInitCUDA();\n{device_guard}'
                    else:
                        self_arg = [f.func.arguments.self_arg.argument] if f.func.arguments.self_arg is not None else []
                        candidate_args = itertools.chain(self_arg, f.func.arguments.out, f.func.arguments.flat_positional)
                        device_of = next((f'{a.name}' for a in candidate_args if a.type.is_tensor_like()), None)
                        if device_of is not None:
                            device_guard = f'const OptionalDeviceGuard device_guard(device_of({device_of}));'
                return f'namespace {{\n\n{returns_type} {name}({args_str}) {{\n  {device_check}\n\n  {device_guard}\n  return {impl_name}({args_exprs_str});\n}}\n\n}} // anonymous namespace\n'
            elif self.target is Target.REGISTRATION:
                if f.manual_kernel_registration or self.skip_dispatcher_op_registration:
                    return None
                else:
                    payload = f'TORCH_FN({name})'
                    return f'm.impl("{f.func.name}",\n{payload});\n'
            else:
                assert_never(self.target)

@dataclass(frozen=True)
class StructuredRegisterDispatchKey(RegisterDispatchKey):
    g: NativeFunctionsGroup

    def gen_class_set_output_functions(self, k: SchemaKind, parent_class: str, generate_super: bool) -> str:
        if False:
            for i in range(10):
                print('nop')
        if generate_super:
            set_output_super = f'{parent_class}::set_output_raw_strided(output_idx, sizes, strides, options, names);'
        else:
            set_output_super = ''

        def gen_set_output_function(name: str, maybe_create_proxy: bool) -> str:
            if False:
                print('Hello World!')
            return f"\nvoid set_output_{name}(\n    int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,\n    TensorOptions options, DimnameList names\n) override {{\n{textwrap.indent(self.gen_class_set_output_body(k, maybe_create_proxy), '    ')}\n    if (!names.empty()) {{\n      namedinference::propagate_names(outputs_[output_idx], names);\n    }}\n    // super must happen after, so that downstream can use maybe_get_output\n    // to retrieve the output\n{textwrap.indent(set_output_super, '    ')}\n}}\n"
        return f"\n{gen_set_output_function('strided', maybe_create_proxy=True)}\n{gen_set_output_function('raw_strided', maybe_create_proxy=False)}\n"

    def gen_class_set_output_body(self, k: SchemaKind, maybe_create_proxy: bool) -> str:
        if False:
            return 10
        if self.backend_index.dispatch_key in [DispatchKey.CUDA, DispatchKey.MPS, DispatchKey.CompositeExplicitAutogradNonFunctional]:
            maybe_set_guard = '\nauto current_device = guard_.current_device();\nif (C10_UNLIKELY(current_device.has_value())) {\n  TORCH_INTERNAL_ASSERT(*current_device == options.device(),\n    "structured kernels don\'t support multi-device outputs");\n} else {\n  guard_.reset_device(options.device());\n}\n'
            maybe_set_guard_line = maybe_set_guard + '\n'
        else:
            maybe_set_guard_line = maybe_set_guard = ''
        if maybe_create_proxy:
            create_proxy = '\nauto maybe_proxy = maybe_create_proxy(out, sizes, strides, options);\nif (C10_UNLIKELY(maybe_proxy.has_value())) {\n    proxy_outputs_[output_idx] = std::move(maybe_proxy).value();\n}\n'
        else:
            create_proxy = ''
        if k is SchemaKind.functional:
            assert self.backend_index.dispatch_key in (DispatchKey.Meta, DispatchKey.CPU, DispatchKey.CUDA, DispatchKey.MPS, DispatchKey.CompositeExplicitAutogradNonFunctional)
            return f'{maybe_set_guard_line}\noutputs_[output_idx] = create_out(sizes, strides, options);'
        elif k is SchemaKind.inplace:
            return f'{maybe_set_guard_line}\nconst auto& out = outputs_[output_idx].get();\ncheck_inplace(out, sizes, options);\n{create_proxy}'
        elif k is SchemaKind.out:
            return f'{maybe_set_guard_line}\nconst auto& out = outputs_[output_idx].get();\nresize_out(out, sizes, strides, options);\n{create_proxy}'
        elif k is SchemaKind.mutable or k is SchemaKind.scratch:
            raise AssertionError(f'{k} structured operators are currently not supported')
        else:
            assert_never(k)

    def gen_class_ctor(self, k: SchemaKind, class_name: str, returns: int) -> str:
        if False:
            i = 10
            return i + 15
        if k is SchemaKind.functional:
            return ''
        elif k is SchemaKind.inplace:
            return f'{class_name}(Tensor& self) : outputs_{{std::ref(self)}} {{}}'
        elif k is SchemaKind.out:
            out_args = ', '.join((f'Tensor& out{i}' for i in range(returns)))
            out_refs = ', '.join((f'std::ref(out{i})' for i in range(returns)))
            return f'{class_name}({out_args}) : outputs_{{ {out_refs} }} {{}}'
        elif k is SchemaKind.mutable or k is SchemaKind.scratch:
            raise AssertionError(f'{k} structured operators are currently not supported')
        else:
            assert_never(k)

    def gen_class(self, f: NativeFunction, k: SchemaKind, *, class_name: str, parent_class: str, generate_super: bool) -> str:
        if False:
            i = 10
            return i + 15
        if k is SchemaKind.functional:
            output_type = 'Tensor'
            output_value = 'outputs_[output_idx]'
            proxy_field = ''
        elif k is SchemaKind.inplace:
            output_type = 'std::reference_wrapper<Tensor>'
            output_value = 'proxy_outputs_[output_idx].has_value() ? *proxy_outputs_[output_idx] : outputs_[output_idx].get()'
            proxy_field = f'std::array<c10::optional<Tensor>, {len(f.func.returns)}> proxy_outputs_;'
        elif k is SchemaKind.out:
            output_type = 'std::reference_wrapper<Tensor>'
            output_value = 'proxy_outputs_[output_idx].has_value() ? *proxy_outputs_[output_idx] : outputs_[output_idx].get()'
            proxy_field = f'std::array<c10::optional<Tensor>, {len(f.func.returns)}> proxy_outputs_;'
        if self.backend_index.dispatch_key == DispatchKey.CUDA:
            if self.rocm:
                guard_field = 'c10::hip::OptionalHIPGuardMasqueradingAsCUDA guard_;'
            else:
                guard_field = 'c10::cuda::OptionalCUDAGuard guard_;'
        elif self.backend_index.dispatch_key == DispatchKey.CompositeExplicitAutogradNonFunctional:
            guard_field = 'c10::OptionalDeviceGuard guard_;'
        elif self.backend_index.dispatch_key == DispatchKey.MPS:
            guard_field = 'c10::OptionalDeviceGuard guard_;'
        else:
            guard_field = ''
        indent = ' ' * 4
        class_ctor_str = self.gen_class_ctor(k, class_name, len(f.func.returns))
        lines = (f'struct {class_name} final : public {parent_class} {{', f'{textwrap.indent(class_ctor_str, indent)}', f'{textwrap.indent(self.gen_class_set_output_functions(k, parent_class, generate_super), indent)}', '    const Tensor& maybe_get_output(int64_t output_idx) override {', f'      return {output_value};\n', '    }', f'    std::array<{output_type}, {len(f.func.returns)}> outputs_;', f'{textwrap.indent(proxy_field, indent)}', f'{textwrap.indent(guard_field, indent)}', '};')
        return '\n'.join((line for line in lines if line))

    @method_with_native_function
    def gen_one(self, f: NativeFunction) -> Optional[str]:
        if False:
            print('Hello World!')
        assert not f.manual_kernel_registration
        if self.target is Target.REGISTRATION and (not self.selector.is_native_function_selected(f)):
            return None
        if self.backend_index.dispatch_key == DispatchKey.CompositeExplicitAutogradNonFunctional and f.func.kind() is SchemaKind.out:
            return None
        cpp_sig_group = CppSignatureGroup.from_native_function(f, method=False, fallback_binding=False)
        kern = self.backend_index.get_kernel(f)
        sig = NativeSignature(f.func, prefix=f'wrapper_{self.backend_index.dispatch_key}_', symint=kern is not None and kern.supports_symint())
        if self.target is Target.NAMESPACED_DECLARATION:
            result = ''
            for cpp_sig in cpp_sig_group.signatures(symint=self.symint):
                result += f'TORCH_API {cpp_sig.decl()};\n'
            return result
        elif self.target is Target.NAMESPACED_DEFINITION:

            def generate_defn(cpp_sig: CppSignature) -> str:
                if False:
                    while True:
                        i = 10
                return f"\n{cpp_sig.defn()} {{\nreturn {sig.name()}({', '.join((e.expr for e in translate(cpp_sig.arguments(), sig.arguments())))});\n}}\n"
            result = ''
            for cpp_sig in cpp_sig_group.signatures(symint=self.symint):
                result += generate_defn(cpp_sig)
            return result
        elif self.target is Target.ANONYMOUS_DEFINITION:
            k = f.func.kind()
            sig_body = []
            context: List[Union[Binding, Expr]] = list(sig.arguments())
            if self.backend_index.dispatch_key is DispatchKey.Meta:
                class_name = f'structured_{meta.name(self.g)}_meta_{k.name}'
                parent_class = f'at::meta::structured_{meta.name(self.g)}'
            elif self.backend_index.dispatch_key is DispatchKey.CompositeExplicitAutogradNonFunctional:
                class_name = f'structured_{meta.name(self.g)}_default_backend_{k.name}'
                parent_class = f'at::meta::structured_{meta.name(self.g)}'
            else:
                metadata = self.backend_index.get_kernel(self.g)
                assert metadata is not None
                class_name = f'structured_{metadata.kernel}_{k.name}'
                parent_class = f'{metadata.cpp_namespace}::structured_{metadata.kernel}'
            if self.backend_index.device_guard:
                device_check_args = itertools.chain(f.func.arguments.out, f.func.arguments.flat_positional)
                sig_body.append(RegisterDispatchKey.gen_device_check(f.device_check, list(device_check_args), sig.name()))
            if k is SchemaKind.functional:
                sig_body.append(f'{class_name} op;')
            elif k is SchemaKind.inplace:
                sig_body.append(f'{class_name} op(self);')
            elif k is SchemaKind.out:
                out_args_str = ', '.join((a.name for a in f.func.arguments.out))
                sig_body.append(f'{class_name} op({out_args_str});')
            meta_exprs = ', '.join((e.expr for e in translate(context, structured.meta_arguments(self.g), method=False)))
            if self.g.out.precomputed:
                sig_body.append(f'auto precompute = op.meta({meta_exprs});')
                precomputed_values = [*self.g.out.precomputed.replace.values(), self.g.out.precomputed.add]
                for precomputed_elems in precomputed_values:
                    for arg in precomputed_elems:
                        context.append(Expr(expr=f'precompute.{arg.name}', type=structured.argument_type(arg, binds=arg.name)))
                sig_body.append('(void)precompute;')
            else:
                sig_body.append(f'op.meta({meta_exprs});')
            out_args = structured.out_arguments(self.g)
            for (i, out_arg) in enumerate(out_args):
                assert ConstRefCType(BaseCType(tensorT)) == out_arg.nctype.type
                if k is SchemaKind.out:
                    expr = f'op.maybe_get_output({i})'
                else:
                    expr = f'op.outputs_[{i}]'
                context.append(Expr(expr=expr, type=NamedCType(out_arg.nctype.name, MutRefCType(BaseCType(tensorT)))))
            if self.backend_index.dispatch_key == DispatchKey.CompositeExplicitAutogradNonFunctional:
                out_sig_group = CppSignatureGroup.from_native_function(self.g.out, method=False, fallback_binding=f.manual_cpp_binding)
                out_sig = out_sig_group.most_faithful_signature()
                api_name = out_sig.name()
                out_exprs = ', '.join((e.expr for e in translate(context, out_sig.arguments(), method=False)))
                sig_body.append(f'at::{api_name}({out_exprs});')
            elif self.backend_index.dispatch_key != DispatchKey.Meta:
                impl_exprs = ', '.join((e.expr for e in translate(context, structured.impl_arguments(self.g), method=False)))
                sig_body.append(f'op.impl({impl_exprs});')
            if k is SchemaKind.out or k is SchemaKind.inplace:
                for i in range(len(f.func.returns)):
                    sig_body.append(f'if (op.proxy_outputs_[{i}].has_value()) op.outputs_[{i}].get().copy_(*op.proxy_outputs_[{i}]);')
            if k is SchemaKind.functional:
                if len(f.func.returns) == 1:
                    ret_expr = 'std::move(op.outputs_[0])'
                else:
                    moved = ', '.join((f'std::move(op.outputs_[{i}])' for i in range(len(f.func.returns))))
                    ret_expr = f'std::make_tuple({moved})'
            elif k is SchemaKind.inplace:
                ret_expr = 'self'
            elif k is SchemaKind.out:
                if len(f.func.returns) == 1:
                    ret_expr = f.func.arguments.out[0].name
                else:
                    refs = ', '.join((a.name for a in f.func.arguments.out))
                    ret_expr = f'std::forward_as_tuple({refs})'
            sig_body.append(f'return {ret_expr};')
            sig_body_str = '\n'.join(sig_body)
            return f'{self.gen_class(f, k, class_name=class_name, parent_class=parent_class, generate_super=self.g.out.structured_inherits is not None)}\n\n{sig.defn()} {{\n{sig_body_str}\n}}\n'
        elif self.target is Target.REGISTRATION:
            return f'm.impl("{f.func.name}", TORCH_FN({sig.name()}));'
        else:
            assert_never(self.target)
            return None