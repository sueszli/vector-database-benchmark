from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.ufunc as ufunc
from torchgen.api.translate import translate
from torchgen.api.types import BaseCType, Binding, CType, Expr, NamedCType, opmath_t, scalar_t, StructuredImplSignature, VectorizedCType
from torchgen.api.ufunc import UfunctorBindings
from torchgen.context import with_native_function
from torchgen.model import Argument, BaseTy, BaseType, DispatchKey, NativeFunctionsGroup, ScalarType, UfuncKey
from torchgen.utils import OrderedSet

@dataclass(frozen=True)
class UfunctorSignature:
    g: NativeFunctionsGroup
    scalar_tensor_idx: Optional[int]
    name: str

    def arguments(self) -> UfunctorBindings:
        if False:
            while True:
                i = 10
        return ufunc.ufunctor_arguments(self.g, scalar_tensor_idx=self.scalar_tensor_idx, scalar_t=scalar_t)

    def fields(self) -> List[Binding]:
        if False:
            print('Hello World!')
        return [b.rename(f'{b.name}_') for b in self.arguments().ctor]

    def returns_type(self) -> CType:
        if False:
            i = 10
            return i + 15
        return BaseCType(scalar_t)

    def decl_fields(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return '\n'.join((f'{f.type} {f.name};' for f in self.fields()))

    def inline_defn_ctor(self) -> str:
        if False:
            print('Hello World!')
        args_str = ', '.join((a.decl() for a in self.arguments().ctor))
        init_str = ', '.join((f'{a.name}_({a.name})' for a in self.arguments().ctor))
        return f'{self.name}({args_str}) : {init_str} {{}}'

    def decl_apply(self) -> str:
        if False:
            return 10
        args_str = ', '.join((a.decl() for a in self.arguments().apply))
        return f'{self.returns_type().cpp_type()} operator()({args_str}) const'

@dataclass(frozen=True)
class UfuncSignature:
    g: NativeFunctionsGroup
    name: str
    compute_t: CType

    def arguments(self) -> List[Binding]:
        if False:
            for i in range(10):
                print('nop')
        return ufunc.ufunc_arguments(self.g, compute_t=self.compute_t)

    def call(self, ctx: Sequence[Union[Binding, Expr]]) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f"{self.name}({', '.join((a.expr for a in translate(ctx, self.arguments())))})"

def eligible_for_binary_scalar_specialization(g: NativeFunctionsGroup) -> bool:
    if False:
        i = 10
        return i + 15
    num_tensors = sum((1 for a in g.functional.func.arguments.flat_non_out if a.type.is_tensor_like()))
    return num_tensors == 2

def compute_ufunc_cuda_functors(g: NativeFunctionsGroup) -> Tuple[Dict[ScalarType, Dict[UfuncKey, UfunctorSignature]], str]:
    if False:
        return 10
    ufunctor_sigs: Dict[ScalarType, Dict[UfuncKey, UfunctorSignature]] = {}
    ufunctors: List[str] = []
    loops = g.out.ufunc_inner_loop
    scalar_tensor_idx_lookup = {UfuncKey.CUDAFunctorOnSelf: 1, UfuncKey.CUDAFunctorOnOther: 0, UfuncKey.CUDAFunctor: None}
    if eligible_for_binary_scalar_specialization(g):
        keys = [UfuncKey.CUDAFunctorOnSelf, UfuncKey.CUDAFunctorOnOther, UfuncKey.CUDAFunctor]
    else:
        keys = [UfuncKey.CUDAFunctor]
        for k in [UfuncKey.CUDAFunctorOnSelf, UfuncKey.CUDAFunctorOnOther]:
            assert k not in loops, f'cannot use {k} on non-binary function'
    for k in keys:
        if k in loops:
            ufunctor_sig = UfunctorSignature(g, scalar_tensor_idx=scalar_tensor_idx_lookup[k], name=loops[k].name)
            for dtype in loops[k].supported_dtypes:
                ufunctor_sigs.setdefault(dtype, {})[k] = ufunctor_sig
            continue
        ufunc_name = None
        supported_dtypes: OrderedSet[ScalarType] = OrderedSet()
        for lk in [UfuncKey.ScalarOnly, UfuncKey.Generic]:
            if lk not in loops:
                continue
            if ufunc_name is None:
                ufunc_name = loops[lk].name
            else:
                assert ufunc_name == loops[lk].name, 'ScalarOnly and Generic must have same ufunc name'
            supported_dtypes |= loops[lk].supported_dtypes
        assert ufunc_name is not None
        name = f'{k}_{ufunc_name}'
        ufunctor_sig = UfunctorSignature(g, scalar_tensor_idx=scalar_tensor_idx_lookup[k], name=name)
        for dtype in supported_dtypes:
            ufunctor_sigs.setdefault(dtype, {})[k] = ufunctor_sig
        ufunc_sig = UfuncSignature(g, name=f'ufunc::{ufunc_name}', compute_t=BaseCType(opmath_t))
        apply_ctx = ufunctor_sig.fields() + ufunctor_sig.arguments().apply
        ufunctors.append(f'\ntemplate <typename scalar_t>\nstruct {ufunctor_sig.name} {{\n  using opmath_t = at::opmath_type<scalar_t>;\n  {ufunctor_sig.decl_fields()}\n  {ufunctor_sig.inline_defn_ctor()}\n  __device__ {ufunctor_sig.decl_apply()} {{\n    return {ufunc_sig.call(apply_ctx)};\n  }}\n}};\n')
    return (ufunctor_sigs, '\n'.join(ufunctors))

@dataclass(frozen=True)
class BinaryScalarSpecializationConfig:
    scalar_idx: int
    ctor_tensor: str
    ufunc_key: UfuncKey
BinaryScalarSpecializationConfigs = [BinaryScalarSpecializationConfig(scalar_idx=0, ctor_tensor='self', ufunc_key=UfuncKey.CUDAFunctorOnOther), BinaryScalarSpecializationConfig(scalar_idx=1, ctor_tensor='other', ufunc_key=UfuncKey.CUDAFunctorOnSelf)]

def compute_ufunc_cuda_dtype_body(g: NativeFunctionsGroup, dtype: ScalarType, inner_loops: Dict[UfuncKey, UfunctorSignature], parent_ctx: Sequence[Binding]) -> str:
    if False:
        i = 10
        return i + 15
    body = 'using opmath_t = at::opmath_type<scalar_t>;'
    body += 'if (false) {}\n'
    for config in BinaryScalarSpecializationConfigs:
        if config.ufunc_key not in inner_loops:
            continue
        ufunctor_sig = inner_loops[config.ufunc_key]
        scalar_idx = config.scalar_idx + 1
        ctx: List[Union[Expr, Binding]] = list(parent_ctx)
        ctx.append(Expr(expr=f'iter.scalar_value<opmath_t>({scalar_idx})', type=NamedCType(config.ctor_tensor, BaseCType(opmath_t))))
        ufunctor_ctor_exprs_str = ', '.join((a.expr for a in translate(ctx, ufunctor_sig.arguments().ctor)))
        body += f'else if (iter.is_cpu_scalar({scalar_idx})) {{\n  {ufunctor_sig.name}<scalar_t> ufunctor({ufunctor_ctor_exprs_str});\n  iter.remove_operand({scalar_idx});\n  gpu_kernel(iter, ufunctor);\n}}'
    ufunctor_sig = inner_loops[UfuncKey.CUDAFunctor]
    ufunctor_ctor_exprs_str = ', '.join((a.expr for a in translate(parent_ctx, ufunctor_sig.arguments().ctor)))
    body += f'\nelse {{\n  gpu_kernel(iter, {ufunctor_sig.name}<scalar_t>({ufunctor_ctor_exprs_str}));\n}}\n    '
    return body

@with_native_function
def compute_ufunc_cuda(g: NativeFunctionsGroup) -> str:
    if False:
        i = 10
        return i + 15
    (ufunctor_sigs, ufunctors) = compute_ufunc_cuda_functors(g)
    sig = StructuredImplSignature(g, ufunc.kernel_name(g, DispatchKey.CUDA))
    dtype_cases = []
    for (dtype, inner_ufunc_sigs) in ufunctor_sigs.items():
        dtype_cases.append(f'\nAT_DISPATCH_CASE(at::ScalarType::{dtype},\n  [&]() {{\n    {compute_ufunc_cuda_dtype_body(g, dtype, inner_ufunc_sigs, sig.arguments())}\n  }}\n)\n')
    dtype_cases_str = '\n'.join(dtype_cases)
    stub_sig = StubSignature(g)
    return f'\n{ufunctors}\n\n{stub_sig.type_defn()};\n{stub_sig.dispatch_decl()};\n\n{stub_sig.kernel_defn()} {{\n  AT_DISPATCH_SWITCH(iter.common_dtype(), "{sig.name}",\n    {dtype_cases_str}\n  );\n}}\nREGISTER_DISPATCH({stub_sig.name}, &{stub_sig.kernel_name});\n\n{sig.defn()} {{\n  {stub_sig.direct_call(sig.arguments())};\n}}\n'

@dataclass(frozen=True)
class StubSignature:
    g: NativeFunctionsGroup

    @property
    def name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'{str(self.g.functional.func.name.name)}_stub'

    @property
    def kernel_name(self) -> str:
        if False:
            print('Hello World!')
        return f'{str(self.g.functional.func.name.name)}_kernel'

    @property
    def type_name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'{str(self.g.functional.func.name.name)}_fn'

    def arguments(self) -> List[Binding]:
        if False:
            while True:
                i = 10
        return ufunc.stub_arguments(self.g)

    def type(self) -> str:
        if False:
            while True:
                i = 10
        cpp_args = self.arguments()
        return f"void(*)(TensorIteratorBase&, {', '.join((a.type for a in cpp_args))})"

    def dispatch_decl(self) -> str:
        if False:
            while True:
                i = 10
        return f'DECLARE_DISPATCH({self.type_name}, {self.name})'

    def dispatch_defn(self) -> str:
        if False:
            print('Hello World!')
        return f'DEFINE_DISPATCH({self.name})'

    def kernel_defn(self) -> str:
        if False:
            print('Hello World!')
        return f"void {self.kernel_name}(TensorIteratorBase& iter, {', '.join((a.defn() for a in self.arguments()))})"

    def type_defn(self) -> str:
        if False:
            while True:
                i = 10
        return f'using {self.type_name} = {self.type()}'

    def call(self, ctx: Sequence[Binding]) -> str:
        if False:
            i = 10
            return i + 15
        return f"{self.name}(device_type(), *this, {', '.join((a.expr for a in translate(ctx, self.arguments())))})"

    def direct_call(self, ctx: Sequence[Binding]) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f"{self.kernel_name}(*this, {', '.join((a.expr for a in translate(ctx, self.arguments())))})"

@with_native_function
def compute_ufunc_cpu(g: NativeFunctionsGroup) -> str:
    if False:
        print('Hello World!')
    stub_sig = StubSignature(g)
    sig = StructuredImplSignature(g, ufunc.kernel_name(g, DispatchKey.CPU))
    return f'\n{stub_sig.type_defn()};\n{stub_sig.dispatch_decl()};\n{stub_sig.dispatch_defn()};\n\n{sig.defn()} {{\n  {stub_sig.call(sig.arguments())};\n}}\n'

def compute_ufunc_cpu_dtype_body(g: NativeFunctionsGroup, dtype: ScalarType, inner_loops: Dict[UfuncKey, UfuncSignature], parent_ctx: Sequence[Binding]) -> str:
    if False:
        for i in range(10):
            print('nop')
    assert UfuncKey.CPUScalar in inner_loops, f'{dtype}, {inner_loops.keys()}'
    assert inner_loops.keys() <= {UfuncKey.CPUScalar, UfuncKey.CPUVector}
    scalar_loop = inner_loops[UfuncKey.CPUScalar]
    vec_loop = None
    if UfuncKey.CPUVector in inner_loops:
        vec_loop = inner_loops[UfuncKey.CPUVector]
    body = []
    ctx = []
    for b in parent_ctx:
        if isinstance(b.argument, Argument) and b.argument.type != BaseType(BaseTy.Scalar):
            continue
        body.append(f'auto _s_{b.name} = {b.name}.to<scalar_t>();')
        ctx.append(Expr(f'_s_{b.name}', NamedCType(b.nctype.name, BaseCType(scalar_t))))
    if vec_loop is not None:
        for b in parent_ctx:
            if isinstance(b.argument, Argument) and b.argument.type != BaseType(BaseTy.Scalar):
                continue
            body.append(f'auto _v_{b.name} = at::vec::Vectorized<scalar_t>(_s_{b.name});')
            ctx.append(Expr(f'_v_{b.name}', NamedCType(b.nctype.name, VectorizedCType(BaseCType(scalar_t)))))
    scalar_bindings = []
    vec_bindings = []
    for a in g.functional.func.arguments.flat_non_out:
        if not a.type.is_tensor_like():
            continue
        assert a.type == BaseType(BaseTy.Tensor)
        scalar_bindings.append(Binding(name=a.name, nctype=NamedCType(a.name, BaseCType(scalar_t)), argument=a))
        if vec_loop is not None:
            vec_bindings.append(Binding(name=a.name, nctype=NamedCType(a.name, VectorizedCType(BaseCType(scalar_t))), argument=a))

    def with_ctx(b: Sequence[Binding]) -> List[Union[Expr, Binding]]:
        if False:
            print('Hello World!')
        r: List[Union[Expr, Binding]] = []
        r.extend(ctx)
        r.extend(b)
        return r
    body_str = '\n'.join(body)
    if vec_loop is not None:
        return f"\n{body_str}\ncpu_kernel_vec(iter,\n  [=]({', '.join((b.decl() for b in scalar_bindings))}) {{ return {scalar_loop.call(with_ctx(scalar_bindings))}; }},\n  [=]({', '.join((b.decl() for b in vec_bindings))}) {{ return {vec_loop.call(with_ctx(vec_bindings))}; }}\n);\n"
    else:
        return f"\n{body_str}\ncpu_kernel(iter,\n  [=]({', '.join((b.decl() for b in scalar_bindings))}) {{ return {scalar_loop.call(with_ctx(scalar_bindings))}; }}\n);\n"

@with_native_function
def compute_ufunc_cpu_kernel(g: NativeFunctionsGroup) -> str:
    if False:
        i = 10
        return i + 15
    stub_sig = StubSignature(g)
    loops = g.out.ufunc_inner_loop
    ufunc_sigs: Dict[ScalarType, Dict[UfuncKey, UfuncSignature]] = {}
    for k in [UfuncKey.CPUScalar, UfuncKey.CPUVector]:
        lks = []
        if k in loops:
            lks.append(k)
        if UfuncKey.ScalarOnly in loops and k is UfuncKey.CPUScalar:
            lks.append(UfuncKey.ScalarOnly)
        if UfuncKey.Generic in loops:
            lks.append(UfuncKey.Generic)
        for lk in lks:
            for dtype in loops[lk].supported_dtypes:
                compute_t: CType
                if k is UfuncKey.CPUScalar:
                    compute_t = BaseCType(scalar_t)
                elif k is UfuncKey.CPUVector:
                    compute_t = VectorizedCType(BaseCType(scalar_t))
                else:
                    raise AssertionError()
                inner_ufunc_sigs = ufunc_sigs.setdefault(dtype, {})
                if k not in inner_ufunc_sigs:
                    inner_ufunc_sigs[k] = UfuncSignature(g, name=f'ufunc::{loops[lk].name}', compute_t=compute_t)
    dtype_cases = []
    for (dtype, inner_ufunc_sigs) in ufunc_sigs.items():
        dtype_cases.append(f'\nAT_DISPATCH_CASE(at::ScalarType::{dtype},\n  [&]() {{\n    {compute_ufunc_cpu_dtype_body(g, dtype, inner_ufunc_sigs, stub_sig.arguments())}\n  }}\n)\n')
    dtype_cases_str = '\n'.join(dtype_cases)
    return f'\nnamespace {{\n\n{stub_sig.kernel_defn()} {{\n  AT_DISPATCH_SWITCH(iter.common_dtype(), "{stub_sig.name}",\n    {dtype_cases_str}\n  );\n}}\n\n}} // anonymous namespace\n\n{stub_sig.type_defn()};\n{stub_sig.dispatch_decl()};\nREGISTER_DISPATCH({stub_sig.name}, &{stub_sig.kernel_name});\n'