from typing import List, Union
from torchgen.api import cpp
from torchgen.api.types import ArgName, ArrayRefCType, BaseCType, Binding, ConstRefCType, dimnameListT, intArrayRefT, iOptTensorListRefT, iTensorListRefT, NamedCType, OptionalCType, optionalIntArrayRefT, optionalScalarRefT, optionalTensorRefT, scalarT, tensorT
from torchgen.model import Argument, BaseTy, BaseType, ListType, NativeFunctionsGroup, OptionalType, SelfArgument, TensorOptionsArguments, Type
from torchgen.utils import assert_never

def argumenttype_type(t: Type, *, mutable: bool, binds: ArgName) -> NamedCType:
    if False:
        return 10
    r = cpp.valuetype_type(t, symint=False, binds=binds)
    if r is not None:
        return r
    if isinstance(t, BaseType):
        if t.name == BaseTy.Tensor:
            return NamedCType(binds, ConstRefCType(BaseCType(tensorT)))
        elif t.name == BaseTy.Scalar:
            return NamedCType(binds, ConstRefCType(BaseCType(scalarT)))
        else:
            raise AssertionError(f'base type should have been value type {t}')
    elif isinstance(t, OptionalType):
        if t.elem == BaseType(BaseTy.Tensor):
            return NamedCType(binds, BaseCType(optionalTensorRefT))
        elif t.elem == BaseType(BaseTy.Scalar):
            return NamedCType(binds, BaseCType(optionalScalarRefT))
        elif isinstance(t.elem, ListType) and str(t.elem.elem) == 'int':
            return NamedCType(binds, BaseCType(optionalIntArrayRefT))
        elem = argumenttype_type(t.elem, mutable=mutable, binds=binds)
        return NamedCType(binds, OptionalCType(elem.type))
    elif isinstance(t, ListType):
        if t.elem == BaseType(BaseTy.Tensor):
            return NamedCType(binds, ConstRefCType(BaseCType(iTensorListRefT)))
        elif t.elem == OptionalType(BaseType(BaseTy.Tensor)):
            return NamedCType(binds, BaseCType(iOptTensorListRefT))
        elif str(t.elem) == 'int':
            return NamedCType(binds, BaseCType(intArrayRefT))
        elif str(t.elem) == 'Dimname':
            return NamedCType(binds, BaseCType(dimnameListT))
        elem = argumenttype_type(t.elem, mutable=mutable, binds=binds)
        return NamedCType(binds, ArrayRefCType(elem.type))
    else:
        raise AssertionError(f'unrecognized type {repr(t)}')

def argument_type(a: Argument, *, binds: ArgName) -> NamedCType:
    if False:
        i = 10
        return i + 15
    return argumenttype_type(a.type, mutable=a.is_write, binds=binds)

def argument(a: Union[Argument, SelfArgument, TensorOptionsArguments]) -> List[Binding]:
    if False:
        i = 10
        return i + 15
    if isinstance(a, Argument):
        return [Binding(nctype=argument_type(a, binds=a.name), name=a.name, default=None, argument=a)]
    elif isinstance(a, SelfArgument):
        return argument(a.argument)
    elif isinstance(a, TensorOptionsArguments):
        raise AssertionError("structured kernels don't support TensorOptions yet")
    else:
        assert_never(a)

def impl_arguments(g: NativeFunctionsGroup) -> List[Binding]:
    if False:
        for i in range(10):
            print('nop')
    args: List[Union[Argument, TensorOptionsArguments, SelfArgument]] = []
    if g.out.precomputed:
        non_out_args_replaced: List[Union[Argument, TensorOptionsArguments, SelfArgument]] = []
        for a in g.out.func.arguments.non_out:
            if isinstance(a, Argument) and a.name in g.out.precomputed.replace:
                for replacement in g.out.precomputed.replace[a.name]:
                    non_out_args_replaced.append(replacement)
            else:
                non_out_args_replaced.append(a)
        args.extend(non_out_args_replaced)
        args.extend(g.out.precomputed.add)
    else:
        args.extend(g.out.func.arguments.non_out)
    args.extend(g.out.func.arguments.out)
    return [r for arg in args for r in argument(arg)]

def meta_arguments(g: NativeFunctionsGroup) -> List[Binding]:
    if False:
        for i in range(10):
            print('nop')
    args: List[Union[Argument, TensorOptionsArguments, SelfArgument]] = []
    args.extend(g.functional.func.arguments.non_out)
    return [r for arg in args for r in argument(arg)]

def out_arguments(g: NativeFunctionsGroup) -> List[Binding]:
    if False:
        i = 10
        return i + 15
    args: List[Union[Argument, TensorOptionsArguments, SelfArgument]] = []
    args.extend(g.out.func.arguments.out)
    return [r for arg in args for r in argument(arg)]