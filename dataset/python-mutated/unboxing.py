from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple
from torchgen.api.types import Binding, CType, NamedCType
from torchgen.model import Argument, BaseTy, BaseType, ListType, NativeFunction, OptionalType, Type
connector = '\n\t'

def name(f: NativeFunction) -> str:
    if False:
        return 10
    return f.func.name.unambiguous_name()

@dataclass(frozen=True)
class Unboxing:
    """
    Takes a sequence of Bindings and unbox EValues to these Bindings. Return generated code that performs correct unboxing.
    A sample generated code:
    // aten::mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    void mul_out(EValue** stack) {
        EValue& self = *stack[0];
        EValue& other = *stack[1];
        EValue& out = *stack[2];
        const torch::executor::Tensor & self_base = self.to<torch::executor::Tensor>();
        const torch::executor::Tensor & other_base = other.to<torch::executor::Tensor>();
        torch::executor::Tensor & out_base = out.to<torch::executor::Tensor>();

        EXECUTORCH_SCOPE_PROF("native_call_mul.out");
        torch::executor::mul_outf(self_base, other_base, out_base);


    }
    """
    argument_type_gen: Callable[..., NamedCType]

    def convert_arguments(self, args: Sequence[Binding]) -> Tuple[List[Binding], List[str]]:
        if False:
            print('Hello World!')
        code_list = [f'EValue& {args[i].name} = *stack[{i}];' for i in range(len(args))]
        binding_list = []
        for arg in args:
            if not isinstance(arg.argument, Argument):
                raise Exception(f'Unexpected argument type, expecting `Argument` but got {arg}')
            argument: Argument = arg.argument
            (unboxed_name, _, code, decl) = self.argumenttype_evalue_convert(argument.type, argument.name, mutable=argument.is_write)
            code_list.extend(decl)
            code_list.extend(code)
            binding_list.append(arg.with_name(unboxed_name))
        return (binding_list, code_list)

    def argumenttype_evalue_convert(self, t: Type, arg_name: str, *, mutable: bool=False) -> Tuple[str, CType, List[str], List[str]]:
        if False:
            while True:
                i = 10
        '\n        Takes in the type, name and mutability corresponding to an argument, and generates a tuple of:\n        (1) the C++ code necessary to unbox the argument\n        (2) A Binding corresponding to the newly created unboxed variable, including variable name and its CType\n        :param t: a `Type` of an argument\n        :param arg_name: argument name\n        :param mutable: boolean for whether this argument type is mutable\n        :return: unboxed result\n        '
        ctype = self.argument_type_gen(t, mutable=mutable, binds=arg_name).type
        if isinstance(t, BaseType):
            out_name = f'{arg_name}_base'
            (code, decl) = self._gen_code_base_type(arg_name=arg_name, out_name=out_name, ctype=ctype)
        elif isinstance(t, OptionalType):
            out_name = f'{arg_name}_opt_out'
            (code, decl) = self._gen_code_optional_type(arg_name=arg_name, out_name=out_name, t=t, ctype=ctype)
        elif isinstance(t, ListType):
            out_name = f'{arg_name}_list_out'
            (code, decl) = self._gen_code_list_type(arg_name=arg_name, out_name=out_name, t=t, ctype=ctype)
        else:
            raise Exception(f'Cannot handle type {t}. arg_name: {arg_name}')
        return (out_name, ctype, code, decl)

    def _gen_code_base_type(self, arg_name: str, out_name: str, ctype: CType) -> Tuple[List[str], List[str]]:
        if False:
            while True:
                i = 10
        return ([f'{ctype.cpp_type()} {out_name} = {arg_name}.to<{ctype.cpp_type(strip_ref=True)}>();'], [])

    def _gen_code_optional_type(self, arg_name: str, out_name: str, t: OptionalType, ctype: CType) -> Tuple[List[str], List[str]]:
        if False:
            return 10
        in_name = f'{arg_name}_opt_in'
        (res_name, base_type, res_code, decl) = self.argumenttype_evalue_convert(t.elem, in_name)
        return (f'\n    {ctype.cpp_type(strip_ref=True)} {out_name} = {arg_name}.toOptional<{base_type.cpp_type(strip_ref=True)}>();\n            '.split('\n'), decl)

    def _gen_code_list_type(self, arg_name: str, out_name: str, t: ListType, ctype: CType) -> Tuple[List[str], List[str]]:
        if False:
            i = 10
            return i + 15
        in_name = f'{arg_name}_list_in'
        elem_name = f'{arg_name}_elem'
        code = []
        (res_name, res_ctype, res_code, decl) = self.argumenttype_evalue_convert(t.elem, elem_name)
        if isinstance(t.elem, BaseType) and t.elem.name == BaseTy.Tensor:
            code.extend(f'\n    {ctype.cpp_type(strip_ref=True)} {out_name} = {arg_name}.toTensorList();\n                '.split('\n'))
        elif isinstance(t.elem, BaseType) and (t.elem.name == BaseTy.int or t.elem.name == BaseTy.SymInt):
            code.extend(f'\n    {ctype.cpp_type(strip_ref=True)} {out_name} = {arg_name}.toIntList();\n                '.split('\n'))
        elif isinstance(t.elem, BaseType) and t.elem.name == BaseTy.float:
            code.extend(f'\n    {ctype.cpp_type(strip_ref=True)} {out_name} = {arg_name}.toDoubleList();\n                '.split('\n'))
        elif isinstance(t.elem, BaseType) and t.elem.name == BaseTy.bool:
            code.extend(f'\n    {ctype.cpp_type(strip_ref=True)} {out_name} = {arg_name}.toBoolList();\n                '.split('\n'))
        elif isinstance(t.elem, OptionalType) and isinstance(t.elem.elem, BaseType) and (t.elem.elem.name == BaseTy.Tensor):
            code.extend(f'\n#ifdef USE_ATEN_LIB\nat::ArrayRef<c10::optional<at::Tensor>> {in_name} = {arg_name}.toListOptionalTensor();\nc10::List<c10::optional<at::Tensor>> {out_name};\nfor (auto {elem_name}: {in_name}) {{\n    {out_name}.push_back({elem_name});\n}}\n#else\ntorch::executor::ArrayRef<torch::executor::optional<torch::executor::Tensor>> {out_name} = {arg_name}.toListOptionalTensor();\n#endif\n                '.split('\n'))
        else:
            vec_name = arg_name + '_vec'
            decl.append(f'std::vector<{res_ctype.cpp_type(strip_ref=True)}> {vec_name};')
            code.extend(f'\n    for (EValue {elem_name}: {in_name}) {{\n        {connector.join(res_code)}\n        {vec_name}.push_back({res_name});\n    }}\n    {ctype.cpp_type(strip_ref=True)} {out_name}({vec_name});\n                '.split('\n'))
        return (code, decl)