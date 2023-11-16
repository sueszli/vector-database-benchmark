from torchgen.api.lazy import LazyArgument, LazyIrSchema
from torchgen.api.types import OptionalCType

def ts_lowering_body(schema: LazyIrSchema) -> str:
    if False:
        for i in range(10):
            print('nop')
    emplace_arguments = []

    def get_value(arg: LazyArgument) -> str:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(arg.lazy_type, OptionalCType):
            return f'has_{arg.name} ? loctx->GetOutputOp(operand(i++)) : nullptr'
        return 'loctx->GetOutputOp(operand(i++))'
    for arg in schema.positional_args:
        if arg.is_lazy_value:
            emplace_arguments.append(get_value(arg))
            continue
        emplace_arguments.append(f'"{arg.name}", {arg.name}')
    emplace_arguments_str = '\n    '.join([f'arguments.emplace_back({a});' for a in emplace_arguments])
    emplace_kwarg_values = [f'"{arg.name}", {get_value(arg)}' for arg in schema.keyword_values]
    emplace_kwarg_scalars = [f'"{arg.name}", {arg.name}' for arg in schema.keyword_scalars]
    emplace_kwarguments = '\n    '.join([f'kwarguments.emplace_back({a});' for a in emplace_kwarg_values + emplace_kwarg_scalars])
    return f'    std::vector<torch::jit::NamedValue> arguments;\n    std::vector<torch::jit::NamedValue> kwarguments;\n    arguments.reserve({len(emplace_arguments)});\n    kwarguments.reserve({len(emplace_kwarg_values + emplace_kwarg_scalars)});\n    size_t i = 0;\n    {emplace_arguments_str}\n    {emplace_kwarguments}\n    torch::lazy::TSOpVector {schema.aten_name}_out = torch::lazy::LowerTSBuiltin(function, op().op, arguments, kwarguments);\n    TORCH_CHECK_EQ({schema.aten_name}_out.size(), {len(schema.returns)});\n\n    return {schema.aten_name}_out;\n'