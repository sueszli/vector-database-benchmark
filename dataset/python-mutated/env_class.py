"""Generate classes representing function environments (+ related operations).

If we have a nested function that has non-local (free) variables, access to the
non-locals is via an instance of an environment class. Example:

    def f() -> int:
        x = 0  # Make 'x' an attribute of an environment class instance

        def g() -> int:
            # We have access to the environment class instance to
            # allow accessing 'x'
            return x + 2

        x = x + 1  # Modify the attribute
        return g()
"""
from __future__ import annotations
from mypy.nodes import Argument, FuncDef, SymbolNode, Var
from mypyc.common import BITMAP_BITS, ENV_ATTR_NAME, SELF_NAME, bitmap_name
from mypyc.ir.class_ir import ClassIR
from mypyc.ir.ops import Call, GetAttr, SetAttr, Value
from mypyc.ir.rtypes import RInstance, bitmap_rprimitive, object_rprimitive
from mypyc.irbuild.builder import IRBuilder, SymbolTarget
from mypyc.irbuild.context import FuncInfo, GeneratorClass, ImplicitClass
from mypyc.irbuild.targets import AssignmentTargetAttr

def setup_env_class(builder: IRBuilder) -> ClassIR:
    if False:
        for i in range(10):
            print('nop')
    'Generate a class representing a function environment.\n\n    Note that the variables in the function environment are not\n    actually populated here. This is because when the environment\n    class is generated, the function environment has not yet been\n    visited. This behavior is allowed so that when the compiler visits\n    nested functions, it can use the returned ClassIR instance to\n    figure out free variables it needs to access.  The remaining\n    attributes of the environment class are populated when the\n    environment registers are loaded.\n\n    Return a ClassIR representing an environment for a function\n    containing a nested function.\n    '
    env_class = ClassIR(f'{builder.fn_info.namespaced_name()}_env', builder.module_name, is_generated=True)
    env_class.attributes[SELF_NAME] = RInstance(env_class)
    if builder.fn_info.is_nested:
        env_class.attributes[ENV_ATTR_NAME] = RInstance(builder.fn_infos[-2].env_class)
    env_class.mro = [env_class]
    builder.fn_info.env_class = env_class
    builder.classes.append(env_class)
    return env_class

def finalize_env_class(builder: IRBuilder) -> None:
    if False:
        print('Hello World!')
    'Generate, instantiate, and set up the environment of an environment class.'
    instantiate_env_class(builder)
    if builder.fn_info.is_nested:
        add_args_to_env(builder, local=False, base=builder.fn_info.callable_class)
    else:
        add_args_to_env(builder, local=False, base=builder.fn_info)

def instantiate_env_class(builder: IRBuilder) -> Value:
    if False:
        i = 10
        return i + 15
    'Assign an environment class to a register named after the given function definition.'
    curr_env_reg = builder.add(Call(builder.fn_info.env_class.ctor, [], builder.fn_info.fitem.line))
    if builder.fn_info.is_nested:
        builder.fn_info.callable_class._curr_env_reg = curr_env_reg
        builder.add(SetAttr(curr_env_reg, ENV_ATTR_NAME, builder.fn_info.callable_class.prev_env_reg, builder.fn_info.fitem.line))
    else:
        builder.fn_info._curr_env_reg = curr_env_reg
    return curr_env_reg

def load_env_registers(builder: IRBuilder) -> None:
    if False:
        return 10
    'Load the registers for the current FuncItem being visited.\n\n    Adds the arguments of the FuncItem to the environment. If the\n    FuncItem is nested inside of another function, then this also\n    loads all of the outer environments of the FuncItem into registers\n    so that they can be used when accessing free variables.\n    '
    add_args_to_env(builder, local=True)
    fn_info = builder.fn_info
    fitem = fn_info.fitem
    if fn_info.is_nested:
        load_outer_envs(builder, fn_info.callable_class)
        if isinstance(fitem, FuncDef) and fn_info.add_nested_funcs_to_env:
            setup_func_for_recursive_call(builder, fitem, fn_info.callable_class)

def load_outer_env(builder: IRBuilder, base: Value, outer_env: dict[SymbolNode, SymbolTarget]) -> Value:
    if False:
        i = 10
        return i + 15
    "Load the environment class for a given base into a register.\n\n    Additionally, iterates through all of the SymbolNode and\n    AssignmentTarget instances of the environment at the given index's\n    symtable, and adds those instances to the environment of the\n    current environment. This is done so that the current environment\n    can access outer environment variables without having to reload\n    all of the environment registers.\n\n    Returns the register where the environment class was loaded.\n    "
    env = builder.add(GetAttr(base, ENV_ATTR_NAME, builder.fn_info.fitem.line))
    assert isinstance(env.type, RInstance), f'{env} must be of type RInstance'
    for (symbol, target) in outer_env.items():
        env.type.class_ir.attributes[symbol.name] = target.type
        symbol_target = AssignmentTargetAttr(env, symbol.name)
        builder.add_target(symbol, symbol_target)
    return env

def load_outer_envs(builder: IRBuilder, base: ImplicitClass) -> None:
    if False:
        print('Hello World!')
    index = len(builder.builders) - 2
    if index > 1:
        outer_env = builder.symtables[index]
        if isinstance(base, GeneratorClass):
            base.prev_env_reg = load_outer_env(builder, base.curr_env_reg, outer_env)
        else:
            base.prev_env_reg = load_outer_env(builder, base.self_reg, outer_env)
        env_reg = base.prev_env_reg
        index -= 1
    while index > 1:
        outer_env = builder.symtables[index]
        env_reg = load_outer_env(builder, env_reg, outer_env)
        index -= 1

def num_bitmap_args(builder: IRBuilder, args: list[Argument]) -> int:
    if False:
        while True:
            i = 10
    n = 0
    for arg in args:
        t = builder.type_to_rtype(arg.variable.type)
        if t.error_overlap and arg.kind.is_optional():
            n += 1
    return (n + (BITMAP_BITS - 1)) // BITMAP_BITS

def add_args_to_env(builder: IRBuilder, local: bool=True, base: FuncInfo | ImplicitClass | None=None, reassign: bool=True) -> None:
    if False:
        while True:
            i = 10
    fn_info = builder.fn_info
    args = fn_info.fitem.arguments
    nb = num_bitmap_args(builder, args)
    if local:
        for arg in args:
            rtype = builder.type_to_rtype(arg.variable.type)
            builder.add_local_reg(arg.variable, rtype, is_arg=True)
        for i in reversed(range(nb)):
            builder.add_local_reg(Var(bitmap_name(i)), bitmap_rprimitive, is_arg=True)
    else:
        for arg in args:
            if is_free_variable(builder, arg.variable) or fn_info.is_generator:
                rtype = builder.type_to_rtype(arg.variable.type)
                assert base is not None, 'base cannot be None for adding nonlocal args'
                builder.add_var_to_env_class(arg.variable, rtype, base, reassign=reassign)

def setup_func_for_recursive_call(builder: IRBuilder, fdef: FuncDef, base: ImplicitClass) -> None:
    if False:
        return 10
    'Enable calling a nested function (with a callable class) recursively.\n\n    Adds the instance of the callable class representing the given\n    FuncDef to a register in the environment so that the function can\n    be called recursively. Note that this needs to be done only for\n    nested functions.\n    '
    prev_env = builder.fn_infos[-2].env_class
    prev_env.attributes[fdef.name] = builder.type_to_rtype(fdef.type)
    if isinstance(base, GeneratorClass):
        prev_env_reg = builder.add(GetAttr(base.curr_env_reg, ENV_ATTR_NAME, -1))
    else:
        prev_env_reg = base.prev_env_reg
    val = builder.add(GetAttr(prev_env_reg, fdef.name, -1))
    target = builder.add_local_reg(fdef, object_rprimitive)
    builder.assign(target, val, -1)

def is_free_variable(builder: IRBuilder, symbol: SymbolNode) -> bool:
    if False:
        return 10
    fitem = builder.fn_info.fitem
    return fitem in builder.free_variables and symbol in builder.free_variables[fitem]