"""Generate a class that represents a nested function.

The class defines __call__ for calling the function and allows access to
non-local variables defined in outer scopes.
"""
from __future__ import annotations
from mypyc.common import ENV_ATTR_NAME, SELF_NAME
from mypyc.ir.class_ir import ClassIR
from mypyc.ir.func_ir import FuncDecl, FuncIR, FuncSignature, RuntimeArg
from mypyc.ir.ops import BasicBlock, Call, Register, Return, SetAttr, Value
from mypyc.ir.rtypes import RInstance, object_rprimitive
from mypyc.irbuild.builder import IRBuilder
from mypyc.irbuild.context import FuncInfo, ImplicitClass
from mypyc.primitives.misc_ops import method_new_op

def setup_callable_class(builder: IRBuilder) -> None:
    if False:
        while True:
            i = 10
    "Generate an (incomplete) callable class representing a function.\n\n    This can be a nested function or a function within a non-extension\n    class.  Also set up the 'self' variable for that class.\n\n    This takes the most recently visited function and returns a\n    ClassIR to represent that function. Each callable class contains\n    an environment attribute which points to another ClassIR\n    representing the environment class where some of its variables can\n    be accessed.\n\n    Note that some methods, such as '__call__', are not yet\n    created here. Use additional functions, such as\n    add_call_to_callable_class(), to add them.\n\n    Return a newly constructed ClassIR representing the callable\n    class for the nested function.\n    "
    name = base_name = f'{builder.fn_info.namespaced_name()}_obj'
    count = 0
    while name in builder.callable_class_names:
        name = base_name + '_' + str(count)
        count += 1
    builder.callable_class_names.add(name)
    callable_class_ir = ClassIR(name, builder.module_name, is_generated=True)
    if builder.fn_info.is_nested:
        callable_class_ir.has_dict = True
    if builder.fn_infos[-2].contains_nested:
        callable_class_ir.attributes[ENV_ATTR_NAME] = RInstance(builder.fn_infos[-2].env_class)
    callable_class_ir.mro = [callable_class_ir]
    builder.fn_info.callable_class = ImplicitClass(callable_class_ir)
    builder.classes.append(callable_class_ir)
    self_target = builder.add_self_to_env(callable_class_ir)
    builder.fn_info.callable_class.self_reg = builder.read(self_target, builder.fn_info.fitem.line)

def add_call_to_callable_class(builder: IRBuilder, args: list[Register], blocks: list[BasicBlock], sig: FuncSignature, fn_info: FuncInfo) -> FuncIR:
    if False:
        print('Hello World!')
    "Generate a '__call__' method for a callable class representing a nested function.\n\n    This takes the blocks and signature associated with a function\n    definition and uses those to build the '__call__' method of a\n    given callable class, used to represent that function.\n    "
    nargs = len(sig.args) - sig.num_bitmap_args
    sig = FuncSignature((RuntimeArg(SELF_NAME, object_rprimitive),) + sig.args[:nargs], sig.ret_type)
    call_fn_decl = FuncDecl('__call__', fn_info.callable_class.ir.name, builder.module_name, sig)
    call_fn_ir = FuncIR(call_fn_decl, args, blocks, fn_info.fitem.line, traceback_name=fn_info.fitem.name)
    fn_info.callable_class.ir.methods['__call__'] = call_fn_ir
    fn_info.callable_class.ir.method_decls['__call__'] = call_fn_decl
    return call_fn_ir

def add_get_to_callable_class(builder: IRBuilder, fn_info: FuncInfo) -> None:
    if False:
        i = 10
        return i + 15
    "Generate the '__get__' method for a callable class."
    line = fn_info.fitem.line
    with builder.enter_method(fn_info.callable_class.ir, '__get__', object_rprimitive, fn_info, self_type=object_rprimitive):
        instance = builder.add_argument('instance', object_rprimitive)
        builder.add_argument('owner', object_rprimitive)
        (instance_block, class_block) = (BasicBlock(), BasicBlock())
        comparison = builder.translate_is_op(builder.read(instance), builder.none_object(), 'is', line)
        builder.add_bool_branch(comparison, class_block, instance_block)
        builder.activate_block(class_block)
        builder.add(Return(builder.self()))
        builder.activate_block(instance_block)
        builder.add(Return(builder.call_c(method_new_op, [builder.self(), builder.read(instance)], line)))

def instantiate_callable_class(builder: IRBuilder, fn_info: FuncInfo) -> Value:
    if False:
        return 10
    'Create an instance of a callable class for a function.\n\n    Calls to the function will actually call this instance.\n\n    Note that fn_info refers to the function being assigned, whereas\n    builder.fn_info refers to the function encapsulating the function\n    being turned into a callable class.\n    '
    fitem = fn_info.fitem
    func_reg = builder.add(Call(fn_info.callable_class.ir.ctor, [], fitem.line))
    curr_env_reg = None
    if builder.fn_info.is_generator:
        curr_env_reg = builder.fn_info.generator_class.curr_env_reg
    elif builder.fn_info.is_nested:
        curr_env_reg = builder.fn_info.callable_class.curr_env_reg
    elif builder.fn_info.contains_nested:
        curr_env_reg = builder.fn_info.curr_env_reg
    if curr_env_reg:
        builder.add(SetAttr(func_reg, ENV_ATTR_NAME, curr_env_reg, fitem.line))
    return func_reg