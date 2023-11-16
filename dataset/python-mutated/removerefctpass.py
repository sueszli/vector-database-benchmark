"""
Implement a rewrite pass on a LLVM module to remove unnecessary 
refcount operations.
"""
from llvmlite.ir.transforms import CallVisitor
from numba.core import types

class _MarkNrtCallVisitor(CallVisitor):
    """
    A pass to mark all NRT_incref and NRT_decref.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self.marked = set()

    def visit_Call(self, instr):
        if False:
            print('Hello World!')
        if getattr(instr.callee, 'name', '') in _accepted_nrtfns:
            self.marked.add(instr)

def _rewrite_function(function):
    if False:
        while True:
            i = 10
    markpass = _MarkNrtCallVisitor()
    markpass.visit_Function(function)
    for bb in function.basic_blocks:
        for inst in list(bb.instructions):
            if inst in markpass.marked:
                bb.instructions.remove(inst)
_accepted_nrtfns = ('NRT_incref', 'NRT_decref')

def _legalize(module, dmm, fndesc):
    if False:
        return 10
    '\n    Legalize the code in the module.\n    Returns True if the module is legal for the rewrite pass that removes\n    unnecessary refcounts.\n    '

    def valid_output(ty):
        if False:
            return 10
        '\n        Valid output are any type that does not need refcount\n        '
        model = dmm[ty]
        return not model.contains_nrt_meminfo()

    def valid_input(ty):
        if False:
            print('Hello World!')
        '\n        Valid input are any type that does not need refcount except Array.\n        '
        return valid_output(ty) or isinstance(ty, types.Array)
    try:
        nmd = module.get_named_metadata('numba_args_may_always_need_nrt')
    except KeyError:
        pass
    else:
        if len(nmd.operands) > 0:
            return False
    argtypes = fndesc.argtypes
    restype = fndesc.restype
    calltypes = fndesc.calltypes
    for argty in argtypes:
        if not valid_input(argty):
            return False
    if not valid_output(restype):
        return False
    for callty in calltypes.values():
        if callty is not None and (not valid_output(callty.return_type)):
            return False
    for fn in module.functions:
        if fn.name.startswith('NRT_'):
            if fn.name not in _accepted_nrtfns:
                return False
    return True

def remove_unnecessary_nrt_usage(function, context, fndesc):
    if False:
        i = 10
        return i + 15
    '\n    Remove unnecessary NRT incref/decref in the given LLVM function.\n    It uses highlevel type info to determine if the function does not need NRT.\n    Such a function does not:\n\n    - return array object(s);\n    - take arguments that need refcounting except array;\n    - call function(s) that return refcounted object.\n\n    In effect, the function will not capture or create references that extend\n    the lifetime of any refcounted objects beyond the lifetime of the function.\n\n    The rewrite is performed in place.\n    If rewrite has happened, this function returns True, otherwise, it returns False.\n    '
    dmm = context.data_model_manager
    if _legalize(function.module, dmm, fndesc):
        _rewrite_function(function)
        return True
    else:
        return False