from numba.core import errors, ir
from numba.core.rewrites import register_rewrite, Rewrite

@register_rewrite('before-inference')
class RewritePrintCalls(Rewrite):
    """
    Rewrite calls to the print() global function to dedicated IR print() nodes.
    """

    def match(self, func_ir, block, typemap, calltypes):
        if False:
            i = 10
            return i + 15
        self.prints = prints = {}
        self.block = block
        for inst in block.find_insts(ir.Assign):
            if isinstance(inst.value, ir.Expr) and inst.value.op == 'call':
                expr = inst.value
                try:
                    callee = func_ir.infer_constant(expr.func)
                except errors.ConstantInferenceError:
                    continue
                if callee is print:
                    if expr.kws:
                        msg = "Numba's print() function implementation does not support keyword arguments."
                        raise errors.UnsupportedError(msg, inst.loc)
                    prints[inst] = expr
        return len(prints) > 0

    def apply(self):
        if False:
            while True:
                i = 10
        '\n        Rewrite `var = call <print function>(...)` as a sequence of\n        `print(...)` and `var = const(None)`.\n        '
        new_block = self.block.copy()
        new_block.clear()
        for inst in self.block.body:
            if inst in self.prints:
                expr = self.prints[inst]
                print_node = ir.Print(args=expr.args, vararg=expr.vararg, loc=expr.loc)
                new_block.append(print_node)
                assign_node = ir.Assign(value=ir.Const(None, loc=expr.loc), target=inst.target, loc=inst.loc)
                new_block.append(assign_node)
            else:
                new_block.append(inst)
        return new_block

@register_rewrite('before-inference')
class DetectConstPrintArguments(Rewrite):
    """
    Detect and store constant arguments to print() nodes.
    """

    def match(self, func_ir, block, typemap, calltypes):
        if False:
            while True:
                i = 10
        self.consts = consts = {}
        self.block = block
        for inst in block.find_insts(ir.Print):
            if inst.consts:
                continue
            for (idx, var) in enumerate(inst.args):
                try:
                    const = func_ir.infer_constant(var)
                except errors.ConstantInferenceError:
                    continue
                consts.setdefault(inst, {})[idx] = const
        return len(consts) > 0

    def apply(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Store detected constant arguments on their nodes.\n        '
        for inst in self.block.body:
            if inst in self.consts:
                inst.consts = self.consts[inst]
        return self.block