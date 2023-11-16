from llvmlite import ir
from llvmlite.ir.transforms import Visitor, CallVisitor

class FastFloatBinOpVisitor(Visitor):
    """
    A pass to add fastmath flag to float-binop instruction if they don't have
    any flags.
    """
    float_binops = frozenset(['fadd', 'fsub', 'fmul', 'fdiv', 'frem', 'fcmp'])

    def __init__(self, flags):
        if False:
            print('Hello World!')
        self.flags = flags

    def visit_Instruction(self, instr):
        if False:
            while True:
                i = 10
        if instr.opname in self.float_binops:
            if not instr.flags:
                for flag in self.flags:
                    instr.flags.append(flag)

class FastFloatCallVisitor(CallVisitor):
    """
    A pass to change all float function calls to use fastmath.
    """

    def __init__(self, flags):
        if False:
            print('Hello World!')
        self.flags = flags

    def visit_Call(self, instr):
        if False:
            print('Hello World!')
        if instr.type in (ir.FloatType(), ir.DoubleType()):
            for flag in self.flags:
                instr.fastmath.add(flag)

def rewrite_module(mod, options):
    if False:
        i = 10
        return i + 15
    '\n    Rewrite the given LLVM module to use fastmath everywhere.\n    '
    flags = options.flags
    FastFloatBinOpVisitor(flags).visit(mod)
    FastFloatCallVisitor(flags).visit(mod)