"""
LLVM pass that converts intrinsic into other math calls
"""
from llvmlite import ir

class _DivmodFixer(ir.Visitor):

    def visit_Instruction(self, instr):
        if False:
            i = 10
            return i + 15
        if instr.type == ir.IntType(64):
            if instr.opname in ['srem', 'urem', 'sdiv', 'udiv']:
                name = 'numba_{op}'.format(op=instr.opname)
                fn = self.module.globals.get(name)
                if fn is None:
                    opty = instr.type
                    sdivfnty = ir.FunctionType(opty, [opty, opty])
                    fn = ir.Function(self.module, sdivfnty, name=name)
                repl = ir.CallInstr(parent=instr.parent, func=fn, args=instr.operands, name=instr.name)
                instr.parent.replace(instr, repl)

def fix_divmod(mod):
    if False:
        i = 10
        return i + 15
    'Replace division and reminder instructions to builtins calls\n    '
    _DivmodFixer().visit(mod)
INTR_TO_CMATH = {'llvm.pow.f32': 'powf', 'llvm.pow.f64': 'pow', 'llvm.sin.f32': 'sinf', 'llvm.sin.f64': 'sin', 'llvm.cos.f32': 'cosf', 'llvm.cos.f64': 'cos', 'llvm.sqrt.f32': 'sqrtf', 'llvm.sqrt.f64': 'sqrt', 'llvm.exp.f32': 'expf', 'llvm.exp.f64': 'exp', 'llvm.log.f32': 'logf', 'llvm.log.f64': 'log', 'llvm.log10.f32': 'log10f', 'llvm.log10.f64': 'log10', 'llvm.fabs.f32': 'fabsf', 'llvm.fabs.f64': 'fabs', 'llvm.floor.f32': 'floorf', 'llvm.floor.f64': 'floor', 'llvm.ceil.f32': 'ceilf', 'llvm.ceil.f64': 'ceil', 'llvm.trunc.f32': 'truncf', 'llvm.trunc.f64': 'trunc'}
OTHER_CMATHS = '\ntan\ntanf\nsinh\nsinhf\ncosh\ncoshf\ntanh\ntanhf\nasin\nasinf\nacos\nacosf\natan\natanf\natan2\natan2f\nasinh\nasinhf\nacosh\nacoshf\natanh\natanhf\nexpm1\nexpm1f\nlog1p\nlog1pf\nlog10\nlog10f\nfmod\nfmodf\nround\nroundf\n'.split()
INTR_MATH = frozenset(INTR_TO_CMATH.values()) | frozenset(OTHER_CMATHS)