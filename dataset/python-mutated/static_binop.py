from numba.core import errors, ir
from numba.core.rewrites import register_rewrite, Rewrite

@register_rewrite('before-inference')
class DetectStaticBinops(Rewrite):
    """
    Detect constant arguments to select binops.
    """
    rhs_operators = {'**'}

    def match(self, func_ir, block, typemap, calltypes):
        if False:
            for i in range(10):
                print('nop')
        self.static_lhs = {}
        self.static_rhs = {}
        self.block = block
        for expr in block.find_exprs(op='binop'):
            try:
                if expr.fn in self.rhs_operators and expr.static_rhs is ir.UNDEFINED:
                    self.static_rhs[expr] = func_ir.infer_constant(expr.rhs)
            except errors.ConstantInferenceError:
                continue
        return len(self.static_lhs) > 0 or len(self.static_rhs) > 0

    def apply(self):
        if False:
            i = 10
            return i + 15
        '\n        Store constant arguments that were detected in match().\n        '
        for (expr, rhs) in self.static_rhs.items():
            expr.static_rhs = rhs
        return self.block