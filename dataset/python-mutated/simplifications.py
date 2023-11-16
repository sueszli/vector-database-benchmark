import logging
from future.utils import viewitems
from miasm.expression import simplifications_common
from miasm.expression import simplifications_cond
from miasm.expression import simplifications_explicit
from miasm.expression.expression_helper import fast_unify
import miasm.expression.expression as m2_expr
from miasm.expression.expression import ExprVisitorCallbackBottomToTop
log_exprsimp = logging.getLogger('exprsimp')
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('[%(levelname)-8s]: %(message)s'))
log_exprsimp.addHandler(console_handler)
log_exprsimp.setLevel(logging.WARNING)

class ExpressionSimplifier(ExprVisitorCallbackBottomToTop):
    """Wrapper on expression simplification passes.

    Instance handle passes lists.

    Available passes lists are:
     - commons: common passes such as constant folding
     - heavy  : rare passes (for instance, in case of obfuscation)
    """
    PASS_COMMONS = {m2_expr.ExprOp: [simplifications_common.simp_cst_propagation, simplifications_common.simp_cond_op_int, simplifications_common.simp_cond_factor, simplifications_common.simp_add_multiple, simplifications_common.simp_cc_conds, simplifications_common.simp_subwc_cf, simplifications_common.simp_subwc_of, simplifications_common.simp_sign_subwc_cf, simplifications_common.simp_double_zeroext, simplifications_common.simp_double_signext, simplifications_common.simp_zeroext_eq_cst, simplifications_common.simp_ext_eq_ext, simplifications_common.simp_ext_cond_int, simplifications_common.simp_sub_cf_zero, simplifications_common.simp_cmp_int, simplifications_common.simp_cmp_bijective_op, simplifications_common.simp_sign_inf_zeroext, simplifications_common.simp_cmp_int_int, simplifications_common.simp_ext_cst, simplifications_common.simp_zeroext_and_cst_eq_cst, simplifications_common.simp_test_signext_inf, simplifications_common.simp_test_zeroext_inf, simplifications_common.simp_cond_inf_eq_unsigned_zero, simplifications_common.simp_compose_and_mask, simplifications_common.simp_bcdadd_cf, simplifications_common.simp_bcdadd, simplifications_common.simp_smod_sext, simplifications_common.simp_flag_cst], m2_expr.ExprSlice: [simplifications_common.simp_slice, simplifications_common.simp_slice_of_ext, simplifications_common.simp_slice_of_sext, simplifications_common.simp_slice_of_op_ext], m2_expr.ExprCompose: [simplifications_common.simp_compose], m2_expr.ExprCond: [simplifications_common.simp_cond, simplifications_common.simp_cond_zeroext, simplifications_common.simp_cond_add, simplifications_common.simp_cond_flag, simplifications_common.simp_cmp_int_arg, simplifications_common.simp_cond_eq_zero, simplifications_common.simp_x_and_cst_eq_cst, simplifications_common.simp_cond_logic_ext, simplifications_common.simp_cond_sign_bit, simplifications_common.simp_cond_eq_1_0, simplifications_common.simp_cond_cc_flag, simplifications_common.simp_cond_sub_cf], m2_expr.ExprMem: [simplifications_common.simp_mem]}
    PASS_HEAVY = {}
    PASS_COND = {m2_expr.ExprSlice: [simplifications_cond.expr_simp_inf_signed, simplifications_cond.expr_simp_inf_unsigned_inversed], m2_expr.ExprOp: [simplifications_cond.expr_simp_inverse], m2_expr.ExprCond: [simplifications_cond.expr_simp_equal]}
    PASS_HIGH_TO_EXPLICIT = {m2_expr.ExprOp: [simplifications_explicit.simp_flags, simplifications_explicit.simp_ext]}

    def __init__(self):
        if False:
            print('Hello World!')
        super(ExpressionSimplifier, self).__init__(self.expr_simp_inner)
        self.expr_simp_cb = {}

    def enable_passes(self, passes):
        if False:
            return 10
        'Add passes from @passes\n        @passes: dict(Expr class : list(callback))\n\n        Callback signature: Expr callback(ExpressionSimplifier, Expr)\n        '
        self.cache.clear()
        for (k, v) in viewitems(passes):
            self.expr_simp_cb[k] = fast_unify(self.expr_simp_cb.get(k, []) + v)

    def apply_simp(self, expression):
        if False:
            print('Hello World!')
        'Apply enabled simplifications on expression\n        @expression: Expr instance\n        Return an Expr instance'
        cls = expression.__class__
        debug_level = log_exprsimp.level >= logging.DEBUG
        for simp_func in self.expr_simp_cb.get(cls, []):
            before = expression
            expression = simp_func(self, expression)
            after = expression
            if debug_level and before != after:
                log_exprsimp.debug('[%s] %s => %s', simp_func, before, after)
            if expression.__class__ is not cls:
                break
        return expression

    def expr_simp_inner(self, expression):
        if False:
            while True:
                i = 10
        'Apply enabled simplifications on expression and find a stable state\n        @expression: Expr instance\n        Return an Expr instance'
        while True:
            new_expr = self.apply_simp(expression.canonize())
            if new_expr == expression:
                return new_expr
            new_expr = self.visit(new_expr)
            expression = new_expr
        return new_expr

    def expr_simp(self, expression):
        if False:
            for i in range(10):
                print('nop')
        'Call simplification recursively'
        return self.visit(expression)

    def __call__(self, expression):
        if False:
            return 10
        'Call simplification recursively'
        return self.visit(expression)
expr_simp = ExpressionSimplifier()
expr_simp.enable_passes(ExpressionSimplifier.PASS_COMMONS)
expr_simp_high_to_explicit = ExpressionSimplifier()
expr_simp_high_to_explicit.enable_passes(ExpressionSimplifier.PASS_HIGH_TO_EXPLICIT)
expr_simp_explicit = ExpressionSimplifier()
expr_simp_explicit.enable_passes(ExpressionSimplifier.PASS_COMMONS)
expr_simp_explicit.enable_passes(ExpressionSimplifier.PASS_HIGH_TO_EXPLICIT)