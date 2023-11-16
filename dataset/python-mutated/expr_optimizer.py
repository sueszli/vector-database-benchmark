"""
Rapids expression optimizer.

:copyright: (c) 2016 H2O.ai
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import h2o.expr

class ExprOptimization(object):
    """
    A generic Rapids expression optimizer

    """

    def __init__(self, supported_ops):
        if False:
            i = 10
            return i + 15
        self._supported_ops = supported_ops

    def supports(self, op):
        if False:
            while True:
                i = 10
        '\n        A quick check if this optimization supports given operator.\n        '
        return op in self._supported_ops

    def is_applicable(self, expr):
        if False:
            for i in range(10):
                print('nop')
        '\n        Is this optimization applicable for given operator\n        This is expensive check and can results in traversal\n        of Rapids expression tree.\n        '
        return False

    def get_optimizer(self, expr):
        if False:
            print('Hello World!')
        '\n        Return a function is transform given expression and context to ExprNode.\n        The function always expects that it is applied in applicable context.\n\n        :param expr:  expression to optimize\n        :return:  a function from context to ExprNode\n        '
        return id(expr)

class FoldExprOptimization(ExprOptimization):
    """
    Fold optimization: support operators which
    accepts array of parameters (e.g., append, cbind):

    For example: append dst (src col_name)+
      (append (append dst srcX col_name_Y) src_A col_name_B) is transformed to
      (append dst src_X col_name_Y src_A col_name_B)

    Objective:
      - the folding save a temporary variable during evaluation
    """

    def __init__(self):
        if False:
            return 10
        super(self.__class__, self).__init__(['append', 'cbind', 'rbind'])

    def is_applicable(self, expr):
        if False:
            return 10
        assert isinstance(expr, h2o.expr.ExprNode)
        return any(expr._children) and expr._children[0]._op == expr._op

    def get_optimizer(self, expr):
        if False:
            while True:
                i = 10

        def foptimizer(ctx):
            if False:
                return 10
            nested_expr = expr.arg(0)
            expr._children = nested_expr._children + expr._children[1:]
            return expr
        return foptimizer

class SkipExprOptimization(ExprOptimization):
    """
    The skip optimization removes unnecessary nodes
    from expression tree.

    For example:
      The expression `(col_py (append frame_with_100_columns dummy_col dummy_name) 1)` can
      be simplified to `(col_py frame_with_100_columns 1)`

    Note: right now this is really specific version only
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(self.__class__, self).__init__(['cols_py'])

    def is_applicable(self, expr):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(expr, h2o.expr.ExprNode)
        if any(expr._children):
            append_expr = expr.arg(0)
            if expr.narg() == 2 and append_expr._op == 'append' and any(append_expr._children):
                append_dst = append_expr.arg(0)
                cols_py_select = expr.arg(1)
                return isinstance(cols_py_select, int) and append_dst._cache.ncols_valid() and (cols_py_select < append_dst._cache.ncols)
        return False

    def get_optimizer(self, expr):
        if False:
            while True:
                i = 10

        def foptimizer(ctx):
            if False:
                i = 10
                return i + 15
            append_expr = expr.arg(0)
            append_dst = append_expr.arg(0)
            expr._children = tuple([append_dst]) + expr._children[1:]
            return expr
        return foptimizer

def optimize(expr):
    if False:
        while True:
            i = 10
    assert isinstance(expr, h2o.expr.ExprNode)
    all_optimizers = get_optimization(expr._op)
    applicable_optimizers = [f for f in all_optimizers if f.is_applicable(expr)]
    if applicable_optimizers:
        return applicable_optimizers[0].get_optimizer(expr)
    else:
        return None

def get_optimization(op):
    if False:
        i = 10
        return i + 15
    return [f for f in __REGISTERED_EXPR_OPTIMIZATIONS__ if f.supports(op)]

def id(expr):
    if False:
        i = 10
        return i + 15
    '\n    This is identity optimization.\n    :param expr:  expression to optimize\n    :return:  a function which always returns expr\n    '

    def identity(ctx):
        if False:
            print('Hello World!')
        return expr
    return identity
__REGISTERED_EXPR_OPTIMIZATIONS__ = [FoldExprOptimization(), SkipExprOptimization()]