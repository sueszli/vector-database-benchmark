"""
Apply simplification passes to an IR cfg
"""
import logging
import warnings
from functools import wraps
from miasm.analysis.ssa import SSADiGraph
from miasm.analysis.outofssa import UnSSADiGraph
from miasm.analysis.data_flow import DiGraphLivenessSSA
from miasm.expression.simplifications import expr_simp
from miasm.ir.ir import AssignBlock, IRBlock
from miasm.analysis.data_flow import DeadRemoval, merge_blocks, remove_empty_assignblks, del_unused_edges, PropagateExpressions, DelDummyPhi
log = logging.getLogger('simplifier')
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('[%(levelname)-8s]: %(message)s'))
log.addHandler(console_handler)
log.setLevel(logging.WARNING)

def fix_point(func):
    if False:
        print('Hello World!')

    @wraps(func)
    def ret_func(self, ircfg, head):
        if False:
            while True:
                i = 10
        log.debug('[%s]: start', func.__name__)
        has_been_modified = False
        modified = True
        while modified:
            modified = func(self, ircfg, head)
            has_been_modified |= modified
        log.debug('[%s]: stop %r', func.__name__, has_been_modified)
        return has_been_modified
    return ret_func

class IRCFGSimplifier(object):
    """
    Simplify an IRCFG
    This class applies passes until reaching a fix point
    """

    def __init__(self, lifter):
        if False:
            return 10
        self.lifter = lifter
        self.init_passes()

    @property
    def ir_arch(self):
        if False:
            i = 10
            return i + 15
        warnings.warn('DEPRECATION WARNING: use ".lifter" instead of ".ir_arch"')
        return self.lifter

    def init_passes(self):
        if False:
            print('Hello World!')
        '\n        Init the array of simplification passes\n        '
        self.passes = []

    @fix_point
    def simplify(self, ircfg, head):
        if False:
            while True:
                i = 10
        '\n        Apply passes until reaching a fix point\n        Return True if the graph has been modified\n\n        @ircfg: IRCFG instance to simplify\n        @head: Location instance of the ircfg head\n        '
        modified = False
        for simplify_pass in self.passes:
            modified |= simplify_pass(ircfg, head)
        return modified

    def __call__(self, ircfg, head):
        if False:
            while True:
                i = 10
        return self.simplify(ircfg, head)

class IRCFGSimplifierCommon(IRCFGSimplifier):
    """
    Simplify an IRCFG
    This class applies following passes until reaching a fix point:
    - simplify_ircfg
    - do_dead_simp_ircfg
    """

    def __init__(self, lifter, expr_simp=expr_simp):
        if False:
            i = 10
            return i + 15
        self.expr_simp = expr_simp
        super(IRCFGSimplifierCommon, self).__init__(lifter)
        self.deadremoval = DeadRemoval(self.lifter)

    def init_passes(self):
        if False:
            while True:
                i = 10
        self.passes = [self.simplify_ircfg, self.do_dead_simp_ircfg]

    @fix_point
    def simplify_ircfg(self, ircfg, _head):
        if False:
            while True:
                i = 10
        '\n        Apply self.expr_simp on the @ircfg until reaching fix point\n        Return True if the graph has been modified\n\n        @ircfg: IRCFG instance to simplify\n        '
        modified = ircfg.simplify(self.expr_simp)
        return modified

    @fix_point
    def do_dead_simp_ircfg(self, ircfg, head):
        if False:
            for i in range(10):
                print('nop')
        '\n        Apply:\n        - dead_simp\n        - remove_empty_assignblks\n        - merge_blocks\n        on the @ircfg until reaching fix point\n        Return True if the graph has been modified\n\n        @ircfg: IRCFG instance to simplify\n        @head: Location instance of the ircfg head\n        '
        modified = self.deadremoval(ircfg)
        modified |= remove_empty_assignblks(ircfg)
        modified |= merge_blocks(ircfg, set([head]))
        return modified

class IRCFGSimplifierSSA(IRCFGSimplifierCommon):
    """
    Simplify an IRCFG.
    The IRCF is first transformed in SSA, then apply transformations passes
    and apply out-of-ssa. Final passes of IRcfgSimplifier are applied

    This class apply following pass until reaching a fix point:
    - do_propagate_expressions
    - do_dead_simp_ssa
    """

    def __init__(self, lifter, expr_simp=expr_simp):
        if False:
            while True:
                i = 10
        super(IRCFGSimplifierSSA, self).__init__(lifter, expr_simp)
        self.lifter.ssa_var = {}
        self.all_ssa_vars = {}
        self.ssa_forbidden_regs = self.get_forbidden_regs()
        self.propag_expressions = PropagateExpressions()
        self.del_dummy_phi = DelDummyPhi()
        self.deadremoval = DeadRemoval(self.lifter, self.all_ssa_vars)

    def get_forbidden_regs(self):
        if False:
            return 10
        '\n        Return a set of immutable register during SSA transformation\n        '
        regs = set([self.lifter.pc, self.lifter.IRDst, self.lifter.arch.regs.exception_flags])
        return regs

    def init_passes(self):
        if False:
            return 10
        '\n        Init the array of simplification passes\n        '
        self.passes = [self.simplify_ssa, self.do_propagate_expressions, self.do_del_dummy_phi, self.do_dead_simp_ssa, self.do_remove_empty_assignblks, self.do_del_unused_edges, self.do_merge_blocks]

    def ircfg_to_ssa(self, ircfg, head):
        if False:
            return 10
        "\n        Apply the SSA transformation to @ircfg using it's @head\n\n        @ircfg: IRCFG instance to simplify\n        @head: Location instance of the ircfg head\n        "
        ssa = SSADiGraph(ircfg)
        ssa.immutable_ids.update(self.ssa_forbidden_regs)
        ssa.ssa_variable_to_expr.update(self.all_ssa_vars)
        ssa.transform(head)
        self.all_ssa_vars.update(ssa.ssa_variable_to_expr)
        self.lifter.ssa_var.update(ssa.ssa_variable_to_expr)
        return ssa

    def ssa_to_unssa(self, ssa, head):
        if False:
            return 10
        "\n        Apply the out-of-ssa transformation to @ssa using it's @head\n\n        @ssa: SSADiGraph instance\n        @head: Location instance of the graph head\n        "
        cfg_liveness = DiGraphLivenessSSA(ssa.graph)
        cfg_liveness.init_var_info(self.lifter)
        cfg_liveness.compute_liveness()
        UnSSADiGraph(ssa, head, cfg_liveness)
        return ssa.graph

    @fix_point
    def simplify_ssa(self, ssa, _head):
        if False:
            while True:
                i = 10
        '\n        Apply self.expr_simp on the @ssa.graph until reaching fix point\n        Return True if the graph has been modified\n\n        @ssa: SSADiGraph instance\n        '
        modified = ssa.graph.simplify(self.expr_simp)
        return modified

    @fix_point
    def do_del_unused_edges(self, ssa, head):
        if False:
            for i in range(10):
                print('nop')
        '\n        Del unused edges of the ssa graph\n        @head: Location instance of the graph head\n        '
        modified = del_unused_edges(ssa.graph, set([head]))
        return modified

    def do_propagate_expressions(self, ssa, head):
        if False:
            return 10
        '\n        Expressions propagation through ExprId in the @ssa graph\n        @head: Location instance of the graph head\n        '
        modified = self.propag_expressions.propagate(ssa, head)
        return modified

    @fix_point
    def do_del_dummy_phi(self, ssa, head):
        if False:
            print('Hello World!')
        '\n        Del dummy phi\n        @head: Location instance of the graph head\n        '
        modified = self.del_dummy_phi.del_dummy_phi(ssa, head)
        return modified

    @fix_point
    def do_remove_empty_assignblks(self, ssa, head):
        if False:
            while True:
                i = 10
        '\n        Remove empty assignblks\n        @head: Location instance of the graph head\n        '
        modified = remove_empty_assignblks(ssa.graph)
        return modified

    @fix_point
    def do_merge_blocks(self, ssa, head):
        if False:
            for i in range(10):
                print('nop')
        '\n        Merge blocks with one parent/son\n        @head: Location instance of the graph head\n        '
        modified = merge_blocks(ssa.graph, set([head]))
        return modified

    @fix_point
    def do_dead_simp_ssa(self, ssa, head):
        if False:
            print('Hello World!')
        '\n        Apply:\n        - deadrm\n        - remove_empty_assignblks\n        - del_unused_edges\n        - merge_blocks\n        on the @ircfg until reaching fix point\n        Return True if the graph has been modified\n\n        @ircfg: IRCFG instance to simplify\n        @head: Location instance of the ircfg head\n        '
        modified = self.deadremoval(ssa.graph)
        return modified

    def do_simplify(self, ssa, head):
        if False:
            return 10
        '\n        Apply passes until reaching a fix point\n        Return True if the graph has been modified\n        '
        return super(IRCFGSimplifierSSA, self).simplify(ssa, head)

    def do_simplify_loop(self, ssa, head):
        if False:
            i = 10
            return i + 15
        '\n        Apply do_simplify until reaching a fix point\n        SSA is updated between each do_simplify\n        Return True if the graph has been modified\n        '
        modified = True
        while modified:
            modified = self.do_simplify(ssa, head)
            ssa = self.ircfg_to_ssa(ssa.graph, head)
        return ssa

    def simplify(self, ircfg, head):
        if False:
            print('Hello World!')
        '\n        Add access to "abi out regs" in each leaf block\n        Apply SSA transformation to @ircfg\n        Apply passes until reaching a fix point\n        Apply out-of-ssa transformation\n        Apply post simplification passes\n\n        Updated simplified IRCFG instance and return it\n\n        @ircfg: IRCFG instance to simplify\n        @head: Location instance of the ircfg head\n        '
        ssa = self.ircfg_to_ssa(ircfg, head)
        ssa = self.do_simplify_loop(ssa, head)
        ircfg = self.ssa_to_unssa(ssa, head)
        ircfg_simplifier = IRCFGSimplifierCommon(self.lifter)
        ircfg_simplifier.deadremoval.add_expr_to_original_expr(self.all_ssa_vars)
        ircfg_simplifier.simplify(ircfg, head)
        return ircfg