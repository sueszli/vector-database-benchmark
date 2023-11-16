import logging
from future.utils import viewitems
from miasm.ir.symbexec import SymbolicExecutionEngine
from miasm.expression.expression import ExprMem
from miasm.expression.expression_helper import possible_values
from miasm.expression.simplifications import expr_simp
from miasm.ir.ir import IRBlock, AssignBlock
LOG_CST_PROPAG = logging.getLogger('cst_propag')
CONSOLE_HANDLER = logging.StreamHandler()
CONSOLE_HANDLER.setFormatter(logging.Formatter('[%(levelname)-8s]: %(message)s'))
LOG_CST_PROPAG.addHandler(CONSOLE_HANDLER)
LOG_CST_PROPAG.setLevel(logging.WARNING)

class SymbExecState(SymbolicExecutionEngine):
    """
    State manager for SymbolicExecution
    """

    def __init__(self, lifter, ircfg, state):
        if False:
            print('Hello World!')
        super(SymbExecState, self).__init__(lifter, {})
        self.set_state(state)

def add_state(ircfg, todo, states, addr, state):
    if False:
        return 10
    '\n    Add or merge the computed @state for the block at @addr. Update @todo\n    @todo: modified block set\n    @states: dictionary linking a label to its entering state.\n    @addr: address of the considered block\n    @state: computed state\n    '
    addr = ircfg.get_loc_key(addr)
    todo.add(addr)
    if addr not in states:
        states[addr] = state
    else:
        states[addr] = states[addr].merge(state)

def is_expr_cst(lifter, expr):
    if False:
        i = 10
        return i + 15
    'Return true if @expr is only composed of ExprInt and init_regs\n    @lifter: Lifter instance\n    @expr: Expression to test'
    elements = expr.get_r(mem_read=True)
    for element in elements:
        if element.is_mem():
            continue
        if element.is_id() and element in lifter.arch.regs.all_regs_ids_init:
            continue
        if element.is_int():
            continue
        return False
    return True

class SymbExecStateFix(SymbolicExecutionEngine):
    """
    Emul blocks and replace expressions with their corresponding constant if
    any.

    """
    is_expr_cst = lambda _, lifter, expr: is_expr_cst(lifter, expr)

    def __init__(self, lifter, ircfg, state, cst_propag_link):
        if False:
            return 10
        self.ircfg = ircfg
        super(SymbExecStateFix, self).__init__(lifter, {})
        self.set_state(state)
        self.cst_propag_link = cst_propag_link

    def propag_expr_cst(self, expr):
        if False:
            while True:
                i = 10
        'Propagate constant expressions in @expr\n        @expr: Expression to update'
        elements = expr.get_r(mem_read=True)
        to_propag = {}
        for element in elements:
            if not element.is_id():
                continue
            value = self.eval_expr(element)
            if self.is_expr_cst(self.lifter, value):
                to_propag[element] = value
        return expr_simp(expr.replace_expr(to_propag))

    def eval_updt_irblock(self, irb, step=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Symbolic execution of the @irb on the current state\n        @irb: IRBlock instance\n        @step: display intermediate steps\n        '
        assignblks = []
        for (index, assignblk) in enumerate(irb):
            new_assignblk = {}
            links = {}
            for (dst, src) in viewitems(assignblk):
                src = self.propag_expr_cst(src)
                if dst.is_mem():
                    ptr = dst.ptr
                    ptr = self.propag_expr_cst(ptr)
                    dst = ExprMem(ptr, dst.size)
                new_assignblk[dst] = src
            if assignblk.instr is not None:
                for arg in assignblk.instr.args:
                    new_arg = self.propag_expr_cst(arg)
                    links[new_arg] = arg
                self.cst_propag_link[irb.loc_key, index] = links
            self.eval_updt_assignblk(assignblk)
            assignblks.append(AssignBlock(new_assignblk, assignblk.instr))
        self.ircfg.blocks[irb.loc_key] = IRBlock(irb.loc_db, irb.loc_key, assignblks)

def compute_cst_propagation_states(lifter, ircfg, init_addr, init_infos):
    if False:
        for i in range(10):
            print('nop')
    '\n    Propagate "constant expressions" in a function.\n    The attribute "constant expression" is true if the expression is based on\n    constants or "init" regs values.\n\n    @lifter: Lifter instance\n    @init_addr: analysis start address\n    @init_infos: dictionary linking expressions to their values at @init_addr\n    '
    done = set()
    state = SymbExecState.StateEngine(init_infos)
    lbl = ircfg.get_loc_key(init_addr)
    todo = set([lbl])
    states = {lbl: state}
    while todo:
        if not todo:
            break
        lbl = todo.pop()
        state = states[lbl]
        if (lbl, state) in done:
            continue
        done.add((lbl, state))
        if lbl not in ircfg.blocks:
            continue
        symbexec_engine = SymbExecState(lifter, ircfg, state)
        addr = symbexec_engine.run_block_at(ircfg, lbl)
        symbexec_engine.del_mem_above_stack(lifter.sp)
        for dst in possible_values(addr):
            value = dst.value
            if value.is_mem():
                LOG_CST_PROPAG.warning('Bad destination: %s', value)
                continue
            elif value.is_int():
                value = ircfg.get_loc_key(value)
            add_state(ircfg, todo, states, value, symbexec_engine.get_state())
    return states

def propagate_cst_expr(lifter, ircfg, addr, init_infos):
    if False:
        return 10
    '\n    Propagate "constant expressions" in a @lifter.\n    The attribute "constant expression" is true if the expression is based on\n    constants or "init" regs values.\n\n    @lifter: Lifter instance\n    @addr: analysis start address\n    @init_infos: dictionary linking expressions to their values at @init_addr\n\n    Returns a mapping between replaced Expression and their new values.\n    '
    states = compute_cst_propagation_states(lifter, ircfg, addr, init_infos)
    cst_propag_link = {}
    for (lbl, state) in viewitems(states):
        if lbl not in ircfg.blocks:
            continue
        symbexec = SymbExecStateFix(lifter, ircfg, state, cst_propag_link)
        symbexec.eval_updt_irblock(ircfg.blocks[lbl])
    return cst_propag_link