"""
Rough concolic execution implementation

Limitations
- tested only on the simpleassert example program in examples/
- only works for 3 ints of stdin

Bugs
- Will probably break if a newly discovered branch gets more input/does another read(2)
- possibly unnecessary deepcopies

"""
import queue
import struct
import itertools
from manticore import set_verbosity
from manticore.native import Manticore
from manticore.core.plugin import ExtendedTracer, Follower, Plugin
from manticore.core.smtlib.constraints import ConstraintSet
from manticore.core.smtlib.solver import Z3Solver
from manticore.core.smtlib.visitors import GetDeclarations
from manticore.utils import config
import copy
from manticore.core.smtlib.expression import *
from pathlib import Path
prog = str(Path(__file__).parent.resolve().parent.joinpath('linux').joinpath('simpleassert'))
VERBOSITY = 0

def _partition(pred, iterable):
    if False:
        while True:
            i = 10
    (t1, t2) = itertools.tee(iterable)
    return (list(itertools.filterfalse(pred, t1)), list(filter(pred, t2)))

def log(s):
    if False:
        return 10
    print('[+]', s)

class TraceReceiver(Plugin):

    def __init__(self, tracer):
        if False:
            while True:
                i = 10
        self._trace = None
        self._tracer = tracer
        super().__init__()

    @property
    def trace(self):
        if False:
            i = 10
            return i + 15
        return self._trace

    def will_terminate_state_callback(self, state, reason):
        if False:
            return 10
        self._trace = state.context.get(self._tracer.context_key, [])
        (instructions, writes) = _partition(lambda x: x['type'] == 'regs', self._trace)
        total = len(self._trace)
        log(f'Recorded concrete trace: {len(instructions)}/{total} instructions, {len(writes)}/{total} writes')

def flip(constraint):
    if False:
        i = 10
        return i + 15
    '\n    flips a constraint (Equal)\n\n    (Equal (BitVecITE Cond IfC ElseC) IfC)\n        ->\n    (Equal (BitVecITE Cond IfC ElseC) ElseC)\n    '
    equal = copy.copy(constraint)
    assert len(equal.operands) == 2
    (ite, forcepc) = equal.operands
    if not (isinstance(ite, BitVecITE) and isinstance(forcepc, BitVecConstant)):
        return constraint
    assert isinstance(ite, BitVecITE) and isinstance(forcepc, BitVecConstant)
    assert len(ite.operands) == 3
    (cond, iifpc, eelsepc) = ite.operands
    assert isinstance(iifpc, BitVecConstant) and isinstance(eelsepc, BitVecConstant)
    equal._operands = (equal.operands[0], eelsepc if forcepc.value == iifpc.value else iifpc)
    return equal

def eq(a, b):
    if False:
        print('Hello World!')
    (ite1, force1) = a.operands
    (ite2, force2) = b.operands
    if force1.value != force2.value:
        return False
    (_, first1, second1) = ite1.operands
    (_, first2, second2) = ite1.operands
    if first1.value != first2.value:
        return False
    if second1.value != second2.value:
        return False
    return True

def perm(lst, func):
    if False:
        while True:
            i = 10
    "Produce permutations of `lst`, where permutations are mutated by `func`. Used for flipping constraints. highly\n    possible that returned constraints can be unsat this does it blindly, without any attention to the constraints\n    themselves\n\n    Considering lst as a list of constraints, e.g.\n\n      [ C1, C2, C3 ]\n\n    we'd like to consider scenarios of all possible permutations of flipped constraints, excluding the original list.\n    So we'd like to generate:\n\n      [ func(C1),      C2 ,       C3 ],\n      [      C1 , func(C2),       C3 ],\n      [ func(C1), func(C2),       C3 ],\n      [      C1 ,      C2 ,  func(C3)],\n      .. etc\n\n    This is effectively treating the list of constraints as a bitmask of width len(lst) and counting up, skipping the\n    0th element (unmodified array).\n\n    The code below yields lists of constraints permuted as above by treating list indeces as bitmasks from 1 to\n     2**len(lst) and applying func to all the set bit offsets.\n\n    "
    for i in range(1, 2 ** len(lst)):
        yield [func(item) if 1 << j & i else item for (j, item) in enumerate(lst)]

def constraints_to_constraintset(constupl):
    if False:
        while True:
            i = 10
    x = ConstraintSet()
    declarations = GetDeclarations()
    for a in constupl:
        declarations.visit(a)
        x.add(a)
    for d in declarations.result:
        x._declare(d)
    return x

def input_from_cons(constupl, datas):
    if False:
        return 10
    'solve bytes in |datas| based on'

    def make_chr(c):
        if False:
            print('Hello World!')
        try:
            return chr(c)
        except Exception:
            return c
    newset = constraints_to_constraintset(constupl)
    ret = ''
    for data in datas:
        for c in data:
            ret += make_chr(Z3Solver.instance().get_value(newset, c))
    return ret

def concrete_run_get_trace(inp):
    if False:
        for i in range(10):
            print('nop')
    consts = config.get_group('core')
    consts.mprocessing = consts.mprocessing.single
    m1 = Manticore.linux(prog, concrete_start=inp, workspace_url='mem:')
    t = ExtendedTracer()
    set_verbosity(VERBOSITY)
    m1.register_plugin(t)
    m1.run()
    for st in m1.all_states:
        return t.get_trace(st)

def symbolic_run_get_cons(trace):
    if False:
        print('Hello World!')
    '\n    Execute a symbolic run that follows a concrete run; return constraints generated\n    and the stdin data produced\n    '
    m2 = Manticore.linux(prog, workspace_url='mem:')
    f = Follower(trace)
    set_verbosity(VERBOSITY)
    m2.register_plugin(f)

    def on_term_testcase(mm, state, err):
        if False:
            print('Hello World!')
        with m2.locked_context() as ctx:
            readdata = []
            for (name, fd, data) in state.platform.syscall_trace:
                if name in ('_receive', '_read') and fd == 0:
                    readdata.append(data)
            ctx['readdata'] = readdata
            ctx['constraints'] = list(state.constraints.constraints)
    m2.subscribe('will_terminate_state', on_term_testcase)
    m2.run()
    constraints = m2.context['constraints']
    datas = m2.context['readdata']
    return (constraints, datas)

def contains(new, olds):
    if False:
        i = 10
        return i + 15
    '__contains__ operator using the `eq` function'
    return any((eq(new, old) for old in olds))

def getnew(oldcons, newcons):
    if False:
        while True:
            i = 10
    "return all constraints in newcons that aren't in oldcons"
    return [new for new in newcons if not contains(new, oldcons)]

def constraints_are_sat(cons):
    if False:
        for i in range(10):
            print('nop')
    'Whether constraints are sat'
    return Z3Solver.instance().check(constraints_to_constraintset(cons))

def get_new_constrs_for_queue(oldcons, newcons):
    if False:
        while True:
            i = 10
    ret = []
    new_constraints = getnew(oldcons, newcons)
    if not new_constraints:
        return ret
    perms = perm(new_constraints, flip)
    for p in perms:
        candidate = oldcons + p
        if constraints_are_sat(candidate):
            ret.append(candidate)
    return ret

def inp2ints(inp):
    if False:
        i = 10
        return i + 15
    (a, b, c) = struct.unpack('<iii', inp)
    return f'a={a} b={b} c={c}'

def ints2inp(*ints):
    if False:
        print('Hello World!')
    return struct.pack('<' + 'i' * len(ints), *ints)
traces = set()

def concrete_input_to_constraints(ci, prev=None):
    if False:
        i = 10
        return i + 15
    global traces
    if prev is None:
        prev = []
    trc = concrete_run_get_trace(ci)
    trace_rips = tuple((x['values']['RIP'] for x in trc if x['type'] == 'regs' and 'RIP' in x['values']))
    if trace_rips in traces:
        return ([], [])
    traces.add(trace_rips)
    log('getting constraints from symbolic run')
    (cons, datas) = symbolic_run_get_cons(trc)
    new_constraints = get_new_constrs_for_queue(prev, cons)
    log(f'permuting constraints and adding {len(new_constraints)} constraints sets to queue')
    return (new_constraints, datas)

def main():
    if False:
        print('Hello World!')
    q = queue.Queue()
    stdin = ints2inp(0, 5, 0)
    log(f'seed input generated ({inp2ints(stdin)}), running initial concrete run.')
    (to_queue, datas) = concrete_input_to_constraints(stdin)
    for each in to_queue:
        q.put(each)
    while not q.empty():
        log(f'get constraint set from queue, queue size: {q.qsize()}')
        cons = q.get()
        inp = input_from_cons(cons, datas)
        (to_queue, new_datas) = concrete_input_to_constraints(inp, cons)
        if len(new_datas) > 0:
            datas = new_datas
        for each in to_queue:
            q.put(each)
    log(f'paths found: {len(traces)}')
if __name__ == '__main__':
    main()