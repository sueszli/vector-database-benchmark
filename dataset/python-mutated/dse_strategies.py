"""Example of DynamicSymbolicExecution engine use

This example highlights how coverage can be obtained on a binary

Expected target: 'simple_test.bin'

Global overview:
 - Prepare a 'jitter' instance with the targeted function
 - Attach a DSE instance
 - Make the function argument symbolic, to track constraints on it
 - Take a snapshot
 - Initialize the argument candidate list with an arbitrary value, 0
 - Main loop:
   - Restore the snapshot (initial state, before running the function)
   - Take an argument candidate and set it in the jitter
   - Run the function
   - Ask the DSE for new candidates, according to its strategy, ie. finding new
block / branch / path
"""
from __future__ import print_function
from argparse import ArgumentParser
from future.utils import viewitems
from miasm.analysis.machine import Machine
from miasm.jitter.csts import PAGE_READ, PAGE_WRITE
from miasm.analysis.dse import DSEPathConstraint
from miasm.expression.expression import ExprMem, ExprId, ExprInt, ExprAssign
from miasm.core.locationdb import LocationDB
parser = ArgumentParser('DSE Example')
parser.add_argument('filename', help='Target x86 shellcode')
parser.add_argument('strategy', choices=['code-cov', 'branch-cov', 'path-cov'], help='Strategy to use for solution creation')
args = parser.parse_args()
strategy = {'code-cov': DSEPathConstraint.PRODUCE_SOLUTION_CODE_COV, 'branch-cov': DSEPathConstraint.PRODUCE_SOLUTION_BRANCH_COV, 'path-cov': DSEPathConstraint.PRODUCE_SOLUTION_PATH_COV}[args.strategy]
loc_db = LocationDB()
run_addr = 262144
machine = Machine('x86_32')
jitter = machine.jitter(loc_db, 'python')
with open(args.filename, 'rb') as fdesc:
    jitter.vm.add_memory_page(run_addr, PAGE_READ | PAGE_WRITE, fdesc.read(), 'Binary')
jitter.init_stack()
jitter.push_uint32_t(0)

def code_sentinelle(jitter):
    if False:
        i = 10
        return i + 15
    jitter.running = False
    return False
ret_addr = 322420463
jitter.add_breakpoint(ret_addr, code_sentinelle)
jitter.push_uint32_t(ret_addr)
jitter.init_run(run_addr)
dse = DSEPathConstraint(machine, loc_db, produce_solution=strategy)
dse.attach(jitter)
dse.update_state_from_concrete()
regs = jitter.lifter.arch.regs
arg = ExprId('ARG', 32)
arg_addr = ExprMem(ExprInt(jitter.cpu.ESP + 4, regs.ESP.size), arg.size)
dse.update_state({arg_addr: arg})
todo = set([ExprInt(0, arg.size)])
done = set()
snapshot = dse.take_snapshot()
reaches = set()
while todo:
    arg_value = todo.pop()
    if arg_value in done:
        continue
    done.add(arg_value)
    print('Run with ARG = %s' % arg_value)
    dse.restore_snapshot(snapshot, keep_known_solutions=True)
    jitter.init_run(run_addr)
    jitter.eval_expr(ExprAssign(arg_addr, arg_value))
    jitter.continue_run()
    for (sol_ident, model) in viewitems(dse.new_solutions):
        print('Found a solution to reach: %s' % str(sol_ident))
        sol_value = model.eval(dse.z3_trans.from_expr(arg)).as_long()
        sol_expr = ExprInt(sol_value, arg.size)
        print('\tARG = %s' % sol_expr)
        todo.add(sol_expr)
        reaches.add(sol_ident)
print('Found %d input, to reach %d element of coverage' % (len(done), len(reaches)))