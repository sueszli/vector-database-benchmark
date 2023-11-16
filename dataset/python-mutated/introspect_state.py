from manticore.native import Manticore
from manticore.core.plugin import StateDescriptor
from manticore.utils.enums import StateStatus
from time import sleep
import typing
import argparse
parser = argparse.ArgumentParser(description='Explore a binary with Manticore and print the tree of states')
parser.add_argument('binary', type=str, nargs='?', default='binaries/multiple-styles', help='The program to run')
args = parser.parse_args()

def print_fork_tree(states: typing.Dict[int, StateDescriptor]):
    if False:
        i = 10
        return i + 15
    '\n    Performs a depth-first traversal of the state tree, where each branch is a different fork\n    '

    def df_print(state_id, depth=0):
        if False:
            i = 10
            return i + 15
        state = states[state_id]
        msg = ''
        if state.status == StateStatus.running:
            msg = '(Exec {} ins)'.format(state.own_execs if state.own_execs is not None else 0)
        elif state.status == StateStatus.waiting_for_solver:
            msg = '(Solving)'
        elif state.status == StateStatus.waiting_for_worker:
            msg = '(Waiting)'
        elif state.status == StateStatus.stopped:
            msg = '({})'.format(state.termination_msg)
        if depth == 0:
            print(state_id, msg)
        else:
            print('     ' * (depth - 1) + '└-->', state_id, msg)
        for c_st in sorted(state.children, key=lambda k: len(states[k].children)):
            df_print(c_st, depth + 1)
    df_print(0)
    print()

def run_every(callee: typing.Callable, duration: int=3) -> typing.Callable:
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns a function that calls <callee> every <duration> seconds\n    '

    def inner(thread):
        if False:
            for i in range(10):
                print('nop')
        while True:
            callee(thread.manticore.introspect())
            sleep(duration)
    return inner
m = Manticore(args.binary)
m.register_daemon(run_every(print_fork_tree, 1))
m.run()
sleep(1)
print('Final fork tree:')
print_fork_tree(m.introspect())