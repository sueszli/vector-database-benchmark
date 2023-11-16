"""
Three different ways of evaluating the Collatz conjecture.

Example 1: Pass a symbolic argument to the `collatz` function
Example 2: Pass a symbolic version of `getchar` as an imported function
Example 3: Concretely evaluate `collatz(1337)` and count the instructions executed

See: examples/collatz.c
"""
from manticore.wasm import ManticoreWASM
from manticore.wasm.types import I32
from manticore.core.plugin import Plugin
from manticore.utils.log import set_verbosity
print('\n\n============ Example 1 ============\n')
m = ManticoreWASM('collatz.wasm')
set_verbosity(2)

def arg_gen(state):
    if False:
        print('Hello World!')
    arg = state.new_symbolic_value(32, 'collatz_arg')
    state.constrain(arg > 3)
    state.constrain(arg < 9)
    state.constrain(arg % 2 == 0)
    return [arg]
m.collatz(arg_gen)
for (idx, val_list) in enumerate(m.collect_returns()):
    print('State', idx, '::', val_list[0])
m.finalize()
print('\n\n============ Example 2 ============\n')

def getchar(state, _addr):
    if False:
        while True:
            i = 10
    '\n    Stub implementation of the getchar function. All WASM cares about is that it accepts the right\n    number of arguments and returns the correct type. All _we_ care about is that it returns a symbolic\n    value, for which Manticore will produce all possible outputs.\n\n    :param state: The current state\n    :param _addr: Memory index of the string that gets printed by getchar\n    :return: A symbolic value of the interval [1, 7]\n    '
    res = state.new_symbolic_value(32, 'getchar_res')
    state.constrain(res > 0)
    state.constrain(res < 8)
    return [res]
m = ManticoreWASM('collatz.wasm', env={'getchar': getchar})
m.main()
for (idx, val_list) in enumerate(m.collect_returns()):
    print('State', idx, '::', val_list[0])
m.finalize()
print('\n\n============ Example 3 ============\n')

class CounterPlugin(Plugin):
    """
    A plugin that counts the number of times each instruction occurs
    """

    def did_execute_instruction_callback(self, state, instruction):
        if False:
            for i in range(10):
                print('nop')
        with self.locked_context('counter', dict) as ctx:
            val = ctx.setdefault(instruction.mnemonic, 0)
            ctx[instruction.mnemonic] = val + 1

    def did_terminate_state_callback(self, state, *args):
        if False:
            for i in range(10):
                print('nop')
        insn_sum = 0
        with self.locked_context('counter') as ctx:
            for (mnemonic, count) in sorted(ctx.items(), key=lambda x: x[1], reverse=True):
                print('{: <10} {: >4}'.format(mnemonic, count))
                insn_sum += count
        print(insn_sum, 'instructions executed')
m = ManticoreWASM('collatz.wasm')
m.register_plugin(CounterPlugin())
m.collatz(lambda s: [I32(1337)])
for (idx, val_list) in enumerate(m.collect_returns()):
    print('State', idx, '::', val_list[0])
m.finalize()