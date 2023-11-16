"""
sandshrew.py

    Unconstrained concolic execution tool for cryptographic verification
    utilizing Manticore as a backend for symbolic execution and Unicorn
    for concrete instruction emulation.

"""
import os
import string
import random
import logging
import argparse
from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection
from manticore import issymbolic
from manticore.core.smtlib import operators
from manticore.native import Manticore
from manticore.native.models import strcmp
from manticore.utils.fallback_emulator import UnicornEmulator
BUFFER_SIZE = 32
PREPEND_SYM = 'SANDSHREW_'

def binary_arch(binary):
    if False:
        print('Hello World!')
    '\n    helper method for determining binary architecture\n\n    :param binary: str for binary to introspect.\n    :rtype bool: True for x86_64, False otherwise\n    '
    with open(binary, 'rb') as f:
        elffile = ELFFile(f)
        if elffile['e_machine'] == 'EM_X86_64':
            return True
        else:
            return False

def binary_symbols(binary):
    if False:
        return 10
    '\n    helper method for getting all binary symbols with SANDSHREW_ prepended.\n    We do this in order to provide the symbols Manticore should hook on to\n    perform main analysis.\n\n    :param binary: str for binary to instrospect.\n    :rtype list: list of symbols from binary\n    '

    def substr_after(string, delim):
        if False:
            while True:
                i = 10
        return string.partition(delim)[2]
    with open(binary, 'rb') as f:
        elffile = ELFFile(f)
        for section in elffile.iter_sections():
            if not isinstance(section, SymbolTableSection):
                continue
            symbols = [sym.name for sym in section.iter_symbols() if sym]
            return [substr_after(name, PREPEND_SYM) for name in symbols if name.startswith(PREPEND_SYM)]

def main():
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser(prog='sandshrew')
    required = parser.add_argument_group('required arguments')
    required.add_argument('-t', '--test', dest='test', required=True, help='Target binary for sandshrew analysis')
    parser.add_argument('-c', '--constraint', dest='constraint', required=False, help='Constraint to apply to symbolic input. Includes ascii, alpha, num, or alphanum')
    parser.add_argument('--debug', dest='debug', action='store_true', required=False, help='If set, turns on debugging output for sandshrew')
    parser.add_argument('--trace', dest='trace', action='store_true', required=False, help='If set, trace instruction recording will be outputted to logger')
    parser.add_argument('--cmpsym', dest='cmp_sym', default='__strcmp_ssse3', required=False, help='Overrides comparison function used to test for equivalence (default is strcmp)')
    args = parser.parse_args()
    if args is None:
        parser.print_help()
        return 0
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    if not binary_arch(args.test):
        raise NotImplementedError('sandshrew only supports x86_64 binary concretization')
    m = Manticore.linux(args.test, ['+' * BUFFER_SIZE])
    m.verbosity(2)
    m.context['syms'] = binary_symbols(args.test)
    m.context['exec_flag'] = False
    m.context['argv1'] = None
    logging.debug(f"Functions for concretization: {m.context['syms']}")
    m.context['trace'] = []

    @m.init
    def init(state):
        if False:
            for i in range(10):
                print('nop')
        logging.debug(f'Checking for symbolic ARGV')
        argv1 = next((sym for sym in state.input_symbols if sym.name == 'ARGV1'))
        if argv1 is None:
            raise RuntimeException('ARGV was not provided and/or made symbolic')
        with m.locked_context() as context:
            context['argv1'] = argv1

    @m.hook(None)
    def record(state):
        if False:
            return 10
        pc = state.cpu.PC
        if args.trace:
            print(f'{hex(pc)}')
        with m.locked_context() as context:
            context['trace'] += [pc]
    for sym in m.context['syms']:

        @m.hook(m.resolve('SANDSHREW_' + sym))
        def concrete_checker(state):
            if False:
                while True:
                    i = 10
            '\n            initial checker hook for SANDSHREW_sym that checks for the presence of symbolic input.\n            If so, an unconstrained hook is attached to the memory location to restore symbolic state after concretization\n            '
            cpu = state.cpu
            with m.locked_context() as context:
                logging.debug(f'Entering target function SANDSHREW_{sym} at {hex(state.cpu.PC)}')
                data = cpu.read_int(cpu.RSI)
                if issymbolic(data):
                    logging.debug(f'Symbolic input parameter to function {sym}() detected')
                    return_pc = context['trace'][-1] + 5

                    @m.hook(return_pc)
                    def unconstrain_hook(state):
                        if False:
                            return 10
                        '\n                        unconstrain_hook writes unconstrained symbolic data to the memory location of the output.\n                        '
                        with m.locked_context() as context:
                            context['return_addr'] = cpu.RAX
                            logging.debug(f'Writing unconstrained buffer to output memory location')
                            return_buf = state.new_symbolic_buffer(BUFFER_SIZE)
                            for i in range(BUFFER_SIZE):
                                if args.constraint == 'alpha':
                                    state.constrain(operators.OR(operators.AND(ord('A') <= return_buf[i], return_buf[i] <= ord('Z')), operators.AND(ord('a') <= return_buf[i], return_buf[i] <= ord('z'))))
                                elif args.constraint == 'num':
                                    state.constrain(operators.AND(ord('0') <= return_buf[i], return_buf[i] <= ord('9')))
                                elif args.constraint == 'alphanum':
                                    raise NotImplementedError('alphanum constraint set not yet implemented')
                                elif args.constraint == 'ascii':
                                    state.constrain(operators.AND(ord(' ') <= return_buf[i], return_buf[i] <= ord('}')))
                            state.cpu.write_bytes(context['return_addr'], return_buf)

        @m.hook(m.resolve(sym))
        def concolic_hook(state):
            if False:
                i = 10
                return i + 15
            '\n            hook used in order to concretize the execution of a `call <sym>` instruction\n            '
            cpu = state.cpu
            with m.locked_context() as context:
                call_pc = context['trace'][-1]
                state.cpu.PC = call_pc
                logging.debug(f'Concretely executing `call <{sym}>` at {hex(call_pc)}')
                state.cpu.decode_instruction(state.cpu.PC)
                emu = UnicornEmulator(state.cpu)
                emu.emulate(state.cpu.instruction)
                logging.debug('Continuing with Manticore symbolic execution')

    @m.hook(m.resolve(args.cmp_sym))
    def cmp_model(state):
        if False:
            print('Hello World!')
        "\n        used in order to invoke Manticore function model for strcmp and/or other comparison operation\n        calls. While a developer can write a test case using a crypto library's built in\n        constant-time comparison operation, it is preferable to use strcmp().\n        "
        logging.debug('Invoking model for comparsion call')
        state.invoke_model(strcmp)

    @m.hook(m.resolve('abort'))
    def fail_state(state):
        if False:
            while True:
                i = 10
        '\n        hook attached at fail state signified by abort call, which indicates that an edge case\n        input is provided and the abort() call is made\n        '
        logging.debug('Entering edge case path')
        with m.locked_context() as context:
            solution = state.solve_one(context['return_addr'], BUFFER_SIZE)
            print(f'Solution found: {solution}')
            rand_str = lambda n: ''.join([random.choice(string.ascii_lowercase) for i in range(n)])
            with open(m.workspace + '/' + 'sandshrew_' + rand_str(4), 'w') as fd:
                fd.write(str(solution))
        m.terminate()
    m.run()
    print(f"Total instructions: {len(m.context['trace'])}\nLast instruction: {hex(m.context['trace'][-1])}")
    return 0
if __name__ == '__main__':
    main()