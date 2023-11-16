import sys
from functools import reduce
import re
import logging
from ..core.plugin import Plugin
from ..core.smtlib import Operators, to_constant
from ..utils.enums import StateLists
import pyevmasm as EVMAsm
logger = logging.getLogger(__name__)

class FilterFunctions(Plugin):

    def __init__(self, regexp='.*', mutability='both', depth='both', fallback=False, include=True, **kwargs):
        if False:
            return 10
        "\n        Constrain input based on function metadata. Include or avoid functions\n        selected by the specified criteria.\n\n        Examples:\n        #Do not explore any human transactions that end up calling a constant function\n        no_human_constant = FilterFunctions(depth='human', mutability='constant', include=False)\n\n        #At human tx depth only accept synthetic check functions\n        only_tests = FilterFunctions(regexp=r'mcore_.*', depth='human', include=False)\n\n        :param regexp: a regular expression over the name of the function '.*' will match all functions\n        :param mutability: mutable, constant or both will match functions declared in the abi to be of such class\n        :param depth: match functions in internal transactions, in human initiated transactions or in both types\n        :param fallback: if True include the fallback function. Hash will be 00000000 for it\n        :param include: if False exclude the selected functions, if True include them\n        "
        super().__init__(**kwargs)
        depth = depth.lower()
        if depth not in ('human', 'internal', 'both'):
            raise ValueError
        mutability = mutability.lower()
        if mutability not in ('mutable', 'constant', 'both'):
            raise ValueError
        self._regexp = regexp
        self._mutability = mutability
        self._depth = depth
        self._fallback = fallback
        self._include = include

    def will_open_transaction_callback(self, state, tx):
        if False:
            for i in range(10):
                print('nop')
        world = state.platform
        tx_cnt = len(world.all_transactions)
        if state.context.get('constrained%d' % id(self), 0) != tx_cnt:
            state.context['constrained%d' % id(self)] = tx_cnt
            if self._depth == 'human' and (not tx.is_human):
                return
            if self._depth == 'internal' and tx.is_human:
                return
            md = self.manticore.get_metadata(tx.address)
            if md is None:
                return
            selected_functions = []
            for func_hsh in md.function_selectors:
                abi = md.get_abi(func_hsh)
                if abi['type'] == 'fallback':
                    continue
                if self._mutability == 'constant' and (not abi.get('constant', False)):
                    continue
                if self._mutability == 'mutable' and abi.get('constant', False):
                    continue
                if not re.match(self._regexp, abi['name']):
                    continue
                selected_functions.append(func_hsh)
            if self._fallback and md.has_non_default_fallback_function:
                selected_functions.append(md.fallback_function_selector)
            if self._include:
                if not selected_functions:
                    logger.warning('No functions selected, adding False to path constraint.')
                constraint = reduce(Operators.OR, (tx.data[:4] == x for x in selected_functions), False)
                state.constrain(constraint)
            else:
                constraint = True
                for func_hsh in md.function_selectors:
                    if func_hsh in selected_functions:
                        constraint = Operators.AND(tx.data[:4] != func_hsh, constraint)
                state.constrain(constraint)

class LoopDepthLimiter(Plugin):
    """This just aborts explorations that are too deep"""

    def __init__(self, loop_count_threshold=5, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.loop_count_threshold = loop_count_threshold

    def will_run_callback(self, *args):
        if False:
            while True:
                i = 10
        with self.manticore.locked_context('seen_rep', dict) as reps:
            reps.clear()

    def will_execute_instruction_callback(self, state, pc, insn):
        if False:
            return 10
        world = state.platform
        with self.manticore.locked_context('seen_rep', dict) as reps:
            item = (world.current_transaction.sort == 'CREATE', world.current_transaction.address, pc)
            if item not in reps:
                reps[item] = 0
            reps[item] += 1
            if reps[item] > self.loop_count_threshold:
                state.abandon()

class VerboseTrace(Plugin):
    """
    Generates a verbose trace of EVM execution and saves in workspace into `state<id>.trace`.

    Example output can be seen in test_eth_plugins.
    """

    def will_evm_execute_instruction_callback(self, state, instruction, arguments):
        if False:
            print('Hello World!')
        current_vm = state.platform.current_vm
        state.context.setdefault('str_trace', []).append(str(current_vm))

    def generate_testcase(self, state, testcase, message):
        if False:
            for i in range(10):
                print('nop')
        trace = state.context.get('str_trace', [])
        with testcase.open_stream('verbose_trace') as vt:
            for t in trace:
                vt.write(t + '\n')

class VerboseTraceStdout(Plugin):
    """
    Same as VerboseTrace but prints to stdout. Note that you should use it only if Manticore
    is run with procs=1 as otherwise, the output will be clobbered.
    """

    def will_evm_execute_instruction_callback(self, state, instruction, arguments):
        if False:
            i = 10
            return i + 15
        print(state.platform.current_vm)

class KeepOnlyIfStorageChanges(Plugin):
    """This plugin discards all transactions that results in states where
    the underlying EVM storage did not change or in other words,
    there were no writes to it.

    This allows to speed-up EVM engine exploration as we don't
    explore states that have the same storage (contract data).

    However, keep in mind that if the (contract) code relies on
    account balance and the balance is not a symbolic value
    it might be that a certain state will not be covered by the
    execution when this plugin is used.
    """

    def did_open_transaction_callback(self, state, tx, *args):
        if False:
            for i in range(10):
                print('nop')
        'We need a stack. Each tx (internal or not) starts with a "False" flag\n        denoting that it did not write anything to the storage\n        '
        state.context['written'].append(False)

    def did_close_transaction_callback(self, state, tx, *args):
        if False:
            return 10
        'When a tx (internal or not) is closed a value is popped out from the\n        flag stack. Depending on the result if the storage is not rolled back the\n        next flag in the stack is updated. Not that if the a tx is reverted the\n        changes it may have done on the storage will not affect the final result.\n\n        '
        flag = state.context['written'].pop()
        if tx.result in {'RETURN', 'STOP'}:
            code_written = tx.result == 'RETURN' and tx.sort == 'CREATE'
            flag = flag or code_written
            if not flag:
                ether_sent = state.can_be_true(tx.value != 0)
                flag = flag or ether_sent
            state.context['written'][-1] = state.context['written'][-1] or flag

    def did_evm_write_storage_callback(self, state, *args):
        if False:
            return 10
        'Turn on the corresponding flag is the storage has been modified.\n        Note: subject to change if the current transaction is reverted'
        state.context['written'][-1] = True

    def will_run_callback(self, *args):
        if False:
            i = 10
            return i + 15
        'Initialize the flag stack at each human tx/run()'
        for st in self.manticore.ready_states:
            st.context['written'] = [False]

    def did_run_callback(self):
        if False:
            return 10
        'When  human tx/run just ended remove the states that have not changed\n        the storage'
        with self.manticore.locked_context('ethereum.saved_states', list) as saved_states:
            for state_id in list(saved_states):
                st = self.manticore._load(state_id)
                if not st.context['written'][-1]:
                    if st.id in self.manticore._ready_states:
                        self._publish('will_transition_state', state_id, StateLists.ready, StateLists.terminated)
                        self.manticore._ready_states.remove(st.id)
                        self.manticore._terminated_states.append(st.id)
                        self._publish('did_transition_state', state_id, StateLists.ready, StateLists.terminated)
                    saved_states.remove(st.id)

    def generate_testcase(self, state, testcase, message):
        if False:
            i = 10
            return i + 15
        with testcase.open_stream('summary') as stream:
            if not state.context.get('written', (False,))[-1]:
                stream.write('State was removed from ready list because the last tx did not write to the storage')

class SkipRevertBasicBlocks(Plugin):

    def _is_revert_bb(self, state, pc):
        if False:
            print('Hello World!')
        world = state.platform

        def read_code(_pc=None):
            if False:
                i = 10
                return i + 15
            while True:
                yield to_constant(world.current_vm.read_code(_pc)[0])
                _pc += 1
        for inst in EVMAsm.disassemble_all(read_code(pc), pc):
            if inst.name == 'REVERT':
                return True
            if inst.is_terminator:
                return False

    def will_evm_execute_instruction_callback(self, state, instruction, arguments):
        if False:
            return 10
        world = state.platform
        if state.platform.current_transaction.sort != 'CREATE':
            if instruction.semantics == 'JUMPI':
                if self._is_revert_bb(state, world.current_vm.pc + instruction.size):
                    state.constrain(arguments[1] == True)
                if self._is_revert_bb(state, arguments[0]):
                    state.constrain(arguments[1] == False)