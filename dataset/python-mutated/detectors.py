import hashlib
from typing import Optional
import logging
from contextlib import contextmanager
from ..core.smtlib import Operators, Constant, simplify, istainted, issymbolic, get_taints, taint_with
from ..core.plugin import Plugin
from ..utils.enums import DetectorClassification
logger = logging.getLogger(__name__)

class Detector(Plugin):
    ARGUMENT: Optional[str] = None
    HELP: Optional[str] = None
    IMPACT: Optional[DetectorClassification] = None
    CONFIDENCE: Optional[DetectorClassification] = None

    @property
    def name(self):
        if False:
            return 10
        return self.__class__.__name__.split('.')[-1]

    def get_findings(self, state):
        if False:
            for i in range(10):
                print('nop')
        return state.context.setdefault('{:s}.findings'.format(self.name), list())

    @contextmanager
    def locked_global_findings(self):
        if False:
            while True:
                i = 10
        with self.manticore.locked_context('{:s}.global_findings'.format(self.name), list) as global_findings:
            yield global_findings

    @property
    def global_findings(self):
        if False:
            print('Hello World!')
        with self.locked_global_findings() as global_findings:
            return global_findings

    def add_finding(self, state, address, pc, finding, at_init, constraint=True):
        if False:
            while True:
                i = 10
        '\n        Logs a finding at specified contract and assembler line.\n        :param state: current state\n        :param address: contract address of the finding\n        :param pc: program counter of the finding\n        :param at_init: true if executing the constructor\n        :param finding: textual description of the finding\n        :param constraint: finding is considered reproducible only when constraint is True\n        '
        if issymbolic(pc):
            pc = simplify(pc)
        if isinstance(pc, Constant):
            pc = pc.value
        if not isinstance(pc, int):
            raise ValueError('PC must be a number')
        self.get_findings(state).append((address, pc, finding, at_init, constraint))
        with self.locked_global_findings() as gf:
            gf.append((address, pc, finding, at_init))
        logger.warning(finding)

    def add_finding_here(self, state, finding, constraint=True):
        if False:
            print('Hello World!')
        '\n        Logs a finding in current contract and assembler line.\n        :param state: current state\n        :param finding: textual description of the finding\n        :param constraint: finding is considered reproducible only when constraint is True\n        '
        address = state.platform.current_vm.address
        pc = state.platform.current_vm.pc
        at_init = state.platform.current_transaction.sort == 'CREATE'
        self.add_finding(state, address, pc, finding, at_init, constraint)

    def _save_current_location(self, state, finding, condition=True):
        if False:
            print('Hello World!')
        '\n        Save current location in the internal locations list and returns a textual id for it.\n        This is used to save locations that could later be promoted to a finding if other conditions hold\n        See _get_location()\n        :param state: current state\n        :param finding: textual description of the finding\n        :param condition: general purpose constraint\n        '
        address = state.platform.current_vm.address
        pc = state.platform.current_vm.pc
        at_init = state.platform.current_transaction.sort == 'CREATE'
        location = (address, pc, finding, at_init, condition)
        hash_id = hashlib.sha1(str(location).encode()).hexdigest()
        state.context.setdefault('{:s}.locations'.format(self.name), {})[hash_id] = location
        return hash_id

    def _get_location(self, state, hash_id):
        if False:
            while True:
                i = 10
        '\n        Get previously saved location\n        A location is composed of: address, pc, finding, at_init, condition\n        '
        return state.context.setdefault('{:s}.locations'.format(self.name), {})[hash_id]

    def _get_src(self, address, pc):
        if False:
            for i in range(10):
                print('nop')
        return self.manticore.get_metadata(address).get_source_for(pc)

class DetectEnvInstruction(Detector):
    """
    Detect the usage of instructions that query environmental/block information:
    BLOCKHASH, COINBASE, TIMESTAMP, NUMBER, DIFFICULTY, GASLIMIT, ORIGIN, GASPRICE

    Sometimes environmental information can be manipulated. Contracts should avoid
    using it. Unless special situations. Notably to programatically detect human transactions
    `sender == origin`
    """
    ARGUMENT = 'env-instr'
    HELP = 'Use of potentially unsafe/manipulable instructions'
    IMPACT = DetectorClassification.MEDIUM
    CONFIDENCE = DetectorClassification.HIGH

    def will_evm_execute_instruction_callback(self, state, instruction, arguments):
        if False:
            print('Hello World!')
        if instruction.semantics in ('BLOCKHASH', 'COINBASE', 'TIMESTAMP', 'NUMBER', 'DIFFICULTY', 'GASLIMIT', 'ORIGIN', 'GASPRICE'):
            self.add_finding_here(state, f'Warning {instruction.semantics} instruction used')

class DetectSuicidal(Detector):
    ARGUMENT = 'suicidal'
    HELP = 'Reachable selfdestruct instructions'
    IMPACT = DetectorClassification.MEDIUM
    CONFIDENCE = DetectorClassification.HIGH

    def will_evm_execute_instruction_callback(self, state, instruction, arguments):
        if False:
            return 10
        if instruction.semantics == 'SELFDESTRUCT':
            self.add_finding_here(state, 'Reachable SELFDESTRUCT')

class DetectExternalCallAndLeak(Detector):
    ARGUMENT = 'ext-call-leak'
    HELP = 'Reachable external call or ether leak to sender or arbitrary address'
    IMPACT = DetectorClassification.MEDIUM
    CONFIDENCE = DetectorClassification.HIGH

    def will_evm_execute_instruction_callback(self, state, instruction, arguments):
        if False:
            for i in range(10):
                print('nop')
        if instruction.semantics == 'CALL':
            dest_address = arguments[1]
            sent_value = arguments[2]
            msg_sender = state.platform.current_vm.caller
            if issymbolic(dest_address):
                self.add_finding_here(state, f'Reachable ether leak to sender via argument', constraint=AND(msg_sender == dest_address, sent_value != 0))
                self.add_finding_here(state, f'Reachable external call to sender via argument', constraint=AND(msg_sender == dest_address, sent_value == 0))
                possible_destinations = state.solve_n(dest_address, 2)
                if len(possible_destinations) > 1:
                    self.add_finding_here(state, f'Reachable ether leak to user controlled address via argument', constraint=AND(msg_sender != dest_address, sent_value != 0))
                    self.add_finding_here(state, f'Reachable external call to user controlled address via argument', constraint=AND(msg_sender != dest_address, sent_value == 0))
            elif msg_sender == dest_address:
                self.add_finding_here(state, f'Reachable ether leak to sender', constraint=sent_value != 0)
                self.add_finding_here(state, f'Reachable external call to sender', constraint=sent_value == 0)

class DetectInvalid(Detector):
    ARGUMENT = 'invalid'
    HELP = 'Enable INVALID instruction detection'
    IMPACT = DetectorClassification.LOW
    CONFIDENCE = DetectorClassification.HIGH

    def __init__(self, only_human=True, **kwargs):
        if False:
            return 10
        '\n        Detects INVALID instructions.\n\n        INVALID instructions are originally designated to signal exceptional code.\n        As in practice the INVALID instruction is used in different ways this\n        detector may Generate a great deal of false positives.\n\n        :param only_human: if True report only INVALID at depth 0 transactions\n        '
        super().__init__(**kwargs)
        self._only_human = only_human

    def will_evm_execute_instruction_callback(self, state, instruction, arguments):
        if False:
            while True:
                i = 10
        mnemonic = instruction.semantics
        if mnemonic == 'INVALID':
            if not self._only_human or state.platform.current_transaction.depth == 0:
                self.add_finding_here(state, 'INVALID instruction')

class DetectReentrancySimple(Detector):
    """
    Simple detector for reentrancy bugs.
    Alert if contract changes the state of storage (does a write) after a call with >2300 gas to a user controlled/symbolic
    external address or the msg.sender address.
    """
    ARGUMENT = 'reentrancy'
    HELP = 'Reentrancy bug'
    IMPACT = DetectorClassification.HIGH
    CONFIDENCE = DetectorClassification.HIGH

    @property
    def _context_key(self):
        if False:
            return 10
        return f'{self.name}.call_locations'

    def will_open_transaction_callback(self, state, tx):
        if False:
            for i in range(10):
                print('nop')
        if tx.is_human:
            state.context[self._context_key] = []

    def will_evm_execute_instruction_callback(self, state, instruction, arguments):
        if False:
            for i in range(10):
                print('nop')
        if instruction.semantics == 'CALL':
            gas = arguments[0]
            dest_address = arguments[1]
            msg_sender = state.platform.current_vm.caller
            pc = state.platform.current_vm.pc
            is_enough_gas = Operators.UGT(gas, 2300)
            if not state.can_be_true(is_enough_gas):
                return
            if issymbolic(dest_address) or msg_sender == dest_address:
                state.context.get(self._context_key, []).append((pc, is_enough_gas))

    def did_evm_write_storage_callback(self, state, address, offset, value):
        if False:
            print('Hello World!')
        locs = state.context.get(self._context_key, [])
        for (callpc, gas_constraint) in locs:
            addr = state.platform.current_vm.address
            at_init = state.platform.current_transaction.sort == 'CREATE'
            self.add_finding(state, addr, callpc, 'Potential reentrancy vulnerability', at_init, constraint=gas_constraint)

class DetectReentrancyAdvanced(Detector):
    """
    Detector for reentrancy bugs.
    Given an optional concrete list of attacker addresses, warn on the following conditions.

    1) A _successful_ call to an attacker address (address in attacker list), or any human account address
    (if no list is given). With enough gas (>2300).

    2) A SSTORE after the execution of the CALL.

    3) The storage slot of the SSTORE must be used in some path to control flow
    """
    ARGUMENT = 'reentrancy-adv'
    HELP = 'Reentrancy bug (different method)'
    IMPACT = DetectorClassification.HIGH
    CONFIDENCE = DetectorClassification.HIGH

    def __init__(self, addresses=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self._addresses = addresses

    @property
    def _read_storage_name(self):
        if False:
            i = 10
            return i + 15
        return '{:s}.read_storage'.format(self.name)

    def will_open_transaction_callback(self, state, tx):
        if False:
            i = 10
            return i + 15
        if tx.is_human:
            state.context[self._read_storage_name] = set()
            state.context['{:s}.locations'.format(self.name)] = dict()

    def did_close_transaction_callback(self, state, tx):
        if False:
            while True:
                i = 10
        world = state.platform
        if not tx.is_human:
            if tx.result:
                if state.can_be_true(Operators.UGE(tx.gas, 2300)):
                    if self._addresses is None and (not world.get_code(tx.address)) or (self._addresses is not None and tx.address in self._addresses):
                        self._save_location_and_reads(state)

    def _save_location_and_reads(self, state):
        if False:
            print('Hello World!')
        name = '{:s}.locations'.format(self.name)
        locations = state.context.get(name, dict)
        world = state.platform
        address = world.current_vm.address
        pc = world.current_vm.pc
        if isinstance(pc, Constant):
            pc = pc.value
        assert isinstance(pc, int)
        at_init = world.current_transaction.sort == 'CREATE'
        location = (address, pc, 'Reentrancy multi-million ether bug', at_init)
        locations[location] = set(state.context[self._read_storage_name])
        state.context[name] = locations

    def _get_location_and_reads(self, state):
        if False:
            while True:
                i = 10
        name = '{:s}.locations'.format(self.name)
        locations = state.context.get(name, dict)
        return locations.items()

    def did_evm_read_storage_callback(self, state, address, offset, value):
        if False:
            print('Hello World!')
        state.context[self._read_storage_name].add((address, offset))

    def did_evm_write_storage_callback(self, state, address, offset, value):
        if False:
            for i in range(10):
                print('nop')
        for (location, reads) in self._get_location_and_reads(state):
            for (address_i, offset_i) in reads:
                if address_i == address:
                    if state.can_be_true(offset == offset_i):
                        self.add_finding(state, *location)

class DetectIntegerOverflow(Detector):
    """
    Detects potential overflow and underflow conditions on ADD and SUB instructions.
    """
    ARGUMENT = 'overflow'
    HELP = 'Integer overflows'
    IMPACT = DetectorClassification.HIGH
    CONFIDENCE = DetectorClassification.HIGH

    @staticmethod
    def _signed_sub_overflow(state, a, b):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sign extend the value to 512 bits and check the result can be represented\n         in 256. Following there is a 32 bit excerpt of this condition:\n        a  -  b   -80000000 -3fffffff -00000001 +00000000 +00000001 +3fffffff +7fffffff\n        +80000000    False    False    False    False     True     True     True\n        +c0000001    False    False    False    False    False    False     True\n        +ffffffff    False    False    False    False    False    False    False\n        +00000000     True    False    False    False    False    False    False\n        +00000001     True    False    False    False    False    False    False\n        +3fffffff     True    False    False    False    False    False    False\n        +7fffffff     True     True     True    False    False    False    False\n        '
        sub = Operators.SEXTEND(a, 256, 512) - Operators.SEXTEND(b, 256, 512)
        cond = Operators.OR(sub < -(1 << 255), sub >= 1 << 255)
        return cond

    @staticmethod
    def _signed_add_overflow(state, a, b):
        if False:
            return 10
        '\n        Sign extend the value to 512 bits and check the result can be represented\n         in 256. Following there is a 32 bit excerpt of this condition:\n\n        a  +  b   -80000000 -3fffffff -00000001 +00000000 +00000001 +3fffffff +7fffffff\n        +80000000     True     True     True    False    False    False    False\n        +c0000001     True    False    False    False    False    False    False\n        +ffffffff     True    False    False    False    False    False    False\n        +00000000    False    False    False    False    False    False    False\n        +00000001    False    False    False    False    False    False     True\n        +3fffffff    False    False    False    False    False    False     True\n        +7fffffff    False    False    False    False     True     True     True\n        '
        add = Operators.SEXTEND(a, 256, 512) + Operators.SEXTEND(b, 256, 512)
        cond = Operators.OR(add < -(1 << 255), add >= 1 << 255)
        return cond

    @staticmethod
    def _unsigned_sub_overflow(state, a, b):
        if False:
            i = 10
            return i + 15
        '\n        Sign extend the value to 512 bits and check the result can be represented\n         in 256. Following there is a 32 bit excerpt of this condition:\n\n        a  -  b   ffffffff bfffffff 80000001 00000000 00000001 3ffffffff 7fffffff\n        ffffffff     True     True     True    False     True     True     True\n        bfffffff     True     True     True    False    False     True     True\n        80000001     True     True     True    False    False     True     True\n        00000000    False    False    False    False    False     True    False\n        00000001     True    False    False    False    False     True    False\n        ffffffff     True     True     True     True     True     True     True\n        7fffffff     True     True     True    False    False     True    False\n        '
        cond = Operators.UGT(b, a)
        return cond

    @staticmethod
    def _unsigned_add_overflow(state, a, b):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sign extend the value to 512 bits and check the result can be represented\n         in 256. Following there is a 32 bit excerpt of this condition:\n\n        a  +  b   ffffffff bfffffff 80000001 00000000 00000001 3ffffffff 7fffffff\n        ffffffff     True     True     True    False     True     True     True\n        bfffffff     True     True     True    False    False     True     True\n        80000001     True     True     True    False    False     True     True\n        00000000    False    False    False    False    False     True    False\n        00000001     True    False    False    False    False     True    False\n        ffffffff     True     True     True     True     True     True     True\n        7fffffff     True     True     True    False    False     True    False\n        '
        add = Operators.ZEXTEND(a, 512) + Operators.ZEXTEND(b, 512)
        cond = Operators.UGE(add, 1 << 256)
        return cond

    @staticmethod
    def _signed_mul_overflow(state, a, b):
        if False:
            while True:
                i = 10
        '\n        Sign extend the value to 512 bits and check the result can be represented\n         in 256. Following there is a 32 bit excerpt of this condition:\n\n        a  *  b           +00000000000000000 +00000000000000001 +0000000003fffffff +0000000007fffffff +00000000080000001 +000000000bfffffff +000000000ffffffff\n        +0000000000000000  +0000000000000000  +0000000000000000  +0000000000000000  +0000000000000000  +0000000000000000  +0000000000000000  +0000000000000000\n        +0000000000000001  +0000000000000000  +0000000000000001  +000000003fffffff  +000000007fffffff  +0000000080000001  +00000000bfffffff  +00000000ffffffff\n        +000000003fffffff  +0000000000000000  +000000003fffffff *+0fffffff80000001 *+1fffffff40000001 *+1fffffffbfffffff *+2fffffff00000001 *+3ffffffec0000001\n        +000000007fffffff  +0000000000000000  +000000007fffffff *+1fffffff40000001 *+3fffffff00000001 *+3fffffffffffffff *+5ffffffec0000001 *+7ffffffe80000001\n        +0000000080000001  +0000000000000000  +0000000080000001 *+1fffffffbfffffff *+3fffffffffffffff *+4000000100000001 *+600000003fffffff *+800000007fffffff\n        +00000000bfffffff  +0000000000000000  +00000000bfffffff *+2fffffff00000001 *+5ffffffec0000001 *+600000003fffffff *+8ffffffe80000001 *+bffffffe40000001\n        +00000000ffffffff  +0000000000000000  +00000000ffffffff *+3ffffffec0000001 *+7ffffffe80000001 *+800000007fffffff *+bffffffe40000001 *+fffffffe00000001\n\n        '
        mul = Operators.SEXTEND(a, 256, 512) * Operators.SEXTEND(b, 256, 512)
        cond = Operators.OR(mul < -(1 << 255), mul >= 1 << 255)
        return cond

    @staticmethod
    def _unsigned_mul_overflow(state, a, b):
        if False:
            while True:
                i = 10
        '\n        Sign extend the value to 512 bits and check the result can be represented\n         in 256. Following there is a 32 bit excerpt of this condition:\n\n        a  *  b           +00000000000000000 +00000000000000001 +0000000003fffffff +0000000007fffffff +00000000080000001 +000000000bfffffff +000000000ffffffff\n        +0000000000000000  +0000000000000000  +0000000000000000  +0000000000000000  +0000000000000000  +0000000000000000  +0000000000000000  +0000000000000000\n        +0000000000000001  +0000000000000000  +0000000000000001  +000000003fffffff  +000000007fffffff  +0000000080000001  +00000000bfffffff  +00000000ffffffff\n        +000000003fffffff  +0000000000000000  +000000003fffffff *+0fffffff80000001 *+1fffffff40000001 *+1fffffffbfffffff *+2fffffff00000001 *+3ffffffec0000001\n        +000000007fffffff  +0000000000000000  +000000007fffffff *+1fffffff40000001 *+3fffffff00000001 *+3fffffffffffffff *+5ffffffec0000001 *+7ffffffe80000001\n        +0000000080000001  +0000000000000000  +0000000080000001 *+1fffffffbfffffff *+3fffffffffffffff *+4000000100000001 *+600000003fffffff *+800000007fffffff\n        +00000000bfffffff  +0000000000000000  +00000000bfffffff *+2fffffff00000001 *+5ffffffec0000001 *+600000003fffffff *+8ffffffe80000001 *+bffffffe40000001\n        +00000000ffffffff  +0000000000000000  +00000000ffffffff *+3ffffffec0000001 *+7ffffffe80000001 *+800000007fffffff *+bffffffe40000001 *+fffffffe00000001\n\n        '
        mul = Operators.SEXTEND(a, 256, 512) * Operators.SEXTEND(b, 256, 512)
        cond = Operators.UGE(mul, 1 << 256)
        return cond

    def _check_finding(self, state, what):
        if False:
            return 10
        if istainted(what, 'SIGNED'):
            for taint in get_taints(what, 'IOS_.*'):
                (address, pc, finding, at_init, condition) = self._get_location(state, taint[4:])
                if state.can_be_true(condition):
                    self.add_finding(state, address, pc, finding, at_init, condition)
        else:
            for taint in get_taints(what, 'IOU_.*'):
                (address, pc, finding, at_init, condition) = self._get_location(state, taint[4:])
                if state.can_be_true(condition):
                    self.add_finding(state, address, pc, finding, at_init, condition)

    def did_evm_execute_instruction_callback(self, state, instruction, arguments, result):
        if False:
            print('Hello World!')
        vm = state.platform.current_vm
        mnemonic = instruction.semantics
        ios = False
        iou = False
        if mnemonic == 'ADD':
            ios = self._signed_add_overflow(state, *arguments)
            iou = self._unsigned_add_overflow(state, *arguments)
        elif mnemonic == 'MUL':
            ios = self._signed_mul_overflow(state, *arguments)
            iou = self._unsigned_mul_overflow(state, *arguments)
        elif mnemonic == 'SUB':
            ios = self._signed_sub_overflow(state, *arguments)
            iou = self._unsigned_sub_overflow(state, *arguments)
        elif mnemonic == 'SSTORE':
            (where, what) = arguments
            self._check_finding(state, what)
        elif mnemonic == 'RETURN':
            world = state.platform
            if world.current_transaction.is_human:
                (offset, size) = arguments
                data = world.current_vm.read_buffer(offset, size)
                self._check_finding(state, data)
        if mnemonic in ('SLT', 'SGT', 'SDIV', 'SMOD'):
            result = taint_with(result, 'SIGNED')
        if mnemonic in ('ADD', 'SUB', 'MUL'):
            id_val = self._save_current_location(state, 'Signed integer overflow at %s instruction' % mnemonic, ios)
            result = taint_with(result, 'IOS_{:s}'.format(id_val))
            id_val = self._save_current_location(state, 'Unsigned integer overflow at %s instruction' % mnemonic, iou)
            result = taint_with(result, 'IOU_{:s}'.format(id_val))
        if mnemonic in ('SLT', 'SGT', 'SDIV', 'SMOD', 'ADD', 'SUB', 'MUL'):
            vm.change_last_result(result)

class DetectUnusedRetVal(Detector):
    """Detects unused return value from internal transactions"""
    ARGUMENT = 'unused-return'
    HELP = 'Unused internal transaction return values'
    IMPACT = DetectorClassification.LOW
    CONFIDENCE = DetectorClassification.HIGH

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        self._stack_name = '{:s}.stack'.format(self.name)

    def _add_retval_taint(self, state, taint):
        if False:
            print('Hello World!')
        taints = state.context[self._stack_name][-1]
        taints.add(taint)
        state.context[self._stack_name][-1] = taints

    def _remove_retval_taint(self, state, taint):
        if False:
            for i in range(10):
                print('nop')
        taints = state.context[self._stack_name][-1]
        if taint in taints:
            taints.remove(taint)
            state.context[self._stack_name][-1] = taints

    def _get_retval_taints(self, state):
        if False:
            while True:
                i = 10
        return state.context[self._stack_name][-1]

    def will_open_transaction_callback(self, state, tx):
        if False:
            for i in range(10):
                print('nop')
        if tx.is_human:
            state.context[self._stack_name] = []
        state.context[self._stack_name].append(set())

    def did_close_transaction_callback(self, state, tx):
        if False:
            print('Hello World!')
        world = state.platform
        for taint in self._get_retval_taints(state):
            id_val = taint[7:]
            (address, pc, finding, at_init, condition) = self._get_location(state, id_val)
            if state.can_be_true(condition):
                self.add_finding(state, address, pc, finding, at_init)
        state.context[self._stack_name].pop()

    def did_evm_execute_instruction_callback(self, state, instruction, arguments, result):
        if False:
            i = 10
            return i + 15
        world = state.platform
        mnemonic = instruction.semantics
        current_vm = world.current_vm
        if instruction.is_starttx:
            id_val = self._save_current_location(state, 'Returned value at {:s} instruction is not used'.format(mnemonic))
            taint = 'RETVAL_{:s}'.format(id_val)
            current_vm.change_last_result(taint_with(result, taint))
            self._add_retval_taint(state, taint)
        elif mnemonic == 'JUMPI':
            (dest, cond) = arguments
            for used_taint in get_taints(cond, 'RETVAL_.*'):
                self._remove_retval_taint(state, used_taint)

class DetectDelegatecall(Detector):
    """
    Detects DELEGATECALLs to controlled addresses and or with controlled function id.
    This detector finds and reports on any delegatecall instruction any the following propositions are hold:
        * the destination address can be controlled by the caller
        * the first 4 bytes of the calldata are controlled by the caller
    """
    ARGUMENT = 'delegatecall'
    HELP = 'Problematic uses of DELEGATECALL instruction'
    IMPACT = DetectorClassification.HIGH
    CONFIDENCE = DetectorClassification.HIGH

    def _to_constant(self, expression):
        if False:
            while True:
                i = 10
        if isinstance(expression, Constant):
            return expression.value
        return expression

    def will_evm_execute_instruction_callback(self, state, instruction, arguments):
        if False:
            while True:
                i = 10
        world = state.platform
        mnemonic = instruction.semantics
        if mnemonic == 'DELEGATECALL':
            (gas, address, in_offset, in_size, out_offset, out_size) = arguments
            if issymbolic(address):
                possible_addresses = state.solve_n(address, 2)
                if len(possible_addresses) > 1:
                    self.add_finding_here(state, 'Delegatecall to user controlled address')
            in_offset = self._to_constant(in_offset)
            in_size = self._to_constant(in_size)
            calldata = world.current_vm.read_buffer(in_offset, in_size)
            func_id = calldata[:4]
            if issymbolic(func_id):
                possible_func_ids = state.solve_n(func_id, 2)
                if len(possible_func_ids) > 1:
                    self.add_finding_here(state, 'Delegatecall to user controlled function')

class DetectUninitializedMemory(Detector):
    """
    Detects uses of uninitialized memory
    """
    ARGUMENT = 'uninitialized-memory'
    HELP = 'Uninitialized memory usage'
    IMPACT = DetectorClassification.MEDIUM
    CONFIDENCE = DetectorClassification.HIGH

    def did_evm_read_memory_callback(self, state, offset, value, size):
        if False:
            while True:
                i = 10
        initialized_memory = state.context.get('{:s}.initialized_memory'.format(self.name), set())
        cbu = True
        current_contract = state.platform.current_vm.address
        for (known_contract, known_offset) in initialized_memory:
            if current_contract == known_contract:
                for offset_i in range(size):
                    cbu = Operators.AND(cbu, offset + offset_i != known_offset)
        if state.can_be_true(cbu):
            self.add_finding_here(state, 'Potentially reading uninitialized memory at instruction (address: %r, offset %r)' % (current_contract, offset))

    def did_evm_write_memory_callback(self, state, offset, value, size):
        if False:
            while True:
                i = 10
        current_contract = state.platform.current_vm.address
        for offset_i in range(size):
            state.context.setdefault('{:s}.initialized_memory'.format(self.name), set()).add((current_contract, offset + offset_i))

class DetectUninitializedStorage(Detector):
    """
    Detects uses of uninitialized storage
    """
    ARGUMENT = 'uninitialized-storage'
    HELP = 'Uninitialized storage usage'
    IMPACT = DetectorClassification.MEDIUM
    CONFIDENCE = DetectorClassification.HIGH

    def did_evm_read_storage_callback(self, state, address, offset, value):
        if False:
            print('Hello World!')
        if not state.can_be_true(value != 0):
            return
        cbu = True
        context_name = '{:s}.initialized_storage'.format(self.name)
        for (known_address, known_offset) in state.context.get(context_name, ()):
            cbu = Operators.AND(cbu, Operators.OR(address != known_address, offset != known_offset))
        if state.can_be_true(cbu):
            self.add_finding_here(state, 'Potentially reading uninitialized storage', cbu)

    def did_evm_write_storage_callback(self, state, address, offset, value):
        if False:
            return 10
        state.context.setdefault('{:s}.initialized_storage'.format(self.name), set()).add((address, offset))

class DetectRaceCondition(Detector):
    """
    Detects possible transaction race conditions (transaction order dependencies)

    The RaceCondition detector might not work properly for contracts that have only a fallback function.
    See the detector's implementation and it's `_in_user_func` method for more information.
    """
    ARGUMENT = 'race-condition'
    HELP = 'Possible transaction race conditions'
    IMPACT = DetectorClassification.LOW
    CONFIDENCE = DetectorClassification.LOW
    TAINT = 'written_storage_slots.'

    def __init__(self, *a, **kw):
        if False:
            for i in range(10):
                print('nop')
        self.__findings = set()
        super().__init__(*a, **kw)

    @staticmethod
    def _in_user_func(state):
        if False:
            i = 10
            return i + 15
        '\n        :param state: current state\n        :return: whether the current execution is in a user-defined function or not.\n\n        NOTE / TODO / FIXME: As this may produce false postives, this is not in the base `Detector` class.\n        It should be fixed at some point and moved there. See below.\n\n        The first 4 bytes of tx data is keccak256 hash of the function signature that is called by given tx.\n\n        All transactions start within Solidity dispatcher function: it takes passed hash and dispatches\n        the execution to given function based on it.\n\n        So: if we are in the dispatcher, *and contract have some functions* one of the first four tx data bytes\n        will effectively have more than one solutions.\n\n        BUT if contract have only a fallback function, the equation below may return more solutions when we are\n        in a dispatcher function.  <--- because of that, we warn that the detector is not that stable\n        for contracts with only a fallback function.\n        '
        in_function = state.context.get('in_function', False)
        prev_tx_count = state.context.get('prev_tx_count', 0)
        curr_tx_count = len(state.platform.transactions)
        new_human_tx = prev_tx_count != curr_tx_count
        if in_function and (not new_human_tx):
            return True
        in_function = len(state.solve_n(state.platform.current_transaction.data[:4], 2)) == 1
        state.context['in_function'] = in_function
        state.context['prev_tx_count'] = curr_tx_count
        return in_function

    def did_evm_write_storage_callback(self, state, storage_address, offset, value):
        if False:
            for i in range(10):
                print('nop')
        world = state.platform
        curr_tx = world.current_transaction
        if curr_tx.sort == 'CREATE' or not self._in_user_func(state):
            return
        key = self.TAINT + str(offset)
        result = taint_with(value, key)
        world.set_storage_data(storage_address, offset, result)
        metadata = self.manticore.metadata[curr_tx.address]
        func_sig = metadata.get_func_signature(state.solve_one(curr_tx.data[:4]))
        state.context.setdefault(key, set()).add(func_sig)

    def did_evm_execute_instruction_callback(self, state, instruction, arguments, result_ref):
        if False:
            i = 10
            return i + 15
        if not self._in_user_func(state):
            return
        if not isinstance(state.platform.current_vm.pc, (int, Constant)):
            return
        world = state.platform
        curr_tx = world.current_transaction
        if curr_tx.sort != 'CREATE' and curr_tx.address in self.manticore.metadata:
            metadata = self.manticore.metadata[curr_tx.address]
            curr_func = metadata.get_func_signature(state.solve_one(curr_tx.data[:4]))
            for arg in arguments:
                if istainted(arg):
                    for taint in get_taints(arg, self.TAINT + '*'):
                        tainted_val = taint[taint.rindex('.') + 1:]
                        try:
                            storage_index = int(tainted_val)
                            storage_index_key = storage_index
                        except ValueError:
                            storage_index = 'which is symbolic'
                            storage_index_key = hash(tainted_val)
                        prev_funcs = state.context[taint]
                        for prev_func in prev_funcs:
                            if prev_func is None:
                                continue
                            msg = 'Potential race condition (transaction order dependency):\n'
                            msg += f'Value has been stored in storage slot/index {storage_index} in transaction that called {prev_func} and is now used in transaction that calls {curr_func}.\nAn attacker seeing a transaction to {curr_func} could create a transaction to {prev_func} with high gas and win a race.'
                            unique_key = (storage_index_key, prev_func, curr_func)
                            if unique_key in self.__findings:
                                continue
                            self.__findings.add(unique_key)
                            self.add_finding_here(state, msg)

class DetectManipulableBalance(Detector):
    """
    Detects the use of manipulable balance in strict compare.
    """
    ARGUMENT = 'lockdrop'
    HELP = 'Use balance in EQ'
    IMPACT = DetectorClassification.HIGH
    CONFIDENCE = DetectorClassification.HIGH

    def did_evm_execute_instruction_callback(self, state, instruction, arguments, result):
        if False:
            while True:
                i = 10
        vm = state.platform.current_vm
        mnemonic = instruction.semantics
        if mnemonic == 'BALANCE':
            result = taint_with(result, 'BALANCE')
            vm.change_last_result(result)
        elif mnemonic == 'EQ':
            for op in arguments:
                if istainted(op, 'BALANCE'):
                    self.add_finding_here(state, 'Manipulable balance used in a strict comparison')