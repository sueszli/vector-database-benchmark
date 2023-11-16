from manticore.platforms.evm import *
from manticore.core.smtlib import *
from manticore.core.smtlib.visitors import *
from manticore.utils import log
config.out_of_gas = 1

def printi(instruction):
    if False:
        i = 10
        return i + 15
    print(f'Instruction: {instruction}')
    print(f'\tdescription: {instruction.description}')
    print(f'\tgroup: {instruction.group}')
    print(f'\taddress: {instruction.offset}')
    print(f'\tsize: {instruction.size}')
    print(f'\thas_operand: {instruction.has_operand}')
    print(f'\toperand_size: {instruction.operand_size}')
    print(f'\toperand: {instruction.operand}')
    print(f'\tsemantics: {instruction.semantics}')
    print(f'\tpops: {instruction.pops}')
    print(f'\tpushes:', instruction.pushes)
    print(f'\tbytes: 0x{instruction.bytes.hex()}')
    print(f'\twrites to stack: {instruction.writes_to_stack}')
    print(f'\treads from stack: {instruction.reads_from_stack}')
    print(f'\twrites to memory: {instruction.writes_to_memory}')
    print(f'\treads from memory: {instruction.reads_from_memory}')
    print(f'\twrites to storage: {instruction.writes_to_storage}')
    print(f'\treads from storage: {instruction.reads_from_storage}')
    print(f'\tis terminator {instruction.is_terminator}')
constraints = ConstraintSet()
code = EVMAsm.assemble('\n    MSTORE\n')
data = constraints.new_array(index_bits=256, name='array')

class callbacks:
    initial_stack = []

    def will_execute_instruction(self, pc, instr):
        if False:
            for i in range(10):
                print('nop')
        for i in range(len(evm.stack), instr.pops):
            e = constraints.new_bitvec(256, name=f'stack_{len(self.initial_stack)}')
            self.initial_stack.append(e)
            evm.stack.insert(0, e)

class DummyWorld:

    def __init__(self, constraints):
        if False:
            while True:
                i = 10
        self.balances = constraints.new_array(index_bits=256, value_bits=256, name='balances')
        self.storage = constraints.new_array(index_bits=256, value_bits=256, name='storage')
        self.origin = constraints.new_bitvec(256, name='origin')
        self.price = constraints.new_bitvec(256, name='price')
        self.timestamp = constraints.new_bitvec(256, name='timestamp')
        self.coinbase = constraints.new_bitvec(256, name='coinbase')
        self.gaslimit = constraints.new_bitvec(256, name='gaslimit')
        self.difficulty = constraints.new_bitvec(256, name='difficulty')
        self.number = constraints.new_bitvec(256, name='number')

    def get_balance(self, address):
        if False:
            i = 10
            return i + 15
        return self.balances[address]

    def tx_origin(self):
        if False:
            i = 10
            return i + 15
        return self.origin

    def tx_gasprice(self):
        if False:
            i = 10
            return i + 15
        return self.price

    def block_coinbase(self):
        if False:
            print('Hello World!')
        return self.coinbase

    def block_timestamp(self):
        if False:
            for i in range(10):
                print('nop')
        return self.timestamp

    def block_number(self):
        if False:
            print('Hello World!')
        return self.number

    def block_difficulty(self):
        if False:
            print('Hello World!')
        return self.difficulty

    def block_gaslimit(self):
        if False:
            return 10
        return self.gaslimit

    def get_storage_data(self, address, offset):
        if False:
            return 10
        return self.storage[offset]

    def set_storage_data(self, address, offset, value):
        if False:
            print('Hello World!')
        self.storage[offset] = value

    def log(self, address, topics, memlog):
        if False:
            print('Hello World!')
        pass

    def send_funds(self, address, recipient, value):
        if False:
            while True:
                i = 10
        orig = self.balances[address] - value
        dest = self.balances[recipient] + value
        self.balances[address] = orig
        self.balances[recipient] = dest
caller = constraints.new_bitvec(256, name='caller')
value = constraints.new_bitvec(256, name='value')
world = DummyWorld(constraints)
callbacks = callbacks()
evm = EVM(constraints, 308176153570658872740176, data, caller, value, code, world=world, gas=1000000)
evm.subscribe('will_execute_instruction', callbacks.will_execute_instruction)
print('CODE:')
while not issymbolic(evm.pc):
    print(f'\t {evm.pc} {evm.instruction}')
    try:
        evm.execute()
    except EndTx as e:
        print(type(e))
        break
print(f'STORAGE = {translate_to_smtlib(world.storage)}')
print(f'MEM = {translate_to_smtlib(evm.memory)}')
for i in range(len(callbacks.initial_stack)):
    print(f'STACK[{i}] = {translate_to_smtlib(callbacks.initial_stack[i])}')
print('CONSTRAINTS:')
print(constraints)
print(f'PC: {translate_to_smtlib(evm.pc)} {solver.get_all_values(constraints, evm.pc, maxcnt=3, silent=True)}')