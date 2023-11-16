import pyevmasm as ea
from binascii import hexlify

def printi(instruction):
    if False:
        return 10
    print(f'Instruction: {instruction}')
    print('\tdescription:', instruction.description)
    print('\tgroup:', instruction.group)
    print('\taddress:', instruction.pc)
    print('\tsize:', instruction.size)
    print('\thas_operand:', instruction.has_operand)
    print('\toperand_size:', instruction.operand_size)
    print('\toperand:', instruction.operand)
    print('\tsemantics:', instruction.semantics)
    print('\tpops:', instruction.pops)
    print('\tpushes:', instruction.pushes)
    print(f'\tbytes: 0x' + hexlify(instruction.bytes).decode())
    print('\twrites to stack:', instruction.writes_to_stack)
    print('\treads from stack:', instruction.reads_from_stack)
    print('\twrites to memory:', instruction.writes_to_memory)
    print('\treads from memory:', instruction.reads_from_memory)
    print('\twrites to storage:', instruction.writes_to_storage)
    print('\treads from storage:', instruction.reads_from_storage)
    print('\tis terminator', instruction.is_terminator)
instruction = ea.disassemble_one('`\x10')
printi(instruction)
instruction = ea.assemble_one('PUSH1 0x10')
printi(instruction)
for instruction in ea.disassemble_all('01'):
    printi(instruction)
for instruction in ea.assemble_all('ADDRESS\nBALANCE'):
    printi(instruction)
print(ea.assemble_hex('PUSH1 0x60\n                           BLOCKHASH\n                           MSTORE\n                           PUSH1 0x2\n                           PUSH2 0x100\n                        '))
print(ea.disassemble_hex('0x606040526002610100'))