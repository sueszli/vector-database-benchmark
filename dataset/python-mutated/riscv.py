from __future__ import annotations
from capstone import *
from capstone.riscv import *
import pwndbg.gdblib.arch
import pwndbg.gdblib.regs

class DisassemblyAssistant(pwndbg.disasm.arch.DisassemblyAssistant):

    def __init__(self, architecture):
        if False:
            return 10
        super().__init__(architecture)
        self.architecture = architecture

    def _is_condition_taken(self, instruction):
        if False:
            while True:
                i = 10
        src1_unsigned = self.register(instruction, instruction.op_find(CS_OP_REG, 1))
        if instruction.op_count(CS_OP_REG) > 1:
            src2_unsigned = self.register(instruction, instruction.op_find(CS_OP_REG, 2))
        else:
            src2_unsigned = 0
        if self.architecture == 'rv32':
            src1_signed = src1_unsigned - ((src1_unsigned & 2147483648) << 1)
            src2_signed = src2_unsigned - ((src2_unsigned & 2147483648) << 1)
        elif self.architecture == 'rv64':
            src1_signed = src1_unsigned - ((src1_unsigned & 9223372036854775808) << 1)
            src2_signed = src2_unsigned - ((src2_unsigned & 9223372036854775808) << 1)
        else:
            raise NotImplementedError(f"architecture '{self.architecture}' not implemented")
        return {RISCV_INS_BEQ: src1_signed == src2_signed, RISCV_INS_BNE: src1_signed != src2_signed, RISCV_INS_BLT: src1_signed < src2_signed, RISCV_INS_BGE: src1_signed >= src2_signed, RISCV_INS_BLTU: src1_unsigned < src2_unsigned, RISCV_INS_BGEU: src1_unsigned >= src2_unsigned, RISCV_INS_C_BEQZ: src1_signed == 0, RISCV_INS_C_BNEZ: src1_signed != 0}.get(instruction.id, None)

    def condition(self, instruction):
        if False:
            for i in range(10):
                print('nop')
        'Checks if the current instruction is a jump that is taken.\n        Returns None if the instruction is executed unconditionally,\n        True if the instruction is executed for sure, False otherwise.\n        '
        if RISCV_GRP_CALL in instruction.groups:
            return None
        if instruction.address != pwndbg.gdblib.regs.pc:
            return False
        if RISCV_GRP_BRANCH_RELATIVE in instruction.groups:
            return self._is_condition_taken(instruction)
        return None

    def next(self, instruction, call=False):
        if False:
            for i in range(10):
                print('nop')
        'Return the address of the jump / conditional jump,\n        None if the next address is not dependent on instruction.\n        '
        ptrmask = pwndbg.gdblib.arch.ptrmask
        if instruction.id in [RISCV_INS_JAL, RISCV_INS_C_JAL]:
            return instruction.address + instruction.op_find(CS_OP_IMM, 1).imm & ptrmask
        if instruction.address != pwndbg.gdblib.regs.pc:
            return None
        if RISCV_GRP_BRANCH_RELATIVE in instruction.groups and self._is_condition_taken(instruction):
            return instruction.address + instruction.op_find(CS_OP_IMM, 1).imm & ptrmask
        if instruction.id in [RISCV_INS_JALR, RISCV_INS_C_JALR]:
            target = self.register(instruction, instruction.op_find(CS_OP_REG, 1)) + instruction.op_find(CS_OP_IMM, 1).imm & ptrmask
            return target ^ target & 1
        return super().next(instruction, call)
assistant_rv32 = DisassemblyAssistant('rv32')
assistant_rv64 = DisassemblyAssistant('rv64')