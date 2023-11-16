from abc import abstractmethod
import capstone as cs

class Instruction:
    """Capstone-like instruction to be used internally"""

    @property
    @abstractmethod
    def address(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        pass

    @property
    @abstractmethod
    def mnemonic(self) -> str:
        if False:
            print('Hello World!')
        pass

    @property
    @abstractmethod
    def op_str(self) -> str:
        if False:
            while True:
                i = 10
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        if False:
            while True:
                i = 10
        pass

    @property
    @abstractmethod
    def operands(self):
        if False:
            i = 10
            return i + 15
        pass

    @property
    @abstractmethod
    def insn_name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        if False:
            print('Hello World!')
        pass

class Disasm:
    """Abstract class for different disassembler interfaces"""

    def __init__(self, disasm):
        if False:
            return 10
        self.disasm = disasm

    @abstractmethod
    def disassemble_instruction(self, code, pc) -> Instruction:
        if False:
            for i in range(10):
                print('nop')
        'Get next instruction based on the disassembler in use\n\n        :param str code: binary blob to be disassembled\n        :param long pc: program counter\n        '

class CapstoneDisasm(Disasm):

    def __init__(self, arch, mode):
        if False:
            print('Hello World!')
        try:
            cap = cs.Cs(arch, mode)
        except Exception as e:
            raise e
        cap.detail = True
        cap.syntax = 0
        super().__init__(cap)

    def disassemble_instruction(self, code: bytes, pc: int) -> Instruction:
        if False:
            print('Hello World!')
        'Get next instruction using the Capstone disassembler\n\n        :param str code: binary blob to be disassembled\n        :param long pc: program counter\n        '
        return next(self.disasm.disasm(code, pc))

def init_disassembler(disassembler, arch, mode, view=None):
    if False:
        i = 10
        return i + 15
    if disassembler == 'capstone':
        return CapstoneDisasm(arch, mode)
    else:
        raise NotImplementedError('Disassembler not implemented')