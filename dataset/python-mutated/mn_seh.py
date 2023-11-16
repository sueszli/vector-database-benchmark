from __future__ import print_function
import sys
from miasm.os_dep.win_api_x86_32_seh import fake_seh_handler, build_teb, set_win_fs_0, return_from_exception, EXCEPTION_PRIV_INSTRUCTION, return_from_seh, DEFAULT_SEH
from miasm.os_dep.win_32_structs import ContextException
from asm_test import Asm_Test_32
from pdb import pm

class Test_SEH(Asm_Test_32):
    """SEH Handling"""

    @staticmethod
    def deal_exception_priv(jitter):
        if False:
            i = 10
            return i + 15
        print('Exception Priv', hex(jitter.cpu.ESP))
        pc = fake_seh_handler(jitter, EXCEPTION_PRIV_INSTRUCTION)
        jitter.pc = pc
        jitter.cpu.EIP = pc
        return True

    def init_machine(self):
        if False:
            return 10
        super(Test_SEH, self).init_machine()
        set_win_fs_0(self.myjit)
        tib_ad = self.myjit.cpu.get_segm_base(self.myjit.cpu.FS)
        build_teb(self.myjit, tib_ad)
        self.myjit.add_exception_handler(1 << 17, Test_SEH.deal_exception_priv)
        self.myjit.add_breakpoint(return_from_exception, return_from_seh)

class Test_SEH_simple(Test_SEH):
    TXT = '\n    main:\n       XOR EAX, EAX\n       XOR EDX, EDX\n\n       PUSH handler\n       PUSH DWORD PTR FS:[EDX]\n       MOV DWORD PTR FS:[EDX], ESP\n\n       STI\n\n       MOV EBX, DWORD PTR [ESP]\n       MOV DWORD PTR FS:[EDX], EBX\n       ADD ESP, 0x8\n\n       RET\n\n    handler:\n       MOV ECX, DWORD PTR [ESP+0xC]\n       INC DWORD PTR [ECX+0x%08x]\n       MOV DWORD PTR [ECX+0x%08x], 0xcafebabe\n       XOR EAX, EAX\n       RET\n    ' % (ContextException.get_offset('eip'), ContextException.get_offset('eax'))

    def check(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.myjit.cpu.EAX == 3405691582
        assert self.myjit.cpu.EBX == DEFAULT_SEH

class Test_SEH_double(Test_SEH_simple):
    TXT = '\n    main:\n       XOR EAX, EAX\n       XOR EDX, EDX\n\n       PUSH handler1\n       PUSH DWORD PTR FS:[EDX]\n       MOV DWORD PTR FS:[EDX], ESP\n\n       PUSH handler2\n       PUSH DWORD PTR FS:[EDX]\n       MOV DWORD PTR FS:[EDX], ESP\n\n       STI\n\n       MOV EBX, DWORD PTR [ESP]\n       MOV DWORD PTR FS:[EDX], EBX\n       ADD ESP, 0x8\n\n       MOV EBX, DWORD PTR [ESP]\n       MOV DWORD PTR FS:[EDX], EBX\n       ADD ESP, 0x8\n\n       RET\n\n    handler1:\n       MOV EAX, 0x1\n       RET\n\n    handler2:\n       MOV ECX, DWORD PTR [ESP+0xC]\n       INC DWORD PTR [ECX+0x%08x]\n       MOV DWORD PTR [ECX+0x%08x], 0xcafebabe\n       XOR EAX, EAX\n       RET\n    ' % (ContextException.get_offset('eip'), ContextException.get_offset('eax'))
if __name__ == '__main__':
    [test(*sys.argv[1:])() for test in [Test_SEH_simple, Test_SEH_double]]