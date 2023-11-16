import sys
from asm_test import Asm_Test_32

class Test_SCAS(Asm_Test_32):
    MYSTRING = 'test string'
    TXT = '\n    main:\n       LEA EDI, DWORD PTR [mystr]\n       XOR  ECX, ECX\n       DEC  ECX\n       REPNE SCASB\n       NOT ECX\n       DEC ECX\n       RET\n\n    mystr:\n    .string "%s"\n    ' % MYSTRING

    def check(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.myjit.cpu.ECX == len(self.MYSTRING)
        mystr = self.myjit.lifter.loc_db.get_name_location('mystr')
        assert self.myjit.cpu.EDI == self.myjit.lifter.loc_db.get_location_offset(mystr) + len(self.MYSTRING) + 1

class Test_MOVS(Asm_Test_32):
    MYSTRING = 'test string'
    TXT = '\n    main:\n       LEA ESI, DWORD PTR [mystr]\n       LEA EDI, DWORD PTR [buffer]\n       MOV ECX, %d\n       REPE  MOVSB\n       RET\n\n    mystr:\n    .string "%s"\n    buffer:\n    .string "%s"\n    ' % (len(MYSTRING), MYSTRING, ' ' * len(MYSTRING))

    def check(self):
        if False:
            print('Hello World!')
        assert self.myjit.cpu.ECX == 0
        buffer = self.myjit.lifter.loc_db.get_name_location('buffer')
        assert self.myjit.cpu.EDI == self.myjit.lifter.loc_db.get_location_offset(buffer) + len(self.MYSTRING)
        mystr = self.myjit.lifter.loc_db.get_name_location('mystr')
        assert self.myjit.cpu.ESI == self.myjit.lifter.loc_db.get_location_offset(mystr) + len(self.MYSTRING)
if __name__ == '__main__':
    [test(*sys.argv[1:])() for test in [Test_SCAS, Test_MOVS]]