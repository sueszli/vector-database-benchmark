import unittest
from manticore.native.cpu.abstractcpu import ConcretizeRegister
from manticore.native.cpu.x86 import AMD64Cpu
from manticore.native.memory import *
from manticore.core.smtlib.solver import Z3Solver
solver = Z3Solver.instance()

class CPUTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    class ROOperand:
        """Mocking class for operand ronly"""

        def __init__(self, size, value):
            if False:
                for i in range(10):
                    print('nop')
            self.size = size
            self.value = value

        def read(self):
            if False:
                while True:
                    i = 10
            return self.value & (1 << self.size) - 1

    class RWOperand(ROOperand):
        """Mocking class for operand rw"""

        def write(self, value):
            if False:
                for i in range(10):
                    print('nop')
            self.value = value & (1 << self.size) - 1
            return self.value

    def test_PUNPCKHDQ_1(self):
        if False:
            return 10
        'Instruction PUNPCKHDQ_1\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = 6179796677514570882216181629000
        cpu.XMM8 = 0
        cpu.RIP = 4299843
        cpu.execute()
        self.assertEqual(mem[4299843:4299848], [b'f', b'A', b'\x0f', b'j', b'\xc0'])
        self.assertEqual(cpu.XMM0, 1438846037749345026124)
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.RIP, 4299848)

    def test_PUNPCKHDQ_10(self):
        if False:
            while True:
                i = 10
        'Instruction PUNPCKHDQ_10\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = 4278320776729504922099018367024
        cpu.XMM8 = 0
        cpu.RIP = 4299843
        cpu.execute()
        self.assertEqual(mem[4299843:4299848], [b'f', b'A', b'\x0f', b'j', b'\xc0'])
        self.assertEqual(cpu.XMM0, 996124179980315787316)
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.RIP, 4299848)

    def test_PUNPCKHDQ_11(self):
        if False:
            while True:
                i = 10
        'Instruction PUNPCKHDQ_11\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = 4912146076991193575471406121016
        cpu.XMM8 = 0
        cpu.RIP = 4299843
        cpu.execute()
        self.assertEqual(mem[4299843:4299848], [b'f', b'A', b'\x0f', b'j', b'\xc0'])
        self.assertEqual(cpu.XMM0, 1143698132569992200252)
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.RIP, 4299848)

    def test_PUNPCKHDQ_12(self):
        if False:
            print('Hello World!')
        'Instruction PUNPCKHDQ_12\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = 11250399079608080109195283660936
        cpu.XMM8 = 0
        cpu.RIP = 4299843
        cpu.execute()
        self.assertEqual(mem[4299843:4299848], [b'f', b'A', b'\x0f', b'j', b'\xc0'])
        self.assertEqual(cpu.XMM0, 2619437658466756329612)
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.RIP, 4299848)

    def test_PUNPCKHDQ_13(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKHDQ_13\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = 18222477382486655296291548954848
        cpu.XMM8 = 0
        cpu.RIP = 4299843
        cpu.execute()
        self.assertEqual(mem[4299843:4299848], [b'f', b'A', b'\x0f', b'j', b'\xc0'])
        self.assertEqual(cpu.XMM0, 4242751136953196871908)
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.RIP, 4299848)

    def test_PUNPCKHDQ_14(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKHDQ_14\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = 9982748479084702802450508152952
        cpu.XMM8 = 0
        cpu.RIP = 4299843
        cpu.execute()
        self.assertEqual(mem[4299843:4299848], [b'f', b'A', b'\x0f', b'j', b'\xc0'])
        self.assertEqual(cpu.XMM0, 2324289753287403503740)
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.RIP, 4299848)

    def test_PUNPCKHDQ_15(self):
        if False:
            i = 10
            return i + 15
        'Instruction PUNPCKHDQ_15\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = 11884224379869768762567671414928
        cpu.XMM8 = 0
        cpu.RIP = 4299843
        cpu.execute()
        self.assertEqual(mem[4299843:4299848], [b'f', b'A', b'\x0f', b'j', b'\xc0'])
        self.assertEqual(cpu.XMM0, 2767011611056432742548)
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.RIP, 4299848)

    def test_PUNPCKHDQ_16(self):
        if False:
            return 10
        'Instruction PUNPCKHDQ_16\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = 475368975159373001864691843072
        cpu.XMM8 = 0
        cpu.RIP = 4299843
        cpu.execute()
        self.assertEqual(mem[4299843:4299848], [b'f', b'A', b'\x0f', b'j', b'\xc0'])
        self.assertEqual(cpu.XMM0, 110680464442257309700)
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.RIP, 4299848)

    def test_PUNPCKHDQ_17(self):
        if False:
            return 10
        'Instruction PUNPCKHDQ_17\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = 16321001481701589336174385692872
        cpu.XMM8 = 0
        cpu.RIP = 4299843
        cpu.execute()
        self.assertEqual(mem[4299843:4299848], [b'f', b'A', b'\x0f', b'j', b'\xc0'])
        self.assertEqual(cpu.XMM0, 3800029279184167633100)
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.RIP, 4299848)

    def test_PUNPCKHDQ_18(self):
        if False:
            print('Hello World!')
        'Instruction PUNPCKHDQ_18\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = 12518049680131457415940059168920
        cpu.XMM8 = 0
        cpu.RIP = 4299843
        cpu.execute()
        self.assertEqual(mem[4299843:4299848], [b'f', b'A', b'\x0f', b'j', b'\xc0'])
        self.assertEqual(cpu.XMM0, 2914585563646109155484)
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.RIP, 4299848)

    def test_PUNPCKHDQ_19(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKHDQ_19\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = 5545971377252882228843793875008
        cpu.XMM8 = 0
        cpu.RIP = 4299843
        cpu.execute()
        self.assertEqual(mem[4299843:4299848], [b'f', b'A', b'\x0f', b'j', b'\xc0'])
        self.assertEqual(cpu.XMM0, 1291272085159668613188)
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.RIP, 4299848)

    def test_PUNPCKHDQ_2(self):
        if False:
            print('Hello World!')
        'Instruction PUNPCKHDQ_2\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = 15053350881178212029429610184888
        cpu.XMM8 = 0
        cpu.RIP = 4299843
        cpu.execute()
        self.assertEqual(mem[4299843:4299848], [b'f', b'A', b'\x0f', b'j', b'\xc0'])
        self.assertEqual(cpu.XMM0, 3504881374004814807228)
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.RIP, 4299848)

    def test_PUNPCKHDQ_20(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKHDQ_20\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = 8081272578299636842333344890976
        cpu.XMM8 = 0
        cpu.RIP = 4299843
        cpu.execute()
        self.assertEqual(mem[4299843:4299848], [b'f', b'A', b'\x0f', b'j', b'\xc0'])
        self.assertEqual(cpu.XMM0, 1881567895518374264932)
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.RIP, 4299848)

    def test_PUNPCKHDQ_21(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKHDQ_21\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = 7447447278037948188960957136984
        cpu.XMM8 = 0
        cpu.RIP = 4299843
        cpu.execute()
        self.assertEqual(mem[4299843:4299848], [b'f', b'A', b'\x0f', b'j', b'\xc0'])
        self.assertEqual(cpu.XMM0, 1733993942928697851996)
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.RIP, 4299848)

    def test_PUNPCKHDQ_3(self):
        if False:
            i = 10
            return i + 15
        'Instruction PUNPCKHDQ_3\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = 8715097878561325495705732644968
        cpu.XMM8 = 0
        cpu.RIP = 4299843
        cpu.execute()
        self.assertEqual(mem[4299843:4299848], [b'f', b'A', b'\x0f', b'j', b'\xc0'])
        self.assertEqual(cpu.XMM0, 2029141848108050677868)
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.RIP, 4299848)

    def test_PUNPCKHDQ_4(self):
        if False:
            while True:
                i = 10
        'Instruction PUNPCKHDQ_4\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = 15687176181439900682801997938880
        cpu.XMM8 = 0
        cpu.RIP = 4299843
        cpu.execute()
        self.assertEqual(mem[4299843:4299848], [b'f', b'A', b'\x0f', b'j', b'\xc0'])
        self.assertEqual(cpu.XMM0, 3652455326594491220164)
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.RIP, 4299848)

    def test_PUNPCKHDQ_5(self):
        if False:
            print('Hello World!')
        'Instruction PUNPCKHDQ_5\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = 14419525580916523376057222430896
        cpu.XMM8 = 0
        cpu.RIP = 4299843
        cpu.execute()
        self.assertEqual(mem[4299843:4299848], [b'f', b'A', b'\x0f', b'j', b'\xc0'])
        self.assertEqual(cpu.XMM0, 3357307421415138394292)
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.RIP, 4299848)

    def test_PUNPCKHDQ_6(self):
        if False:
            print('Hello World!')
        'Instruction PUNPCKHDQ_6\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = 13785700280654834722684834676904
        cpu.XMM8 = 0
        cpu.RIP = 4299843
        cpu.execute()
        self.assertEqual(mem[4299843:4299848], [b'f', b'A', b'\x0f', b'j', b'\xc0'])
        self.assertEqual(cpu.XMM0, 3209733468825461981356)
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.RIP, 4299848)

    def test_PUNPCKHDQ_7(self):
        if False:
            print('Hello World!')
        'Instruction PUNPCKHDQ_7\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = 1109194275421061655237079597064
        cpu.XMM8 = 0
        cpu.RIP = 4299843
        cpu.execute()
        self.assertEqual(mem[4299843:4299848], [b'f', b'A', b'\x0f', b'j', b'\xc0'])
        self.assertEqual(cpu.XMM0, 258254417031933722636)
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.RIP, 4299848)

    def test_PUNPCKHDQ_8(self):
        if False:
            i = 10
            return i + 15
        'Instruction PUNPCKHDQ_8\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = 10616573779346391455822895906944
        cpu.XMM8 = 0
        cpu.RIP = 4299843
        cpu.execute()
        self.assertEqual(mem[4299843:4299848], [b'f', b'A', b'\x0f', b'j', b'\xc0'])
        self.assertEqual(cpu.XMM0, 2471863705877079916676)
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.RIP, 4299848)

    def test_PUNPCKHDQ_9(self):
        if False:
            print('Hello World!')
        'Instruction PUNPCKHDQ_9\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = 16954826781963277989546773446864
        cpu.XMM8 = 0
        cpu.RIP = 4299843
        cpu.execute()
        self.assertEqual(mem[4299843:4299848], [b'f', b'A', b'\x0f', b'j', b'\xc0'])
        self.assertEqual(cpu.XMM0, 3947603231773844046036)
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.RIP, 4299848)

    def test_PUNPCKHQDQ_1(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKHQDQ_1\n        Groups: sse2\n        0x419c71:   punpckhqdq      xmm1, xmm1\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299889, 'f\x0fmÉ')
        cpu.XMM1 = 131446628328818805501115096
        cpu.RIP = 4299889
        cpu.execute()
        self.assertEqual(mem[4299889:4299893], [b'f', b'\x0f', b'm', b'\xc9'])
        self.assertEqual(cpu.XMM1, 131446628328818805501115112)
        self.assertEqual(cpu.RIP, 4299893)

    def test_PUNPCKHQDQ_10(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKHQDQ_10\n        Groups: sse2\n        0x419c71:   punpckhqdq      xmm1, xmm1\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299889, 'f\x0fmÉ')
        cpu.XMM1 = 131437183595853066210687192
        cpu.RIP = 4299889
        cpu.execute()
        self.assertEqual(mem[4299889:4299893], [b'f', b'\x0f', b'm', b'\xc9'])
        self.assertEqual(cpu.XMM1, 131437183595853066210687208)
        self.assertEqual(cpu.RIP, 4299893)

    def test_PUNPCKHQDQ_11(self):
        if False:
            i = 10
            return i + 15
        'Instruction PUNPCKHQDQ_11\n        Groups: sse2\n        0x419c86:   punpckhqdq      xmm0, xmm0\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299910, 'f\x0fmÀ')
        cpu.XMM0 = 131444857441387729384159864
        cpu.RIP = 4299910
        cpu.execute()
        self.assertEqual(mem[4299910:4299914], [b'f', b'\x0f', b'm', b'\xc0'])
        self.assertEqual(cpu.XMM0, 131444857441387729384159880)
        self.assertEqual(cpu.RIP, 4299914)

    def test_PUNPCKHQDQ_12(self):
        if False:
            i = 10
            return i + 15
        'Instruction PUNPCKHQDQ_12\n        Groups: sse2\n        0x419c71:   punpckhqdq      xmm1, xmm1\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299889, 'f\x0fmÉ')
        cpu.XMM1 = 131445447737198088089811608
        cpu.RIP = 4299889
        cpu.execute()
        self.assertEqual(mem[4299889:4299893], [b'f', b'\x0f', b'm', b'\xc9'])
        self.assertEqual(cpu.XMM1, 131445447737198088089811624)
        self.assertEqual(cpu.RIP, 4299893)

    def test_PUNPCKHQDQ_13(self):
        if False:
            print('Hello World!')
        'Instruction PUNPCKHQDQ_13\n        Groups: sse2\n        0x419c71:   punpckhqdq      xmm1, xmm1\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299889, 'f\x0fmÉ')
        cpu.XMM1 = 131465517794750284081970904
        cpu.RIP = 4299889
        cpu.execute()
        self.assertEqual(mem[4299889:4299893], [b'f', b'\x0f', b'm', b'\xc9'])
        self.assertEqual(cpu.XMM1, 131465517794750284081970920)
        self.assertEqual(cpu.RIP, 4299893)

    def test_PUNPCKHQDQ_14(self):
        if False:
            i = 10
            return i + 15
        'Instruction PUNPCKHQDQ_14\n        Groups: sse2\n        0x419c86:   punpckhqdq      xmm0, xmm0\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299910, 'f\x0fmÀ')
        cpu.XMM0 = 131467288682181360198926136
        cpu.RIP = 4299910
        cpu.execute()
        self.assertEqual(mem[4299910:4299914], [b'f', b'\x0f', b'm', b'\xc0'])
        self.assertEqual(cpu.XMM0, 131467288682181360198926152)
        self.assertEqual(cpu.RIP, 4299914)

    def test_PUNPCKHQDQ_15(self):
        if False:
            print('Hello World!')
        'Instruction PUNPCKHQDQ_15\n        Groups: sse2\n        0x419c86:   punpckhqdq      xmm0, xmm0\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299910, 'f\x0fmÀ')
        cpu.XMM0 = 131460205132457055731105208
        cpu.RIP = 4299910
        cpu.execute()
        self.assertEqual(mem[4299910:4299914], [b'f', b'\x0f', b'm', b'\xc0'])
        self.assertEqual(cpu.XMM0, 131460205132457055731105224)
        self.assertEqual(cpu.RIP, 4299914)

    def test_PUNPCKHQDQ_16(self):
        if False:
            i = 10
            return i + 15
        'Instruction PUNPCKHQDQ_16\n        Groups: sse2\n        0x419c71:   punpckhqdq      xmm1, xmm1\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299889, 'f\x0fmÉ')
        cpu.XMM1 = 131439544779094501033294168
        cpu.RIP = 4299889
        cpu.execute()
        self.assertEqual(mem[4299889:4299893], [b'f', b'\x0f', b'm', b'\xc9'])
        self.assertEqual(cpu.XMM1, 131439544779094501033294184)
        self.assertEqual(cpu.RIP, 4299893)

    def test_PUNPCKHQDQ_17(self):
        if False:
            return 10
        'Instruction PUNPCKHQDQ_17\n        Groups: sse2\n        0x419c86:   punpckhqdq      xmm0, xmm0\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299910, 'f\x0fmÀ')
        cpu.XMM0 = 131448399216249881618070328
        cpu.RIP = 4299910
        cpu.execute()
        self.assertEqual(mem[4299910:4299914], [b'f', b'\x0f', b'm', b'\xc0'])
        self.assertEqual(cpu.XMM0, 131448399216249881618070344)
        self.assertEqual(cpu.RIP, 4299914)

    def test_PUNPCKHQDQ_18(self):
        if False:
            i = 10
            return i + 15
        'Instruction PUNPCKHQDQ_18\n        Groups: sse2\n        0x419c86:   punpckhqdq      xmm0, xmm0\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299910, 'f\x0fmÀ')
        cpu.XMM0 = 131437773891663424916338936
        cpu.RIP = 4299910
        cpu.execute()
        self.assertEqual(mem[4299910:4299914], [b'f', b'\x0f', b'm', b'\xc0'])
        self.assertEqual(cpu.XMM0, 131437773891663424916338952)
        self.assertEqual(cpu.RIP, 4299914)

    def test_PUNPCKHQDQ_19(self):
        if False:
            while True:
                i = 10
        'Instruction PUNPCKHQDQ_19\n        Groups: sse2\n        0x419c71:   punpckhqdq      xmm1, xmm1\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299889, 'f\x0fmÉ')
        cpu.XMM1 = 131470240161233153727184856
        cpu.RIP = 4299889
        cpu.execute()
        self.assertEqual(mem[4299889:4299893], [b'f', b'\x0f', b'm', b'\xc9'])
        self.assertEqual(cpu.XMM1, 131470240161233153727184872)
        self.assertEqual(cpu.RIP, 4299893)

    def test_PUNPCKHQDQ_2(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKHQDQ_2\n        Groups: sse2\n        0x419c86:   punpckhqdq      xmm0, xmm0\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299910, 'f\x0fmÀ')
        cpu.XMM0 = 131435412708421990093731960
        cpu.RIP = 4299910
        cpu.execute()
        self.assertEqual(mem[4299910:4299914], [b'f', b'\x0f', b'm', b'\xc0'])
        self.assertEqual(cpu.XMM0, 131435412708421990093731976)
        self.assertEqual(cpu.RIP, 4299914)

    def test_PUNPCKHQDQ_20(self):
        if False:
            while True:
                i = 10
        'Instruction PUNPCKHQDQ_20\n        Groups: sse2\n        0x419c71:   punpckhqdq      xmm1, xmm1\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299889, 'f\x0fmÉ')
        cpu.XMM1 = 131438364187473783621990680
        cpu.RIP = 4299889
        cpu.execute()
        self.assertEqual(mem[4299889:4299893], [b'f', b'\x0f', b'm', b'\xc9'])
        self.assertEqual(cpu.XMM1, 131438364187473783621990696)
        self.assertEqual(cpu.RIP, 4299893)

    def test_PUNPCKHQDQ_21(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKHQDQ_21\n        Groups: sse2\n        0x419c86:   punpckhqdq      xmm0, xmm0\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299910, 'f\x0fmÀ')
        cpu.XMM0 = 131446038033008446795463352
        cpu.RIP = 4299910
        cpu.execute()
        self.assertEqual(mem[4299910:4299914], [b'f', b'\x0f', b'm', b'\xc0'])
        self.assertEqual(cpu.XMM0, 131446038033008446795463368)
        self.assertEqual(cpu.RIP, 4299914)

    def test_PUNPCKHQDQ_3(self):
        if False:
            i = 10
            return i + 15
        'Instruction PUNPCKHQDQ_3\n        Groups: sse2\n        0x419c71:   punpckhqdq      xmm1, xmm1\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299889, 'f\x0fmÉ')
        cpu.XMM1 = 131451350695301675146329048
        cpu.RIP = 4299889
        cpu.execute()
        self.assertEqual(mem[4299889:4299893], [b'f', b'\x0f', b'm', b'\xc9'])
        self.assertEqual(cpu.XMM1, 131451350695301675146329064)
        self.assertEqual(cpu.RIP, 4299893)

    def test_PUNPCKHQDQ_4(self):
        if False:
            return 10
        'Instruction PUNPCKHQDQ_4\n        Groups: sse2\n        0x419c86:   punpckhqdq      xmm0, xmm0\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299910, 'f\x0fmÀ')
        cpu.XMM0 = 131459024540836338319801720
        cpu.RIP = 4299910
        cpu.execute()
        self.assertEqual(mem[4299910:4299914], [b'f', b'\x0f', b'm', b'\xc0'])
        self.assertEqual(cpu.XMM0, 131459024540836338319801736)
        self.assertEqual(cpu.RIP, 4299914)

    def test_PUNPCKHQDQ_5(self):
        if False:
            i = 10
            return i + 15
        'Instruction PUNPCKHQDQ_5\n        Groups: sse2\n        0x419c71:   punpckhqdq      xmm1, xmm1\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299889, 'f\x0fmÉ')
        cpu.XMM1 = 131434822412611631388080216
        cpu.RIP = 4299889
        cpu.execute()
        self.assertEqual(mem[4299889:4299893], [b'f', b'\x0f', b'm', b'\xc9'])
        self.assertEqual(cpu.XMM1, 131434822412611631388080232)
        self.assertEqual(cpu.RIP, 4299893)

    def test_PUNPCKHQDQ_6(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKHQDQ_6\n        Groups: sse2\n        0x419c71:   punpckhqdq      xmm1, xmm1\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299889, 'f\x0fmÉ')
        cpu.XMM1 = 131460795428267414436756952
        cpu.RIP = 4299889
        cpu.execute()
        self.assertEqual(mem[4299889:4299893], [b'f', b'\x0f', b'm', b'\xc9'])
        self.assertEqual(cpu.XMM1, 131460795428267414436756968)
        self.assertEqual(cpu.RIP, 4299893)

    def test_PUNPCKHQDQ_7(self):
        if False:
            return 10
        'Instruction PUNPCKHQDQ_7\n        Groups: sse2\n        0x419c71:   punpckhqdq      xmm1, xmm1\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299889, 'f\x0fmÉ')
        cpu.XMM1 = 131457253653405262202846488
        cpu.RIP = 4299889
        cpu.execute()
        self.assertEqual(mem[4299889:4299893], [b'f', b'\x0f', b'm', b'\xc9'])
        self.assertEqual(cpu.XMM1, 131457253653405262202846504)
        self.assertEqual(cpu.RIP, 4299893)

    def test_PUNPCKHQDQ_8(self):
        if False:
            print('Hello World!')
        'Instruction PUNPCKHQDQ_8\n        Groups: sse2\n        0x419c71:   punpckhqdq      xmm1, xmm1\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299889, 'f\x0fmÉ')
        cpu.XMM1 = 131436003004232348799383704
        cpu.RIP = 4299889
        cpu.execute()
        self.assertEqual(mem[4299889:4299893], [b'f', b'\x0f', b'm', b'\xc9'])
        self.assertEqual(cpu.XMM1, 131436003004232348799383720)
        self.assertEqual(cpu.RIP, 4299893)

    def test_PUNPCKHQDQ_9(self):
        if False:
            return 10
        'Instruction PUNPCKHQDQ_9\n        Groups: sse2\n        0x419c86:   punpckhqdq      xmm0, xmm0\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299910, 'f\x0fmÀ')
        cpu.XMM0 = 131438954483284142327642424
        cpu.RIP = 4299910
        cpu.execute()
        self.assertEqual(mem[4299910:4299914], [b'f', b'\x0f', b'm', b'\xc0'])
        self.assertEqual(cpu.XMM0, 131438954483284142327642440)
        self.assertEqual(cpu.RIP, 4299914)

    def test_PUNPCKLBW_1(self):
        if False:
            i = 10
            return i + 15
        'Instruction PUNPCKLBW_1\n        Groups: sse2\n        0x4668ac:   punpcklbw       xmm1, xmm1\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4612096, 4096, 'rwx')
        mem.write(4614316, 'f\x0f`É')
        cpu.XMM1 = 47
        cpu.RIP = 4614316
        cpu.execute()
        self.assertEqual(mem[4614316:4614320], [b'f', b'\x0f', b'`', b'\xc9'])
        self.assertEqual(cpu.XMM1, 12079)
        self.assertEqual(cpu.RIP, 4614320)

    def test_PUNPCKLDQ_1(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKLDQ_1\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = 0
        cpu.XMM1 = 13151874980393146069312446922912
        cpu.RIP = 4299848
        cpu.execute()
        self.assertEqual(mem[4299848:4299853], [b'f', b'A', b'\x0f', b'b', b'\xc8'])
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.XMM1, 2988372539940947361952)
        self.assertEqual(cpu.RIP, 4299853)

    def test_PUNPCKLDQ_10(self):
        if False:
            return 10
        'Instruction PUNPCKLDQ_10\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = 0
        cpu.XMM1 = 4912146076991193575471406121016
        cpu.RIP = 4299848
        cpu.execute()
        self.assertEqual(mem[4299848:4299853], [b'f', b'A', b'\x0f', b'b', b'\xc8'])
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.XMM1, 1069911156275153993784)
        self.assertEqual(cpu.RIP, 4299853)

    def test_PUNPCKLDQ_11(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKLDQ_11\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = 0
        cpu.XMM1 = 475368975159373001864691843072
        cpu.RIP = 4299848
        cpu.execute()
        self.assertEqual(mem[4299848:4299853], [b'f', b'A', b'\x0f', b'b', b'\xc8'])
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.XMM1, 36893488147419103232)
        self.assertEqual(cpu.RIP, 4299853)

    def test_PUNPCKLDQ_12(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKLDQ_12\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = 0
        cpu.XMM1 = 2376844875944438961981855105048
        cpu.RIP = 4299848
        cpu.execute()
        self.assertEqual(mem[4299848:4299853], [b'f', b'A', b'\x0f', b'b', b'\xc8'])
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.XMM1, 479615345916448342040)
        self.assertEqual(cpu.RIP, 4299853)

    def test_PUNPCKLDQ_13(self):
        if False:
            return 10
        'Instruction PUNPCKLDQ_13\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = 0
        cpu.XMM1 = 1109194275421061655237079597064
        cpu.RIP = 4299848
        cpu.execute()
        self.assertEqual(mem[4299848:4299853], [b'f', b'A', b'\x0f', b'b', b'\xc8'])
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.XMM1, 184467440737095516168)
        self.assertEqual(cpu.RIP, 4299853)

    def test_PUNPCKLDQ_14(self):
        if False:
            print('Hello World!')
        'Instruction PUNPCKLDQ_14\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = 0
        cpu.XMM1 = 18856302682748343949663936708840
        cpu.RIP = 4299848
        cpu.execute()
        self.assertEqual(mem[4299848:4299853], [b'f', b'A', b'\x0f', b'b', b'\xc8'])
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.XMM1, 4316538113248035078376)
        self.assertEqual(cpu.RIP, 4299853)

    def test_PUNPCKLDQ_15(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKLDQ_15\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = 0
        cpu.XMM1 = 3644495476467816268726630613032
        cpu.RIP = 4299848
        cpu.execute()
        self.assertEqual(mem[4299848:4299853], [b'f', b'A', b'\x0f', b'b', b'\xc8'])
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.XMM1, 774763251095801167912)
        self.assertEqual(cpu.RIP, 4299853)

    def test_PUNPCKLDQ_16(self):
        if False:
            i = 10
            return i + 15
        'Instruction PUNPCKLDQ_16\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = 0
        cpu.XMM1 = 19490127983010032603036324462832
        cpu.RIP = 4299848
        cpu.execute()
        self.assertEqual(mem[4299848:4299853], [b'f', b'A', b'\x0f', b'b', b'\xc8'])
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.XMM1, 4464112065837711491312)
        self.assertEqual(cpu.RIP, 4299853)

    def test_PUNPCKLDQ_17(self):
        if False:
            while True:
                i = 10
        'Instruction PUNPCKLDQ_17\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = 0
        cpu.XMM1 = 12518049680131457415940059168920
        cpu.RIP = 4299848
        cpu.execute()
        self.assertEqual(mem[4299848:4299853], [b'f', b'A', b'\x0f', b'b', b'\xc8'])
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.XMM1, 2840798587351270949016)
        self.assertEqual(cpu.RIP, 4299853)

    def test_PUNPCKLDQ_18(self):
        if False:
            return 10
        'Instruction PUNPCKLDQ_18\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = 0
        cpu.XMM1 = 1743019575682750308609467351056
        cpu.RIP = 4299848
        cpu.execute()
        self.assertEqual(mem[4299848:4299853], [b'f', b'A', b'\x0f', b'b', b'\xc8'])
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.XMM1, 332041393326771929104)
        self.assertEqual(cpu.RIP, 4299853)

    def test_PUNPCKLDQ_19(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKLDQ_19\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = 0
        cpu.XMM1 = 11884224379869768762567671414928
        cpu.RIP = 4299848
        cpu.execute()
        self.assertEqual(mem[4299848:4299853], [b'f', b'A', b'\x0f', b'b', b'\xc8'])
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.XMM1, 2693224634761594536080)
        self.assertEqual(cpu.RIP, 4299853)

    def test_PUNPCKLDQ_2(self):
        if False:
            return 10
        'Instruction PUNPCKLDQ_2\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = 0
        cpu.XMM1 = 9348923178823014149078120398960
        cpu.RIP = 4299848
        cpu.execute()
        self.assertEqual(mem[4299848:4299853], [b'f', b'A', b'\x0f', b'b', b'\xc8'])
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.XMM1, 2102928824402888884336)
        self.assertEqual(cpu.RIP, 4299853)

    def test_PUNPCKLDQ_20(self):
        if False:
            print('Hello World!')
        'Instruction PUNPCKLDQ_20\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = 0
        cpu.XMM1 = 17588652082224966642919161200856
        cpu.RIP = 4299848
        cpu.execute()
        self.assertEqual(mem[4299848:4299853], [b'f', b'A', b'\x0f', b'b', b'\xc8'])
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.XMM1, 4021390208068682252504)
        self.assertEqual(cpu.RIP, 4299853)

    def test_PUNPCKLDQ_21(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKLDQ_21\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = 0
        cpu.XMM1 = 18222477382486655296291548954848
        cpu.RIP = 4299848
        cpu.execute()
        self.assertEqual(mem[4299848:4299853], [b'f', b'A', b'\x0f', b'b', b'\xc8'])
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.XMM1, 4168964160658358665440)
        self.assertEqual(cpu.RIP, 4299853)

    def test_PUNPCKLDQ_3(self):
        if False:
            i = 10
            return i + 15
        'Instruction PUNPCKLDQ_3\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = 0
        cpu.XMM1 = 4278320776729504922099018367024
        cpu.RIP = 4299848
        cpu.execute()
        self.assertEqual(mem[4299848:4299853], [b'f', b'A', b'\x0f', b'b', b'\xc8'])
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.XMM1, 922337203685477580848)
        self.assertEqual(cpu.RIP, 4299853)

    def test_PUNPCKLDQ_4(self):
        if False:
            i = 10
            return i + 15
        'Instruction PUNPCKLDQ_4\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = 0
        cpu.XMM1 = 8715097878561325495705732644968
        cpu.RIP = 4299848
        cpu.execute()
        self.assertEqual(mem[4299848:4299853], [b'f', b'A', b'\x0f', b'b', b'\xc8'])
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.XMM1, 1955354871813212471400)
        self.assertEqual(cpu.RIP, 4299853)

    def test_PUNPCKLDQ_5(self):
        if False:
            print('Hello World!')
        'Instruction PUNPCKLDQ_5\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = 0
        cpu.XMM1 = 9982748479084702802450508152952
        cpu.RIP = 4299848
        cpu.execute()
        self.assertEqual(mem[4299848:4299853], [b'f', b'A', b'\x0f', b'b', b'\xc8'])
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.XMM1, 2250502776992565297272)
        self.assertEqual(cpu.RIP, 4299853)

    def test_PUNPCKLDQ_6(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKLDQ_6\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = 0
        cpu.XMM1 = 3010670176206127615354242859040
        cpu.RIP = 4299848
        cpu.execute()
        self.assertEqual(mem[4299848:4299853], [b'f', b'A', b'\x0f', b'b', b'\xc8'])
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.XMM1, 627189298506124754976)
        self.assertEqual(cpu.RIP, 4299853)

    def test_PUNPCKLDQ_7(self):
        if False:
            return 10
        'Instruction PUNPCKLDQ_7\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = 0
        cpu.XMM1 = 15053350881178212029429610184888
        cpu.RIP = 4299848
        cpu.execute()
        self.assertEqual(mem[4299848:4299853], [b'f', b'A', b'\x0f', b'b', b'\xc8'])
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.XMM1, 3431094397709976600760)
        self.assertEqual(cpu.RIP, 4299853)

    def test_PUNPCKLDQ_8(self):
        if False:
            return 10
        'Instruction PUNPCKLDQ_8\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = 0
        cpu.XMM1 = 16321001481701589336174385692872
        cpu.RIP = 4299848
        cpu.execute()
        self.assertEqual(mem[4299848:4299853], [b'f', b'A', b'\x0f', b'b', b'\xc8'])
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.XMM1, 3726242302889329426632)
        self.assertEqual(cpu.RIP, 4299853)

    def test_PUNPCKLDQ_9(self):
        if False:
            i = 10
            return i + 15
        'Instruction PUNPCKLDQ_9\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = 0
        cpu.XMM1 = 5545971377252882228843793875008
        cpu.RIP = 4299848
        cpu.execute()
        self.assertEqual(mem[4299848:4299853], [b'f', b'A', b'\x0f', b'b', b'\xc8'])
        self.assertEqual(cpu.XMM8, 0)
        self.assertEqual(cpu.XMM1, 1217485108864830406720)
        self.assertEqual(cpu.RIP, 4299853)

    def test_PUNPCKLQDQ_1(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKLQDQ_1\n        Groups: sse2\n        0x419c82:   punpcklqdq      xmm1, xmm0\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299906, 'f\x0flÈ')
        cpu.XMM0 = 131469649865422795021533112
        cpu.XMM1 = 131469649865422795021533112
        cpu.RIP = 4299906
        cpu.execute()
        self.assertEqual(mem[4299906:4299910], [b'f', b'\x0f', b'l', b'\xc8'])
        self.assertEqual(cpu.XMM0, 131469649865422795021533112)
        self.assertEqual(cpu.XMM1, 131469354717517615668707256)
        self.assertEqual(cpu.RIP, 4299910)

    def test_PUNPCKLQDQ_10(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKLQDQ_10\n        Groups: sse2\n        0x419c6c:   punpcklqdq      xmm8, xmm1\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299884, 'fD\x0flÁ')
        cpu.XMM8 = 131453711878543109968936024
        cpu.XMM1 = 131453711878543109968936024
        cpu.RIP = 4299884
        cpu.execute()
        self.assertEqual(mem[4299884:4299889], [b'f', b'D', b'\x0f', b'l', b'\xc1'])
        self.assertEqual(cpu.XMM8, 131453416730637930616110168)
        self.assertEqual(cpu.XMM1, 131453711878543109968936024)
        self.assertEqual(cpu.RIP, 4299889)

    def test_PUNPCKLQDQ_11(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKLQDQ_11\n        Groups: sse2\n        0x419c82:   punpcklqdq      xmm1, xmm0\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299906, 'f\x0flÈ')
        cpu.XMM0 = 131448399216249881618070328
        cpu.XMM1 = 131448399216249881618070328
        cpu.RIP = 4299906
        cpu.execute()
        self.assertEqual(mem[4299906:4299910], [b'f', b'\x0f', b'l', b'\xc8'])
        self.assertEqual(cpu.XMM0, 131448399216249881618070328)
        self.assertEqual(cpu.XMM1, 131448104068344702265244472)
        self.assertEqual(cpu.RIP, 4299910)

    def test_PUNPCKLQDQ_12(self):
        if False:
            return 10
        'Instruction PUNPCKLQDQ_12\n        Groups: sse2\n        0x419c6c:   punpcklqdq      xmm8, xmm1\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299884, 'fD\x0flÁ')
        cpu.XMM8 = 131460795428267414436756952
        cpu.XMM1 = 131460795428267414436756952
        cpu.RIP = 4299884
        cpu.execute()
        self.assertEqual(mem[4299884:4299889], [b'f', b'D', b'\x0f', b'l', b'\xc1'])
        self.assertEqual(cpu.XMM8, 131460500280362235083931096)
        self.assertEqual(cpu.XMM1, 131460795428267414436756952)
        self.assertEqual(cpu.RIP, 4299889)

    def test_PUNPCKLQDQ_13(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKLQDQ_13\n        Groups: sse2\n        0x419c6c:   punpcklqdq      xmm8, xmm1\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299884, 'fD\x0flÁ')
        cpu.XMM8 = 131465517794750284081970904
        cpu.XMM1 = 131465517794750284081970904
        cpu.RIP = 4299884
        cpu.execute()
        self.assertEqual(mem[4299884:4299889], [b'f', b'D', b'\x0f', b'l', b'\xc1'])
        self.assertEqual(cpu.XMM8, 131465222646845104729145048)
        self.assertEqual(cpu.XMM1, 131465517794750284081970904)
        self.assertEqual(cpu.RIP, 4299889)

    def test_PUNPCKLQDQ_14(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKLQDQ_14\n        Groups: sse2\n        0x419c6c:   punpcklqdq      xmm8, xmm1\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299884, 'fD\x0flÁ')
        cpu.XMM8 = 131450170103680957735025560
        cpu.XMM1 = 131450170103680957735025560
        cpu.RIP = 4299884
        cpu.execute()
        self.assertEqual(mem[4299884:4299889], [b'f', b'D', b'\x0f', b'l', b'\xc1'])
        self.assertEqual(cpu.XMM8, 131449874955775778382199704)
        self.assertEqual(cpu.XMM1, 131450170103680957735025560)
        self.assertEqual(cpu.RIP, 4299889)

    def test_PUNPCKLQDQ_15(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKLQDQ_15\n        Groups: sse2\n        0x419c82:   punpcklqdq      xmm1, xmm0\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299906, 'f\x0flÈ')
        cpu.XMM0 = 131455482765974186085891256
        cpu.XMM1 = 131455482765974186085891256
        cpu.RIP = 4299906
        cpu.execute()
        self.assertEqual(mem[4299906:4299910], [b'f', b'\x0f', b'l', b'\xc8'])
        self.assertEqual(cpu.XMM0, 131455482765974186085891256)
        self.assertEqual(cpu.XMM1, 131455187618069006733065400)
        self.assertEqual(cpu.RIP, 4299910)

    def test_PUNPCKLQDQ_16(self):
        if False:
            print('Hello World!')
        'Instruction PUNPCKLQDQ_16\n        Groups: sse2\n        0x419c82:   punpcklqdq      xmm1, xmm0\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299906, 'f\x0flÈ')
        cpu.XMM0 = 131461385724077773142408696
        cpu.XMM1 = 131461385724077773142408696
        cpu.RIP = 4299906
        cpu.execute()
        self.assertEqual(mem[4299906:4299910], [b'f', b'\x0f', b'l', b'\xc8'])
        self.assertEqual(cpu.XMM0, 131461385724077773142408696)
        self.assertEqual(cpu.XMM1, 131461090576172593789582840)
        self.assertEqual(cpu.RIP, 4299910)

    def test_PUNPCKLQDQ_17(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKLQDQ_17\n        Groups: sse2\n        0x419c82:   punpcklqdq      xmm1, xmm0\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299906, 'f\x0flÈ')
        cpu.XMM0 = 131464927498939925376319160
        cpu.XMM1 = 131464927498939925376319160
        cpu.RIP = 4299906
        cpu.execute()
        self.assertEqual(mem[4299906:4299910], [b'f', b'\x0f', b'l', b'\xc8'])
        self.assertEqual(cpu.XMM0, 131464927498939925376319160)
        self.assertEqual(cpu.XMM1, 131464632351034746023493304)
        self.assertEqual(cpu.RIP, 4299910)

    def test_PUNPCKLQDQ_18(self):
        if False:
            i = 10
            return i + 15
        'Instruction PUNPCKLQDQ_18\n        Groups: sse2\n        0x419c6c:   punpcklqdq      xmm8, xmm1\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299884, 'fD\x0flÁ')
        cpu.XMM8 = 131451350695301675146329048
        cpu.XMM1 = 131451350695301675146329048
        cpu.RIP = 4299884
        cpu.execute()
        self.assertEqual(mem[4299884:4299889], [b'f', b'D', b'\x0f', b'l', b'\xc1'])
        self.assertEqual(cpu.XMM8, 131451055547396495793503192)
        self.assertEqual(cpu.XMM1, 131451350695301675146329048)
        self.assertEqual(cpu.RIP, 4299889)

    def test_PUNPCKLQDQ_19(self):
        if False:
            while True:
                i = 10
        'Instruction PUNPCKLQDQ_19\n        Groups: sse2\n        0x419c82:   punpcklqdq      xmm1, xmm0\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299906, 'f\x0flÈ')
        cpu.XMM0 = 131454302174353468674587768
        cpu.XMM1 = 131454302174353468674587768
        cpu.RIP = 4299906
        cpu.execute()
        self.assertEqual(mem[4299906:4299910], [b'f', b'\x0f', b'l', b'\xc8'])
        self.assertEqual(cpu.XMM0, 131454302174353468674587768)
        self.assertEqual(cpu.XMM1, 131454007026448289321761912)
        self.assertEqual(cpu.RIP, 4299910)

    def test_PUNPCKLQDQ_2(self):
        if False:
            return 10
        'Instruction PUNPCKLQDQ_2\n        Groups: sse2\n        0x419c6c:   punpcklqdq      xmm8, xmm1\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299884, 'fD\x0flÁ')
        cpu.XMM8 = 131467878977991718904577880
        cpu.XMM1 = 131467878977991718904577880
        cpu.RIP = 4299884
        cpu.execute()
        self.assertEqual(mem[4299884:4299889], [b'f', b'D', b'\x0f', b'l', b'\xc1'])
        self.assertEqual(cpu.XMM8, 131467583830086539551752024)
        self.assertEqual(cpu.XMM1, 131467878977991718904577880)
        self.assertEqual(cpu.RIP, 4299889)

    def test_PUNPCKLQDQ_20(self):
        if False:
            while True:
                i = 10
        'Instruction PUNPCKLQDQ_20\n        Groups: sse2\n        0x419c82:   punpcklqdq      xmm1, xmm0\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299906, 'f\x0flÈ')
        cpu.XMM0 = 131444857441387729384159864
        cpu.XMM1 = 131444857441387729384159864
        cpu.RIP = 4299906
        cpu.execute()
        self.assertEqual(mem[4299906:4299910], [b'f', b'\x0f', b'l', b'\xc8'])
        self.assertEqual(cpu.XMM0, 131444857441387729384159864)
        self.assertEqual(cpu.XMM1, 131444562293482550031334008)
        self.assertEqual(cpu.RIP, 4299910)

    def test_PUNPCKLQDQ_21(self):
        if False:
            while True:
                i = 10
        'Instruction PUNPCKLQDQ_21\n        Groups: sse2\n        0x419c6c:   punpcklqdq      xmm8, xmm1\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299884, 'fD\x0flÁ')
        cpu.XMM8 = 131439544779094501033294168
        cpu.XMM1 = 131439544779094501033294168
        cpu.RIP = 4299884
        cpu.execute()
        self.assertEqual(mem[4299884:4299889], [b'f', b'D', b'\x0f', b'l', b'\xc1'])
        self.assertEqual(cpu.XMM8, 131439249631189321680468312)
        self.assertEqual(cpu.XMM1, 131439544779094501033294168)
        self.assertEqual(cpu.RIP, 4299889)

    def test_PUNPCKLQDQ_3(self):
        if False:
            while True:
                i = 10
        'Instruction PUNPCKLQDQ_3\n        Groups: sse2\n        0x419c6c:   punpcklqdq      xmm8, xmm1\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299884, 'fD\x0flÁ')
        cpu.XMM8 = 131447808920439522912418584
        cpu.XMM1 = 131447808920439522912418584
        cpu.RIP = 4299884
        cpu.execute()
        self.assertEqual(mem[4299884:4299889], [b'f', b'D', b'\x0f', b'l', b'\xc1'])
        self.assertEqual(cpu.XMM8, 131447513772534343559592728)
        self.assertEqual(cpu.XMM1, 131447808920439522912418584)
        self.assertEqual(cpu.RIP, 4299889)

    def test_PUNPCKLQDQ_4(self):
        if False:
            while True:
                i = 10
        'Instruction PUNPCKLQDQ_4\n        Groups: sse2\n        0x419c6c:   punpcklqdq      xmm8, xmm1\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299884, 'fD\x0flÁ')
        cpu.XMM8 = 131436003004232348799383704
        cpu.XMM1 = 131436003004232348799383704
        cpu.RIP = 4299884
        cpu.execute()
        self.assertEqual(mem[4299884:4299889], [b'f', b'D', b'\x0f', b'l', b'\xc1'])
        self.assertEqual(cpu.XMM8, 131435707856327169446557848)
        self.assertEqual(cpu.XMM1, 131436003004232348799383704)
        self.assertEqual(cpu.RIP, 4299889)

    def test_PUNPCKLQDQ_5(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKLQDQ_5\n        Groups: sse2\n        0x419c6c:   punpcklqdq      xmm8, xmm1\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299884, 'fD\x0flÁ')
        cpu.XMM8 = 131452531286922392557632536
        cpu.XMM1 = 131452531286922392557632536
        cpu.RIP = 4299884
        cpu.execute()
        self.assertEqual(mem[4299884:4299889], [b'f', b'D', b'\x0f', b'l', b'\xc1'])
        self.assertEqual(cpu.XMM8, 131452236139017213204806680)
        self.assertEqual(cpu.XMM1, 131452531286922392557632536)
        self.assertEqual(cpu.RIP, 4299889)

    def test_PUNPCKLQDQ_6(self):
        if False:
            i = 10
            return i + 15
        'Instruction PUNPCKLQDQ_6\n        Groups: sse2\n        0x419c82:   punpcklqdq      xmm1, xmm0\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299906, 'f\x0flÈ')
        cpu.XMM0 = 131467288682181360198926136
        cpu.XMM1 = 131467288682181360198926136
        cpu.RIP = 4299906
        cpu.execute()
        self.assertEqual(mem[4299906:4299910], [b'f', b'\x0f', b'l', b'\xc8'])
        self.assertEqual(cpu.XMM0, 131467288682181360198926136)
        self.assertEqual(cpu.XMM1, 131466993534276180846100280)
        self.assertEqual(cpu.RIP, 4299910)

    def test_PUNPCKLQDQ_7(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKLQDQ_7\n        Groups: sse2\n        0x419c6c:   punpcklqdq      xmm8, xmm1\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299884, 'fD\x0flÁ')
        cpu.XMM8 = 131444267145577370678508120
        cpu.XMM1 = 131444267145577370678508120
        cpu.RIP = 4299884
        cpu.execute()
        self.assertEqual(mem[4299884:4299889], [b'f', b'D', b'\x0f', b'l', b'\xc1'])
        self.assertEqual(cpu.XMM8, 131443971997672191325682264)
        self.assertEqual(cpu.XMM1, 131444267145577370678508120)
        self.assertEqual(cpu.RIP, 4299889)

    def test_PUNPCKLQDQ_8(self):
        if False:
            i = 10
            return i + 15
        'Instruction PUNPCKLQDQ_8\n        Groups: sse2\n        0x419c82:   punpcklqdq      xmm1, xmm0\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299906, 'f\x0flÈ')
        cpu.XMM0 = 131442496258146294561552888
        cpu.XMM1 = 131442496258146294561552888
        cpu.RIP = 4299906
        cpu.execute()
        self.assertEqual(mem[4299906:4299910], [b'f', b'\x0f', b'l', b'\xc8'])
        self.assertEqual(cpu.XMM0, 131442496258146294561552888)
        self.assertEqual(cpu.XMM1, 131442201110241115208727032)
        self.assertEqual(cpu.RIP, 4299910)

    def test_PUNPCKLQDQ_9(self):
        if False:
            while True:
                i = 10
        'Instruction PUNPCKLQDQ_9\n        Groups: sse2\n        0x419c82:   punpcklqdq      xmm1, xmm0\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299906, 'f\x0flÈ')
        cpu.XMM0 = 131460205132457055731105208
        cpu.XMM1 = 131460205132457055731105208
        cpu.RIP = 4299906
        cpu.execute()
        self.assertEqual(mem[4299906:4299910], [b'f', b'\x0f', b'l', b'\xc8'])
        self.assertEqual(cpu.XMM0, 131460205132457055731105208)
        self.assertEqual(cpu.XMM1, 131459909984551876378279352)
        self.assertEqual(cpu.RIP, 4299910)

    def test_PUNPCKLWD_1(self):
        if False:
            print('Hello World!')
        'Instruction PUNPCKLWD_1\n        Groups: sse2\n        0x4668b6:   punpcklwd       xmm1, xmm1\n        '
        mem = Memory64()
        cpu = AMD64Cpu(mem)
        mem.mmap(4612096, 4096, 'rwx')
        mem.write(4614326, 'f\x0faÉ')
        cpu.XMM1 = 12079
        cpu.RIP = 4614326
        cpu.execute()
        self.assertEqual(mem[4614326:4614330], [b'f', b'\x0f', b'a', b'\xc9'])
        self.assertEqual(cpu.XMM1, 791621423)
        self.assertEqual(cpu.RIP, 4614330)

    def test_PUNPCKHDQ_1_symbolic(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKHDQ_1\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 6179796677514570882216181629000)
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.RIP = 4299843
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299843, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299844, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299845, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299846, 8) == ord('j'))
        condition = Operators.AND(condition, cpu.read_int(4299847, 8) == ord('À'))
        condition = Operators.AND(condition, cpu.XMM0 == 1438846037749345026124)
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.RIP == 4299848)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHDQ_10_symbolic(self):
        if False:
            return 10
        'Instruction PUNPCKHDQ_10\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 4278320776729504922099018367024)
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.RIP = 4299843
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299843, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299844, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299845, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299846, 8) == ord('j'))
        condition = Operators.AND(condition, cpu.read_int(4299847, 8) == ord('À'))
        condition = Operators.AND(condition, cpu.XMM0 == 996124179980315787316)
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.RIP == 4299848)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHDQ_11_symbolic(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKHDQ_11\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 4912146076991193575471406121016)
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.RIP = 4299843
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299843, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299844, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299845, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299846, 8) == ord('j'))
        condition = Operators.AND(condition, cpu.read_int(4299847, 8) == ord('À'))
        condition = Operators.AND(condition, cpu.XMM0 == 1143698132569992200252)
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.RIP == 4299848)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHDQ_12_symbolic(self):
        if False:
            return 10
        'Instruction PUNPCKHDQ_12\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 11250399079608080109195283660936)
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.RIP = 4299843
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299843, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299844, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299845, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299846, 8) == ord('j'))
        condition = Operators.AND(condition, cpu.read_int(4299847, 8) == ord('À'))
        condition = Operators.AND(condition, cpu.XMM0 == 2619437658466756329612)
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.RIP == 4299848)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHDQ_13_symbolic(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKHDQ_13\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 18222477382486655296291548954848)
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.RIP = 4299843
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299843, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299844, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299845, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299846, 8) == ord('j'))
        condition = Operators.AND(condition, cpu.read_int(4299847, 8) == ord('À'))
        condition = Operators.AND(condition, cpu.XMM0 == 4242751136953196871908)
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.RIP == 4299848)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHDQ_14_symbolic(self):
        if False:
            return 10
        'Instruction PUNPCKHDQ_14\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 9982748479084702802450508152952)
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.RIP = 4299843
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299843, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299844, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299845, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299846, 8) == ord('j'))
        condition = Operators.AND(condition, cpu.read_int(4299847, 8) == ord('À'))
        condition = Operators.AND(condition, cpu.XMM0 == 2324289753287403503740)
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.RIP == 4299848)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHDQ_15_symbolic(self):
        if False:
            return 10
        'Instruction PUNPCKHDQ_15\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 11884224379869768762567671414928)
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.RIP = 4299843
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299843, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299844, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299845, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299846, 8) == ord('j'))
        condition = Operators.AND(condition, cpu.read_int(4299847, 8) == ord('À'))
        condition = Operators.AND(condition, cpu.XMM0 == 2767011611056432742548)
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.RIP == 4299848)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHDQ_16_symbolic(self):
        if False:
            while True:
                i = 10
        'Instruction PUNPCKHDQ_16\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 475368975159373001864691843072)
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.RIP = 4299843
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299843, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299844, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299845, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299846, 8) == ord('j'))
        condition = Operators.AND(condition, cpu.read_int(4299847, 8) == ord('À'))
        condition = Operators.AND(condition, cpu.XMM0 == 110680464442257309700)
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.RIP == 4299848)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHDQ_17_symbolic(self):
        if False:
            return 10
        'Instruction PUNPCKHDQ_17\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 16321001481701589336174385692872)
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.RIP = 4299843
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299843, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299844, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299845, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299846, 8) == ord('j'))
        condition = Operators.AND(condition, cpu.read_int(4299847, 8) == ord('À'))
        condition = Operators.AND(condition, cpu.XMM0 == 3800029279184167633100)
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.RIP == 4299848)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHDQ_18_symbolic(self):
        if False:
            return 10
        'Instruction PUNPCKHDQ_18\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 12518049680131457415940059168920)
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.RIP = 4299843
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299843, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299844, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299845, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299846, 8) == ord('j'))
        condition = Operators.AND(condition, cpu.read_int(4299847, 8) == ord('À'))
        condition = Operators.AND(condition, cpu.XMM0 == 2914585563646109155484)
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.RIP == 4299848)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHDQ_19_symbolic(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKHDQ_19\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 5545971377252882228843793875008)
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.RIP = 4299843
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299843, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299844, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299845, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299846, 8) == ord('j'))
        condition = Operators.AND(condition, cpu.read_int(4299847, 8) == ord('À'))
        condition = Operators.AND(condition, cpu.XMM0 == 1291272085159668613188)
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.RIP == 4299848)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHDQ_2_symbolic(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKHDQ_2\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 15053350881178212029429610184888)
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.RIP = 4299843
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299843, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299844, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299845, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299846, 8) == ord('j'))
        condition = Operators.AND(condition, cpu.read_int(4299847, 8) == ord('À'))
        condition = Operators.AND(condition, cpu.XMM0 == 3504881374004814807228)
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.RIP == 4299848)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHDQ_20_symbolic(self):
        if False:
            return 10
        'Instruction PUNPCKHDQ_20\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 8081272578299636842333344890976)
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.RIP = 4299843
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299843, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299844, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299845, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299846, 8) == ord('j'))
        condition = Operators.AND(condition, cpu.read_int(4299847, 8) == ord('À'))
        condition = Operators.AND(condition, cpu.XMM0 == 1881567895518374264932)
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.RIP == 4299848)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHDQ_21_symbolic(self):
        if False:
            print('Hello World!')
        'Instruction PUNPCKHDQ_21\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 7447447278037948188960957136984)
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.RIP = 4299843
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299843, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299844, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299845, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299846, 8) == ord('j'))
        condition = Operators.AND(condition, cpu.read_int(4299847, 8) == ord('À'))
        condition = Operators.AND(condition, cpu.XMM0 == 1733993942928697851996)
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.RIP == 4299848)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHDQ_3_symbolic(self):
        if False:
            return 10
        'Instruction PUNPCKHDQ_3\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 8715097878561325495705732644968)
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.RIP = 4299843
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299843, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299844, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299845, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299846, 8) == ord('j'))
        condition = Operators.AND(condition, cpu.read_int(4299847, 8) == ord('À'))
        condition = Operators.AND(condition, cpu.XMM0 == 2029141848108050677868)
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.RIP == 4299848)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHDQ_4_symbolic(self):
        if False:
            while True:
                i = 10
        'Instruction PUNPCKHDQ_4\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 15687176181439900682801997938880)
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.RIP = 4299843
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299843, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299844, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299845, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299846, 8) == ord('j'))
        condition = Operators.AND(condition, cpu.read_int(4299847, 8) == ord('À'))
        condition = Operators.AND(condition, cpu.XMM0 == 3652455326594491220164)
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.RIP == 4299848)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHDQ_5_symbolic(self):
        if False:
            return 10
        'Instruction PUNPCKHDQ_5\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 14419525580916523376057222430896)
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.RIP = 4299843
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299843, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299844, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299845, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299846, 8) == ord('j'))
        condition = Operators.AND(condition, cpu.read_int(4299847, 8) == ord('À'))
        condition = Operators.AND(condition, cpu.XMM0 == 3357307421415138394292)
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.RIP == 4299848)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHDQ_6_symbolic(self):
        if False:
            i = 10
            return i + 15
        'Instruction PUNPCKHDQ_6\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 13785700280654834722684834676904)
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.RIP = 4299843
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299843, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299844, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299845, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299846, 8) == ord('j'))
        condition = Operators.AND(condition, cpu.read_int(4299847, 8) == ord('À'))
        condition = Operators.AND(condition, cpu.XMM0 == 3209733468825461981356)
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.RIP == 4299848)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHDQ_7_symbolic(self):
        if False:
            while True:
                i = 10
        'Instruction PUNPCKHDQ_7\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 1109194275421061655237079597064)
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.RIP = 4299843
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299843, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299844, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299845, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299846, 8) == ord('j'))
        condition = Operators.AND(condition, cpu.read_int(4299847, 8) == ord('À'))
        condition = Operators.AND(condition, cpu.XMM0 == 258254417031933722636)
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.RIP == 4299848)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHDQ_8_symbolic(self):
        if False:
            return 10
        'Instruction PUNPCKHDQ_8\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 10616573779346391455822895906944)
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.RIP = 4299843
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299843, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299844, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299845, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299846, 8) == ord('j'))
        condition = Operators.AND(condition, cpu.read_int(4299847, 8) == ord('À'))
        condition = Operators.AND(condition, cpu.XMM0 == 2471863705877079916676)
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.RIP == 4299848)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHDQ_9_symbolic(self):
        if False:
            return 10
        'Instruction PUNPCKHDQ_9\n        Groups: sse2\n        0x419c43:   punpckhdq       xmm0, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299843, 'fA\x0fjÀ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 16954826781963277989546773446864)
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.RIP = 4299843
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299843, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299844, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299845, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299846, 8) == ord('j'))
        condition = Operators.AND(condition, cpu.read_int(4299847, 8) == ord('À'))
        condition = Operators.AND(condition, cpu.XMM0 == 3947603231773844046036)
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.RIP == 4299848)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHQDQ_1_symbolic(self):
        if False:
            print('Hello World!')
        'Instruction PUNPCKHQDQ_1\n        Groups: sse2\n        0x419c71:   punpckhqdq      xmm1, xmm1\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299889, 'f\x0fmÉ')
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 131446628328818805501115096)
        cpu.RIP = 4299889
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299889, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299890, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299891, 8) == ord('m'))
        condition = Operators.AND(condition, cpu.read_int(4299892, 8) == ord('É'))
        condition = Operators.AND(condition, cpu.XMM1 == 131446628328818805501115112)
        condition = Operators.AND(condition, cpu.RIP == 4299893)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHQDQ_10_symbolic(self):
        if False:
            i = 10
            return i + 15
        'Instruction PUNPCKHQDQ_10\n        Groups: sse2\n        0x419c71:   punpckhqdq      xmm1, xmm1\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299889, 'f\x0fmÉ')
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 131437183595853066210687192)
        cpu.RIP = 4299889
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299889, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299890, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299891, 8) == ord('m'))
        condition = Operators.AND(condition, cpu.read_int(4299892, 8) == ord('É'))
        condition = Operators.AND(condition, cpu.XMM1 == 131437183595853066210687208)
        condition = Operators.AND(condition, cpu.RIP == 4299893)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHQDQ_11_symbolic(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKHQDQ_11\n        Groups: sse2\n        0x419c86:   punpckhqdq      xmm0, xmm0\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299910, 'f\x0fmÀ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 131444857441387729384159864)
        cpu.RIP = 4299910
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299912, 8) == ord('m'))
        condition = Operators.AND(condition, cpu.read_int(4299913, 8) == ord('À'))
        condition = Operators.AND(condition, cpu.read_int(4299910, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299911, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.XMM0 == 131444857441387729384159880)
        condition = Operators.AND(condition, cpu.RIP == 4299914)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHQDQ_12_symbolic(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKHQDQ_12\n        Groups: sse2\n        0x419c71:   punpckhqdq      xmm1, xmm1\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299889, 'f\x0fmÉ')
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 131445447737198088089811608)
        cpu.RIP = 4299889
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299889, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299890, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299891, 8) == ord('m'))
        condition = Operators.AND(condition, cpu.read_int(4299892, 8) == ord('É'))
        condition = Operators.AND(condition, cpu.XMM1 == 131445447737198088089811624)
        condition = Operators.AND(condition, cpu.RIP == 4299893)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHQDQ_13_symbolic(self):
        if False:
            i = 10
            return i + 15
        'Instruction PUNPCKHQDQ_13\n        Groups: sse2\n        0x419c71:   punpckhqdq      xmm1, xmm1\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299889, 'f\x0fmÉ')
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 131465517794750284081970904)
        cpu.RIP = 4299889
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299889, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299890, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299891, 8) == ord('m'))
        condition = Operators.AND(condition, cpu.read_int(4299892, 8) == ord('É'))
        condition = Operators.AND(condition, cpu.XMM1 == 131465517794750284081970920)
        condition = Operators.AND(condition, cpu.RIP == 4299893)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHQDQ_14_symbolic(self):
        if False:
            print('Hello World!')
        'Instruction PUNPCKHQDQ_14\n        Groups: sse2\n        0x419c86:   punpckhqdq      xmm0, xmm0\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299910, 'f\x0fmÀ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 131467288682181360198926136)
        cpu.RIP = 4299910
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299912, 8) == ord('m'))
        condition = Operators.AND(condition, cpu.read_int(4299913, 8) == ord('À'))
        condition = Operators.AND(condition, cpu.read_int(4299910, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299911, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.XMM0 == 131467288682181360198926152)
        condition = Operators.AND(condition, cpu.RIP == 4299914)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHQDQ_15_symbolic(self):
        if False:
            print('Hello World!')
        'Instruction PUNPCKHQDQ_15\n        Groups: sse2\n        0x419c86:   punpckhqdq      xmm0, xmm0\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299910, 'f\x0fmÀ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 131460205132457055731105208)
        cpu.RIP = 4299910
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299912, 8) == ord('m'))
        condition = Operators.AND(condition, cpu.read_int(4299913, 8) == ord('À'))
        condition = Operators.AND(condition, cpu.read_int(4299910, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299911, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.XMM0 == 131460205132457055731105224)
        condition = Operators.AND(condition, cpu.RIP == 4299914)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHQDQ_16_symbolic(self):
        if False:
            print('Hello World!')
        'Instruction PUNPCKHQDQ_16\n        Groups: sse2\n        0x419c71:   punpckhqdq      xmm1, xmm1\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299889, 'f\x0fmÉ')
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 131439544779094501033294168)
        cpu.RIP = 4299889
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299889, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299890, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299891, 8) == ord('m'))
        condition = Operators.AND(condition, cpu.read_int(4299892, 8) == ord('É'))
        condition = Operators.AND(condition, cpu.XMM1 == 131439544779094501033294184)
        condition = Operators.AND(condition, cpu.RIP == 4299893)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHQDQ_17_symbolic(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKHQDQ_17\n        Groups: sse2\n        0x419c86:   punpckhqdq      xmm0, xmm0\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299910, 'f\x0fmÀ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 131448399216249881618070328)
        cpu.RIP = 4299910
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299912, 8) == ord('m'))
        condition = Operators.AND(condition, cpu.read_int(4299913, 8) == ord('À'))
        condition = Operators.AND(condition, cpu.read_int(4299910, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299911, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.XMM0 == 131448399216249881618070344)
        condition = Operators.AND(condition, cpu.RIP == 4299914)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHQDQ_18_symbolic(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKHQDQ_18\n        Groups: sse2\n        0x419c86:   punpckhqdq      xmm0, xmm0\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299910, 'f\x0fmÀ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 131437773891663424916338936)
        cpu.RIP = 4299910
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299912, 8) == ord('m'))
        condition = Operators.AND(condition, cpu.read_int(4299913, 8) == ord('À'))
        condition = Operators.AND(condition, cpu.read_int(4299910, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299911, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.XMM0 == 131437773891663424916338952)
        condition = Operators.AND(condition, cpu.RIP == 4299914)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHQDQ_19_symbolic(self):
        if False:
            while True:
                i = 10
        'Instruction PUNPCKHQDQ_19\n        Groups: sse2\n        0x419c71:   punpckhqdq      xmm1, xmm1\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299889, 'f\x0fmÉ')
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 131470240161233153727184856)
        cpu.RIP = 4299889
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299889, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299890, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299891, 8) == ord('m'))
        condition = Operators.AND(condition, cpu.read_int(4299892, 8) == ord('É'))
        condition = Operators.AND(condition, cpu.XMM1 == 131470240161233153727184872)
        condition = Operators.AND(condition, cpu.RIP == 4299893)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHQDQ_2_symbolic(self):
        if False:
            print('Hello World!')
        'Instruction PUNPCKHQDQ_2\n        Groups: sse2\n        0x419c86:   punpckhqdq      xmm0, xmm0\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299910, 'f\x0fmÀ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 131435412708421990093731960)
        cpu.RIP = 4299910
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299912, 8) == ord('m'))
        condition = Operators.AND(condition, cpu.read_int(4299913, 8) == ord('À'))
        condition = Operators.AND(condition, cpu.read_int(4299910, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299911, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.XMM0 == 131435412708421990093731976)
        condition = Operators.AND(condition, cpu.RIP == 4299914)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHQDQ_20_symbolic(self):
        if False:
            return 10
        'Instruction PUNPCKHQDQ_20\n        Groups: sse2\n        0x419c71:   punpckhqdq      xmm1, xmm1\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299889, 'f\x0fmÉ')
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 131438364187473783621990680)
        cpu.RIP = 4299889
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299889, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299890, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299891, 8) == ord('m'))
        condition = Operators.AND(condition, cpu.read_int(4299892, 8) == ord('É'))
        condition = Operators.AND(condition, cpu.XMM1 == 131438364187473783621990696)
        condition = Operators.AND(condition, cpu.RIP == 4299893)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHQDQ_21_symbolic(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKHQDQ_21\n        Groups: sse2\n        0x419c86:   punpckhqdq      xmm0, xmm0\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299910, 'f\x0fmÀ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 131446038033008446795463352)
        cpu.RIP = 4299910
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299912, 8) == ord('m'))
        condition = Operators.AND(condition, cpu.read_int(4299913, 8) == ord('À'))
        condition = Operators.AND(condition, cpu.read_int(4299910, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299911, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.XMM0 == 131446038033008446795463368)
        condition = Operators.AND(condition, cpu.RIP == 4299914)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHQDQ_3_symbolic(self):
        if False:
            while True:
                i = 10
        'Instruction PUNPCKHQDQ_3\n        Groups: sse2\n        0x419c71:   punpckhqdq      xmm1, xmm1\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299889, 'f\x0fmÉ')
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 131451350695301675146329048)
        cpu.RIP = 4299889
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299889, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299890, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299891, 8) == ord('m'))
        condition = Operators.AND(condition, cpu.read_int(4299892, 8) == ord('É'))
        condition = Operators.AND(condition, cpu.XMM1 == 131451350695301675146329064)
        condition = Operators.AND(condition, cpu.RIP == 4299893)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHQDQ_4_symbolic(self):
        if False:
            while True:
                i = 10
        'Instruction PUNPCKHQDQ_4\n        Groups: sse2\n        0x419c86:   punpckhqdq      xmm0, xmm0\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299910, 'f\x0fmÀ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 131459024540836338319801720)
        cpu.RIP = 4299910
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299912, 8) == ord('m'))
        condition = Operators.AND(condition, cpu.read_int(4299913, 8) == ord('À'))
        condition = Operators.AND(condition, cpu.read_int(4299910, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299911, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.XMM0 == 131459024540836338319801736)
        condition = Operators.AND(condition, cpu.RIP == 4299914)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHQDQ_5_symbolic(self):
        if False:
            i = 10
            return i + 15
        'Instruction PUNPCKHQDQ_5\n        Groups: sse2\n        0x419c71:   punpckhqdq      xmm1, xmm1\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299889, 'f\x0fmÉ')
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 131434822412611631388080216)
        cpu.RIP = 4299889
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299889, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299890, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299891, 8) == ord('m'))
        condition = Operators.AND(condition, cpu.read_int(4299892, 8) == ord('É'))
        condition = Operators.AND(condition, cpu.XMM1 == 131434822412611631388080232)
        condition = Operators.AND(condition, cpu.RIP == 4299893)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHQDQ_6_symbolic(self):
        if False:
            while True:
                i = 10
        'Instruction PUNPCKHQDQ_6\n        Groups: sse2\n        0x419c71:   punpckhqdq      xmm1, xmm1\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299889, 'f\x0fmÉ')
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 131460795428267414436756952)
        cpu.RIP = 4299889
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299889, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299890, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299891, 8) == ord('m'))
        condition = Operators.AND(condition, cpu.read_int(4299892, 8) == ord('É'))
        condition = Operators.AND(condition, cpu.XMM1 == 131460795428267414436756968)
        condition = Operators.AND(condition, cpu.RIP == 4299893)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHQDQ_7_symbolic(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKHQDQ_7\n        Groups: sse2\n        0x419c71:   punpckhqdq      xmm1, xmm1\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299889, 'f\x0fmÉ')
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 131457253653405262202846488)
        cpu.RIP = 4299889
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299889, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299890, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299891, 8) == ord('m'))
        condition = Operators.AND(condition, cpu.read_int(4299892, 8) == ord('É'))
        condition = Operators.AND(condition, cpu.XMM1 == 131457253653405262202846504)
        condition = Operators.AND(condition, cpu.RIP == 4299893)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHQDQ_8_symbolic(self):
        if False:
            while True:
                i = 10
        'Instruction PUNPCKHQDQ_8\n        Groups: sse2\n        0x419c71:   punpckhqdq      xmm1, xmm1\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299889, 'f\x0fmÉ')
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 131436003004232348799383704)
        cpu.RIP = 4299889
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299889, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299890, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299891, 8) == ord('m'))
        condition = Operators.AND(condition, cpu.read_int(4299892, 8) == ord('É'))
        condition = Operators.AND(condition, cpu.XMM1 == 131436003004232348799383720)
        condition = Operators.AND(condition, cpu.RIP == 4299893)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKHQDQ_9_symbolic(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKHQDQ_9\n        Groups: sse2\n        0x419c86:   punpckhqdq      xmm0, xmm0\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299910, 'f\x0fmÀ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 131438954483284142327642424)
        cpu.RIP = 4299910
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299912, 8) == ord('m'))
        condition = Operators.AND(condition, cpu.read_int(4299913, 8) == ord('À'))
        condition = Operators.AND(condition, cpu.read_int(4299910, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299911, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.XMM0 == 131438954483284142327642440)
        condition = Operators.AND(condition, cpu.RIP == 4299914)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLBW_1_symbolic(self):
        if False:
            i = 10
            return i + 15
        'Instruction PUNPCKLBW_1\n        Groups: sse2\n        0x4668ac:   punpcklbw       xmm1, xmm1\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4612096, 4096, 'rwx')
        mem.write(4614316, 'f\x0f`É')
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 47)
        cpu.RIP = 4614316
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4614316, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4614317, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4614318, 8) == ord('`'))
        condition = Operators.AND(condition, cpu.read_int(4614319, 8) == ord('É'))
        condition = Operators.AND(condition, cpu.XMM1 == 12079)
        condition = Operators.AND(condition, cpu.RIP == 4614320)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLDQ_1_symbolic(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKLDQ_1\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 13151874980393146069312446922912)
        cpu.RIP = 4299848
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299848, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299849, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299850, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299851, 8) == ord('b'))
        condition = Operators.AND(condition, cpu.read_int(4299852, 8) == ord('È'))
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.XMM1 == 2988372539940947361952)
        condition = Operators.AND(condition, cpu.RIP == 4299853)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLDQ_10_symbolic(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKLDQ_10\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 4912146076991193575471406121016)
        cpu.RIP = 4299848
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299848, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299849, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299850, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299851, 8) == ord('b'))
        condition = Operators.AND(condition, cpu.read_int(4299852, 8) == ord('È'))
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.XMM1 == 1069911156275153993784)
        condition = Operators.AND(condition, cpu.RIP == 4299853)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLDQ_11_symbolic(self):
        if False:
            return 10
        'Instruction PUNPCKLDQ_11\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 475368975159373001864691843072)
        cpu.RIP = 4299848
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299848, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299849, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299850, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299851, 8) == ord('b'))
        condition = Operators.AND(condition, cpu.read_int(4299852, 8) == ord('È'))
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.XMM1 == 36893488147419103232)
        condition = Operators.AND(condition, cpu.RIP == 4299853)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLDQ_12_symbolic(self):
        if False:
            return 10
        'Instruction PUNPCKLDQ_12\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 2376844875944438961981855105048)
        cpu.RIP = 4299848
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299848, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299849, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299850, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299851, 8) == ord('b'))
        condition = Operators.AND(condition, cpu.read_int(4299852, 8) == ord('È'))
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.XMM1 == 479615345916448342040)
        condition = Operators.AND(condition, cpu.RIP == 4299853)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLDQ_13_symbolic(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKLDQ_13\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 1109194275421061655237079597064)
        cpu.RIP = 4299848
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299848, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299849, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299850, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299851, 8) == ord('b'))
        condition = Operators.AND(condition, cpu.read_int(4299852, 8) == ord('È'))
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.XMM1 == 184467440737095516168)
        condition = Operators.AND(condition, cpu.RIP == 4299853)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLDQ_14_symbolic(self):
        if False:
            print('Hello World!')
        'Instruction PUNPCKLDQ_14\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 18856302682748343949663936708840)
        cpu.RIP = 4299848
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299848, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299849, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299850, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299851, 8) == ord('b'))
        condition = Operators.AND(condition, cpu.read_int(4299852, 8) == ord('È'))
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.XMM1 == 4316538113248035078376)
        condition = Operators.AND(condition, cpu.RIP == 4299853)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLDQ_15_symbolic(self):
        if False:
            while True:
                i = 10
        'Instruction PUNPCKLDQ_15\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 3644495476467816268726630613032)
        cpu.RIP = 4299848
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299848, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299849, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299850, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299851, 8) == ord('b'))
        condition = Operators.AND(condition, cpu.read_int(4299852, 8) == ord('È'))
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.XMM1 == 774763251095801167912)
        condition = Operators.AND(condition, cpu.RIP == 4299853)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLDQ_16_symbolic(self):
        if False:
            while True:
                i = 10
        'Instruction PUNPCKLDQ_16\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 19490127983010032603036324462832)
        cpu.RIP = 4299848
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299848, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299849, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299850, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299851, 8) == ord('b'))
        condition = Operators.AND(condition, cpu.read_int(4299852, 8) == ord('È'))
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.XMM1 == 4464112065837711491312)
        condition = Operators.AND(condition, cpu.RIP == 4299853)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLDQ_17_symbolic(self):
        if False:
            while True:
                i = 10
        'Instruction PUNPCKLDQ_17\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 12518049680131457415940059168920)
        cpu.RIP = 4299848
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299848, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299849, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299850, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299851, 8) == ord('b'))
        condition = Operators.AND(condition, cpu.read_int(4299852, 8) == ord('È'))
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.XMM1 == 2840798587351270949016)
        condition = Operators.AND(condition, cpu.RIP == 4299853)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLDQ_18_symbolic(self):
        if False:
            return 10
        'Instruction PUNPCKLDQ_18\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 1743019575682750308609467351056)
        cpu.RIP = 4299848
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299848, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299849, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299850, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299851, 8) == ord('b'))
        condition = Operators.AND(condition, cpu.read_int(4299852, 8) == ord('È'))
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.XMM1 == 332041393326771929104)
        condition = Operators.AND(condition, cpu.RIP == 4299853)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLDQ_19_symbolic(self):
        if False:
            print('Hello World!')
        'Instruction PUNPCKLDQ_19\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 11884224379869768762567671414928)
        cpu.RIP = 4299848
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299848, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299849, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299850, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299851, 8) == ord('b'))
        condition = Operators.AND(condition, cpu.read_int(4299852, 8) == ord('È'))
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.XMM1 == 2693224634761594536080)
        condition = Operators.AND(condition, cpu.RIP == 4299853)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLDQ_2_symbolic(self):
        if False:
            while True:
                i = 10
        'Instruction PUNPCKLDQ_2\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 9348923178823014149078120398960)
        cpu.RIP = 4299848
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299848, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299849, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299850, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299851, 8) == ord('b'))
        condition = Operators.AND(condition, cpu.read_int(4299852, 8) == ord('È'))
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.XMM1 == 2102928824402888884336)
        condition = Operators.AND(condition, cpu.RIP == 4299853)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLDQ_20_symbolic(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKLDQ_20\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 17588652082224966642919161200856)
        cpu.RIP = 4299848
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299848, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299849, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299850, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299851, 8) == ord('b'))
        condition = Operators.AND(condition, cpu.read_int(4299852, 8) == ord('È'))
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.XMM1 == 4021390208068682252504)
        condition = Operators.AND(condition, cpu.RIP == 4299853)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLDQ_21_symbolic(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKLDQ_21\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 18222477382486655296291548954848)
        cpu.RIP = 4299848
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299848, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299849, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299850, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299851, 8) == ord('b'))
        condition = Operators.AND(condition, cpu.read_int(4299852, 8) == ord('È'))
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.XMM1 == 4168964160658358665440)
        condition = Operators.AND(condition, cpu.RIP == 4299853)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLDQ_3_symbolic(self):
        if False:
            while True:
                i = 10
        'Instruction PUNPCKLDQ_3\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 4278320776729504922099018367024)
        cpu.RIP = 4299848
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299848, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299849, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299850, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299851, 8) == ord('b'))
        condition = Operators.AND(condition, cpu.read_int(4299852, 8) == ord('È'))
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.XMM1 == 922337203685477580848)
        condition = Operators.AND(condition, cpu.RIP == 4299853)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLDQ_4_symbolic(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKLDQ_4\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 8715097878561325495705732644968)
        cpu.RIP = 4299848
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299848, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299849, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299850, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299851, 8) == ord('b'))
        condition = Operators.AND(condition, cpu.read_int(4299852, 8) == ord('È'))
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.XMM1 == 1955354871813212471400)
        condition = Operators.AND(condition, cpu.RIP == 4299853)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLDQ_5_symbolic(self):
        if False:
            print('Hello World!')
        'Instruction PUNPCKLDQ_5\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 9982748479084702802450508152952)
        cpu.RIP = 4299848
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299848, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299849, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299850, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299851, 8) == ord('b'))
        condition = Operators.AND(condition, cpu.read_int(4299852, 8) == ord('È'))
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.XMM1 == 2250502776992565297272)
        condition = Operators.AND(condition, cpu.RIP == 4299853)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLDQ_6_symbolic(self):
        if False:
            i = 10
            return i + 15
        'Instruction PUNPCKLDQ_6\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 3010670176206127615354242859040)
        cpu.RIP = 4299848
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299848, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299849, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299850, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299851, 8) == ord('b'))
        condition = Operators.AND(condition, cpu.read_int(4299852, 8) == ord('È'))
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.XMM1 == 627189298506124754976)
        condition = Operators.AND(condition, cpu.RIP == 4299853)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLDQ_7_symbolic(self):
        if False:
            while True:
                i = 10
        'Instruction PUNPCKLDQ_7\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 15053350881178212029429610184888)
        cpu.RIP = 4299848
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299848, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299849, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299850, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299851, 8) == ord('b'))
        condition = Operators.AND(condition, cpu.read_int(4299852, 8) == ord('È'))
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.XMM1 == 3431094397709976600760)
        condition = Operators.AND(condition, cpu.RIP == 4299853)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLDQ_8_symbolic(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKLDQ_8\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 16321001481701589336174385692872)
        cpu.RIP = 4299848
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299848, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299849, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299850, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299851, 8) == ord('b'))
        condition = Operators.AND(condition, cpu.read_int(4299852, 8) == ord('È'))
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.XMM1 == 3726242302889329426632)
        condition = Operators.AND(condition, cpu.RIP == 4299853)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLDQ_9_symbolic(self):
        if False:
            print('Hello World!')
        'Instruction PUNPCKLDQ_9\n        Groups: sse2\n        0x419c48:   punpckldq       xmm1, xmm8\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299848, 'fA\x0fbÈ')
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 0)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 5545971377252882228843793875008)
        cpu.RIP = 4299848
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299848, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299849, 8) == ord('A'))
        condition = Operators.AND(condition, cpu.read_int(4299850, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299851, 8) == ord('b'))
        condition = Operators.AND(condition, cpu.read_int(4299852, 8) == ord('È'))
        condition = Operators.AND(condition, cpu.XMM8 == 0)
        condition = Operators.AND(condition, cpu.XMM1 == 1217485108864830406720)
        condition = Operators.AND(condition, cpu.RIP == 4299853)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLQDQ_1_symbolic(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKLQDQ_1\n        Groups: sse2\n        0x419c82:   punpcklqdq      xmm1, xmm0\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299906, 'f\x0flÈ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 131469649865422795021533112)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 131469649865422795021533112)
        cpu.RIP = 4299906
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299906, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299907, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299908, 8) == ord('l'))
        condition = Operators.AND(condition, cpu.read_int(4299909, 8) == ord('È'))
        condition = Operators.AND(condition, cpu.XMM0 == 131469649865422795021533112)
        condition = Operators.AND(condition, cpu.XMM1 == 131469354717517615668707256)
        condition = Operators.AND(condition, cpu.RIP == 4299910)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLQDQ_10_symbolic(self):
        if False:
            print('Hello World!')
        'Instruction PUNPCKLQDQ_10\n        Groups: sse2\n        0x419c6c:   punpcklqdq      xmm8, xmm1\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299884, 'fD\x0flÁ')
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 131453711878543109968936024)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 131453711878543109968936024)
        cpu.RIP = 4299884
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299888, 8) == ord('Á'))
        condition = Operators.AND(condition, cpu.read_int(4299884, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299885, 8) == ord('D'))
        condition = Operators.AND(condition, cpu.read_int(4299886, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299887, 8) == ord('l'))
        condition = Operators.AND(condition, cpu.XMM8 == 131453416730637930616110168)
        condition = Operators.AND(condition, cpu.XMM1 == 131453711878543109968936024)
        condition = Operators.AND(condition, cpu.RIP == 4299889)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLQDQ_11_symbolic(self):
        if False:
            i = 10
            return i + 15
        'Instruction PUNPCKLQDQ_11\n        Groups: sse2\n        0x419c82:   punpcklqdq      xmm1, xmm0\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299906, 'f\x0flÈ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 131448399216249881618070328)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 131448399216249881618070328)
        cpu.RIP = 4299906
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299906, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299907, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299908, 8) == ord('l'))
        condition = Operators.AND(condition, cpu.read_int(4299909, 8) == ord('È'))
        condition = Operators.AND(condition, cpu.XMM0 == 131448399216249881618070328)
        condition = Operators.AND(condition, cpu.XMM1 == 131448104068344702265244472)
        condition = Operators.AND(condition, cpu.RIP == 4299910)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLQDQ_12_symbolic(self):
        if False:
            print('Hello World!')
        'Instruction PUNPCKLQDQ_12\n        Groups: sse2\n        0x419c6c:   punpcklqdq      xmm8, xmm1\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299884, 'fD\x0flÁ')
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 131460795428267414436756952)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 131460795428267414436756952)
        cpu.RIP = 4299884
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299888, 8) == ord('Á'))
        condition = Operators.AND(condition, cpu.read_int(4299884, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299885, 8) == ord('D'))
        condition = Operators.AND(condition, cpu.read_int(4299886, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299887, 8) == ord('l'))
        condition = Operators.AND(condition, cpu.XMM8 == 131460500280362235083931096)
        condition = Operators.AND(condition, cpu.XMM1 == 131460795428267414436756952)
        condition = Operators.AND(condition, cpu.RIP == 4299889)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLQDQ_13_symbolic(self):
        if False:
            while True:
                i = 10
        'Instruction PUNPCKLQDQ_13\n        Groups: sse2\n        0x419c6c:   punpcklqdq      xmm8, xmm1\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299884, 'fD\x0flÁ')
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 131465517794750284081970904)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 131465517794750284081970904)
        cpu.RIP = 4299884
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299888, 8) == ord('Á'))
        condition = Operators.AND(condition, cpu.read_int(4299884, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299885, 8) == ord('D'))
        condition = Operators.AND(condition, cpu.read_int(4299886, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299887, 8) == ord('l'))
        condition = Operators.AND(condition, cpu.XMM8 == 131465222646845104729145048)
        condition = Operators.AND(condition, cpu.XMM1 == 131465517794750284081970904)
        condition = Operators.AND(condition, cpu.RIP == 4299889)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLQDQ_14_symbolic(self):
        if False:
            return 10
        'Instruction PUNPCKLQDQ_14\n        Groups: sse2\n        0x419c6c:   punpcklqdq      xmm8, xmm1\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299884, 'fD\x0flÁ')
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 131450170103680957735025560)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 131450170103680957735025560)
        cpu.RIP = 4299884
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299888, 8) == ord('Á'))
        condition = Operators.AND(condition, cpu.read_int(4299884, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299885, 8) == ord('D'))
        condition = Operators.AND(condition, cpu.read_int(4299886, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299887, 8) == ord('l'))
        condition = Operators.AND(condition, cpu.XMM8 == 131449874955775778382199704)
        condition = Operators.AND(condition, cpu.XMM1 == 131450170103680957735025560)
        condition = Operators.AND(condition, cpu.RIP == 4299889)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLQDQ_15_symbolic(self):
        if False:
            while True:
                i = 10
        'Instruction PUNPCKLQDQ_15\n        Groups: sse2\n        0x419c82:   punpcklqdq      xmm1, xmm0\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299906, 'f\x0flÈ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 131455482765974186085891256)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 131455482765974186085891256)
        cpu.RIP = 4299906
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299906, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299907, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299908, 8) == ord('l'))
        condition = Operators.AND(condition, cpu.read_int(4299909, 8) == ord('È'))
        condition = Operators.AND(condition, cpu.XMM0 == 131455482765974186085891256)
        condition = Operators.AND(condition, cpu.XMM1 == 131455187618069006733065400)
        condition = Operators.AND(condition, cpu.RIP == 4299910)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLQDQ_16_symbolic(self):
        if False:
            while True:
                i = 10
        'Instruction PUNPCKLQDQ_16\n        Groups: sse2\n        0x419c82:   punpcklqdq      xmm1, xmm0\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299906, 'f\x0flÈ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 131461385724077773142408696)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 131461385724077773142408696)
        cpu.RIP = 4299906
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299906, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299907, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299908, 8) == ord('l'))
        condition = Operators.AND(condition, cpu.read_int(4299909, 8) == ord('È'))
        condition = Operators.AND(condition, cpu.XMM0 == 131461385724077773142408696)
        condition = Operators.AND(condition, cpu.XMM1 == 131461090576172593789582840)
        condition = Operators.AND(condition, cpu.RIP == 4299910)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLQDQ_17_symbolic(self):
        if False:
            return 10
        'Instruction PUNPCKLQDQ_17\n        Groups: sse2\n        0x419c82:   punpcklqdq      xmm1, xmm0\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299906, 'f\x0flÈ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 131464927498939925376319160)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 131464927498939925376319160)
        cpu.RIP = 4299906
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299906, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299907, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299908, 8) == ord('l'))
        condition = Operators.AND(condition, cpu.read_int(4299909, 8) == ord('È'))
        condition = Operators.AND(condition, cpu.XMM0 == 131464927498939925376319160)
        condition = Operators.AND(condition, cpu.XMM1 == 131464632351034746023493304)
        condition = Operators.AND(condition, cpu.RIP == 4299910)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLQDQ_18_symbolic(self):
        if False:
            while True:
                i = 10
        'Instruction PUNPCKLQDQ_18\n        Groups: sse2\n        0x419c6c:   punpcklqdq      xmm8, xmm1\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299884, 'fD\x0flÁ')
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 131451350695301675146329048)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 131451350695301675146329048)
        cpu.RIP = 4299884
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299888, 8) == ord('Á'))
        condition = Operators.AND(condition, cpu.read_int(4299884, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299885, 8) == ord('D'))
        condition = Operators.AND(condition, cpu.read_int(4299886, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299887, 8) == ord('l'))
        condition = Operators.AND(condition, cpu.XMM8 == 131451055547396495793503192)
        condition = Operators.AND(condition, cpu.XMM1 == 131451350695301675146329048)
        condition = Operators.AND(condition, cpu.RIP == 4299889)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLQDQ_19_symbolic(self):
        if False:
            print('Hello World!')
        'Instruction PUNPCKLQDQ_19\n        Groups: sse2\n        0x419c82:   punpcklqdq      xmm1, xmm0\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299906, 'f\x0flÈ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 131454302174353468674587768)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 131454302174353468674587768)
        cpu.RIP = 4299906
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299906, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299907, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299908, 8) == ord('l'))
        condition = Operators.AND(condition, cpu.read_int(4299909, 8) == ord('È'))
        condition = Operators.AND(condition, cpu.XMM0 == 131454302174353468674587768)
        condition = Operators.AND(condition, cpu.XMM1 == 131454007026448289321761912)
        condition = Operators.AND(condition, cpu.RIP == 4299910)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLQDQ_2_symbolic(self):
        if False:
            while True:
                i = 10
        'Instruction PUNPCKLQDQ_2\n        Groups: sse2\n        0x419c6c:   punpcklqdq      xmm8, xmm1\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299884, 'fD\x0flÁ')
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 131467878977991718904577880)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 131467878977991718904577880)
        cpu.RIP = 4299884
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299888, 8) == ord('Á'))
        condition = Operators.AND(condition, cpu.read_int(4299884, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299885, 8) == ord('D'))
        condition = Operators.AND(condition, cpu.read_int(4299886, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299887, 8) == ord('l'))
        condition = Operators.AND(condition, cpu.XMM8 == 131467583830086539551752024)
        condition = Operators.AND(condition, cpu.XMM1 == 131467878977991718904577880)
        condition = Operators.AND(condition, cpu.RIP == 4299889)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLQDQ_20_symbolic(self):
        if False:
            i = 10
            return i + 15
        'Instruction PUNPCKLQDQ_20\n        Groups: sse2\n        0x419c82:   punpcklqdq      xmm1, xmm0\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299906, 'f\x0flÈ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 131444857441387729384159864)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 131444857441387729384159864)
        cpu.RIP = 4299906
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299906, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299907, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299908, 8) == ord('l'))
        condition = Operators.AND(condition, cpu.read_int(4299909, 8) == ord('È'))
        condition = Operators.AND(condition, cpu.XMM0 == 131444857441387729384159864)
        condition = Operators.AND(condition, cpu.XMM1 == 131444562293482550031334008)
        condition = Operators.AND(condition, cpu.RIP == 4299910)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLQDQ_21_symbolic(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKLQDQ_21\n        Groups: sse2\n        0x419c6c:   punpcklqdq      xmm8, xmm1\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299884, 'fD\x0flÁ')
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 131439544779094501033294168)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 131439544779094501033294168)
        cpu.RIP = 4299884
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299888, 8) == ord('Á'))
        condition = Operators.AND(condition, cpu.read_int(4299884, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299885, 8) == ord('D'))
        condition = Operators.AND(condition, cpu.read_int(4299886, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299887, 8) == ord('l'))
        condition = Operators.AND(condition, cpu.XMM8 == 131439249631189321680468312)
        condition = Operators.AND(condition, cpu.XMM1 == 131439544779094501033294168)
        condition = Operators.AND(condition, cpu.RIP == 4299889)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLQDQ_3_symbolic(self):
        if False:
            while True:
                i = 10
        'Instruction PUNPCKLQDQ_3\n        Groups: sse2\n        0x419c6c:   punpcklqdq      xmm8, xmm1\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299884, 'fD\x0flÁ')
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 131447808920439522912418584)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 131447808920439522912418584)
        cpu.RIP = 4299884
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299888, 8) == ord('Á'))
        condition = Operators.AND(condition, cpu.read_int(4299884, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299885, 8) == ord('D'))
        condition = Operators.AND(condition, cpu.read_int(4299886, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299887, 8) == ord('l'))
        condition = Operators.AND(condition, cpu.XMM8 == 131447513772534343559592728)
        condition = Operators.AND(condition, cpu.XMM1 == 131447808920439522912418584)
        condition = Operators.AND(condition, cpu.RIP == 4299889)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLQDQ_4_symbolic(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKLQDQ_4\n        Groups: sse2\n        0x419c6c:   punpcklqdq      xmm8, xmm1\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299884, 'fD\x0flÁ')
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 131436003004232348799383704)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 131436003004232348799383704)
        cpu.RIP = 4299884
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299888, 8) == ord('Á'))
        condition = Operators.AND(condition, cpu.read_int(4299884, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299885, 8) == ord('D'))
        condition = Operators.AND(condition, cpu.read_int(4299886, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299887, 8) == ord('l'))
        condition = Operators.AND(condition, cpu.XMM8 == 131435707856327169446557848)
        condition = Operators.AND(condition, cpu.XMM1 == 131436003004232348799383704)
        condition = Operators.AND(condition, cpu.RIP == 4299889)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLQDQ_5_symbolic(self):
        if False:
            i = 10
            return i + 15
        'Instruction PUNPCKLQDQ_5\n        Groups: sse2\n        0x419c6c:   punpcklqdq      xmm8, xmm1\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299884, 'fD\x0flÁ')
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 131452531286922392557632536)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 131452531286922392557632536)
        cpu.RIP = 4299884
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299888, 8) == ord('Á'))
        condition = Operators.AND(condition, cpu.read_int(4299884, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299885, 8) == ord('D'))
        condition = Operators.AND(condition, cpu.read_int(4299886, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299887, 8) == ord('l'))
        condition = Operators.AND(condition, cpu.XMM8 == 131452236139017213204806680)
        condition = Operators.AND(condition, cpu.XMM1 == 131452531286922392557632536)
        condition = Operators.AND(condition, cpu.RIP == 4299889)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLQDQ_6_symbolic(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKLQDQ_6\n        Groups: sse2\n        0x419c82:   punpcklqdq      xmm1, xmm0\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299906, 'f\x0flÈ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 131467288682181360198926136)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 131467288682181360198926136)
        cpu.RIP = 4299906
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299906, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299907, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299908, 8) == ord('l'))
        condition = Operators.AND(condition, cpu.read_int(4299909, 8) == ord('È'))
        condition = Operators.AND(condition, cpu.XMM0 == 131467288682181360198926136)
        condition = Operators.AND(condition, cpu.XMM1 == 131466993534276180846100280)
        condition = Operators.AND(condition, cpu.RIP == 4299910)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLQDQ_7_symbolic(self):
        if False:
            print('Hello World!')
        'Instruction PUNPCKLQDQ_7\n        Groups: sse2\n        0x419c6c:   punpcklqdq      xmm8, xmm1\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299884, 'fD\x0flÁ')
        cpu.XMM8 = cs.new_bitvec(128)
        cs.add(cpu.XMM8 == 131444267145577370678508120)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 131444267145577370678508120)
        cpu.RIP = 4299884
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299888, 8) == ord('Á'))
        condition = Operators.AND(condition, cpu.read_int(4299884, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299885, 8) == ord('D'))
        condition = Operators.AND(condition, cpu.read_int(4299886, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299887, 8) == ord('l'))
        condition = Operators.AND(condition, cpu.XMM8 == 131443971997672191325682264)
        condition = Operators.AND(condition, cpu.XMM1 == 131444267145577370678508120)
        condition = Operators.AND(condition, cpu.RIP == 4299889)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLQDQ_8_symbolic(self):
        if False:
            for i in range(10):
                print('nop')
        'Instruction PUNPCKLQDQ_8\n        Groups: sse2\n        0x419c82:   punpcklqdq      xmm1, xmm0\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299906, 'f\x0flÈ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 131442496258146294561552888)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 131442496258146294561552888)
        cpu.RIP = 4299906
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299906, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299907, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299908, 8) == ord('l'))
        condition = Operators.AND(condition, cpu.read_int(4299909, 8) == ord('È'))
        condition = Operators.AND(condition, cpu.XMM0 == 131442496258146294561552888)
        condition = Operators.AND(condition, cpu.XMM1 == 131442201110241115208727032)
        condition = Operators.AND(condition, cpu.RIP == 4299910)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLQDQ_9_symbolic(self):
        if False:
            print('Hello World!')
        'Instruction PUNPCKLQDQ_9\n        Groups: sse2\n        0x419c82:   punpcklqdq      xmm1, xmm0\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4296704, 4096, 'rwx')
        mem.write(4299906, 'f\x0flÈ')
        cpu.XMM0 = cs.new_bitvec(128)
        cs.add(cpu.XMM0 == 131460205132457055731105208)
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 131460205132457055731105208)
        cpu.RIP = 4299906
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4299906, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4299907, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.read_int(4299908, 8) == ord('l'))
        condition = Operators.AND(condition, cpu.read_int(4299909, 8) == ord('È'))
        condition = Operators.AND(condition, cpu.XMM0 == 131460205132457055731105208)
        condition = Operators.AND(condition, cpu.XMM1 == 131459909984551876378279352)
        condition = Operators.AND(condition, cpu.RIP == 4299910)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))

    def test_PUNPCKLWD_1_symbolic(self):
        if False:
            i = 10
            return i + 15
        'Instruction PUNPCKLWD_1\n        Groups: sse2\n        0x4668b6:   punpcklwd       xmm1, xmm1\n        '
        cs = ConstraintSet()
        mem = SMemory64(cs)
        cpu = AMD64Cpu(mem)
        mem.mmap(4612096, 4096, 'rwx')
        mem.write(4614326, 'f\x0faÉ')
        cpu.XMM1 = cs.new_bitvec(128)
        cs.add(cpu.XMM1 == 12079)
        cpu.RIP = 4614326
        done = False
        while not done:
            try:
                cpu.execute()
                done = True
            except ConcretizeRegister as e:
                symbol = getattr(cpu, e.reg_name)
                values = solver.get_all_values(cs, symbol)
                self.assertEqual(len(values), 1)
                setattr(cpu, e.reg_name, values[0])
        condition = True
        condition = Operators.AND(condition, cpu.read_int(4614328, 8) == ord('a'))
        condition = Operators.AND(condition, cpu.read_int(4614329, 8) == ord('É'))
        condition = Operators.AND(condition, cpu.read_int(4614326, 8) == ord('f'))
        condition = Operators.AND(condition, cpu.read_int(4614327, 8) == ord('\x0f'))
        condition = Operators.AND(condition, cpu.XMM1 == 791621423)
        condition = Operators.AND(condition, cpu.RIP == 4614330)
        with cs as temp_cs:
            temp_cs.add(condition)
            self.assertTrue(solver.check(temp_cs))
        with cs as temp_cs:
            temp_cs.add(condition == False)
            self.assertFalse(solver.check(temp_cs))
if __name__ == '__main__':
    unittest.main()