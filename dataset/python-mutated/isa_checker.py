from cpuinfo import CPUID

class ISAChecker:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        cpuid = CPUID()
        self.flags = cpuid.get_flags(cpuid.get_max_extension_support())
        if self._avx_vnni(cpuid):
            self.flags.append('avxvnni')

    def _avx_vnni(self, cpuid):
        if False:
            i = 10
            return i + 15
        eax = cpuid._run_asm(b'\xb9\x01\x00\x00\x00', b'\xb8\x07\x00\x00\x00', b'\x0f\xa2', b'\xc3')
        return 16 & eax != 0

    def check_avx(self):
        if False:
            print('Hello World!')
        return 'avx' in self.flags

    def check_avx2(self):
        if False:
            while True:
                i = 10
        return 'avx2' in self.flags

    def check_avx_vnni(self):
        if False:
            i = 10
            return i + 15
        return 'avxvnni' in self.flags

    def check_avx512(self):
        if False:
            i = 10
            return i + 15
        return 'avx512f' in self.flags and 'avx512bw' in self.flags and ('avx512cd' in self.flags) and ('avx512dq' in self.flags) and ('avx512vl' in self.flags)

    def check_avx512_vnni(self):
        if False:
            i = 10
            return i + 15
        return 'avx512vnni' in self.flags
isa_checker = ISAChecker()

def check_avx():
    if False:
        for i in range(10):
            print('nop')
    return isa_checker.check_avx()

def check_avx2():
    if False:
        i = 10
        return i + 15
    return isa_checker.check_avx2()

def check_avx_vnni():
    if False:
        i = 10
        return i + 15
    return isa_checker.check_avx_vnni() and isa_checker.check_avx2()

def check_avx512():
    if False:
        while True:
            i = 10
    return isa_checker.check_avx512()

def check_avx512_vnni():
    if False:
        print('Hello World!')
    return isa_checker.check_avx512_vnni() and isa_checker.check_avx512()

def is_server():
    if False:
        print('Hello World!')
    return check_avx512_vnni()

def is_spr():
    if False:
        i = 10
        return i + 15
    return check_avx_vnni() and check_avx512_vnni()