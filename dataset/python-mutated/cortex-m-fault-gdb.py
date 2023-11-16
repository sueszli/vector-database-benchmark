"""Source this file into gdb `source ../../tools/cortex-m-fault-gdb.py` then run
   `cortex-m-fault` to print basic info about the fault registers."""
SCS = 3758153728
SCB = SCS + 3328
CPUID = SCB + 0
ICSR = SCB + 4
VTOR = SCB + 8
AIRCR = SCB + 12
SCR = SCB + 16
CCR = SCB + 20
SHCSR = SCB + 36
CFSR = SCB + 40
HFSR = SCB + 44
DFSR = SCB + 48
MMFAR = SCB + 52
BFAR = SCB + 56
AFSR = SCB + 60
PARTS = {3111: 'Cortex M7', 3168: 'Cortex M0+'}
EXCEPTIONS = {0: 'Thread mode', 2: 'Non Maskable Interrupt', 3: 'Hard Fault', 4: 'MemManage Fault', 5: 'Bus Fault', 6: 'Usage Fault', 11: 'SVCAll', 14: 'PendSV', 15: 'SysTick'}

class CortexMFault(gdb.Command):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(CortexMFault, self).__init__('cortex-m-fault', gdb.COMMAND_USER)

    def _read(self, address):
        if False:
            while True:
                i = 10
        i = gdb.selected_inferior()
        return i.read_memory(address, 4).cast('I')[0]

    def _armv6m_fault(self):
        if False:
            while True:
                i = 10
        vtor = self._read(VTOR)
        print('vtor', hex(vtor))
        icsr = self._read(ICSR)
        vectactive = icsr & 511
        print('icsr', hex(icsr), vectactive)
        if vectactive != 0:
            if vectactive in EXCEPTIONS:
                vectactive = EXCEPTIONS[vectactive]
            else:
                vectactive -= 16
            print('Active interrupt:', vectactive)
        vectpending = icsr >> 12 & 511
        if vectpending != 0:
            if vectpending in EXCEPTIONS:
                vectpending = EXCEPTIONS[vectpending]
            else:
                vectpending -= 16
            print('Pending interrupt:', vectpending)

    def _armv7m_fault(self):
        if False:
            print('Hello World!')
        icsr = self._read(ICSR)
        if icsr & 1 << 11 != 0:
            print('No preempted exceptions')
        else:
            print('Another exception was preempted')
        print('icsr', hex(icsr))
        vectactive = icsr & 511
        if vectactive != 0:
            if vectactive in EXCEPTIONS:
                print(EXCEPTIONS[vectactive])
            else:
                print(vectactive - 16)
        vectpending = icsr >> 12 & 511
        if vectpending != 0:
            if vectpending in EXCEPTIONS:
                print(EXCEPTIONS[vectpending])
            else:
                print(vectpending - 16)
        vtor = self._read(VTOR)
        print('vtor', hex(vtor))
        cfsr = self._read(CFSR)
        ufsr = cfsr >> 16
        bfsr = cfsr >> 8 & 255
        mmfsr = cfsr & 255
        print('ufsr', hex(ufsr), 'bfsr', hex(bfsr), 'mmfsr', hex(mmfsr))
        if bfsr & 1 << 7 != 0:
            print('Bad address', hex(self._read(BFAR)))
        if bfsr & 1 << 3 != 0:
            print('Unstacking from exception error')
        if bfsr & 1 << 2 != 0:
            print('Imprecise data bus error')
        if bfsr & 1 << 1 != 0:
            print('Precise data bus error')
        if bfsr & 1 << 0 != 0:
            print('Instruction bus error')
        if mmfsr & 1 << 7 != 0:
            print('Bad address', hex(self._read(MMFAR)))
        if mmfsr & 1 << 3 != 0:
            print('Unstacking from exception error')
        if mmfsr & 1 << 1 != 0:
            print('Data access violation')
        if mmfsr & 1 << 0 != 0:
            print('Instruction access violation')
        if ufsr & 1 << 8 != 0:
            print('Unaligned access')
        if ufsr & 1 << 0 != 0:
            print('Undefined instruction')
        hfsr = self._read(HFSR)
        if hfsr & 1 << 30 != 0:
            print('Forced hard fault')
        if hfsr & 1 << 1 != 0:
            print('Bus fault when reading vector table')
            print('VTOR', hex(vtor))

    def invoke(self, arg, from_tty):
        if False:
            i = 10
            return i + 15
        cpuid = self._read(CPUID)
        implementer = cpuid >> 24
        if implementer != 65:
            raise RuntimeError()
        variant = cpuid >> 20 & 15
        architecture = cpuid >> 16 & 15
        revision = cpuid & 15
        part_no = cpuid >> 4 & 4095
        print(PARTS[part_no])
        if architecture == 15:
            self._armv7m_fault()
        elif architecture == 12:
            self._armv6m_fault()
        else:
            raise RuntimeError(f'Unknown architecture {architecture:x}')
CortexMFault()