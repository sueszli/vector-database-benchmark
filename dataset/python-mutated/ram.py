from array import array
from random import getrandbits
INTERNAL_RAM0 = 8 * 1024
INTERNAL_RAM0_CGB = INTERNAL_RAM0 * 4
NON_IO_INTERNAL_RAM0 = 96
IO_PORTS = 76
NON_IO_INTERNAL_RAM1 = 52
INTERNAL_RAM1 = 127

class RAM:

    def __init__(self, cgb, randomize=False):
        if False:
            i = 10
            return i + 15
        self.cgb = cgb
        self.internal_ram0 = array('B', [0] * (INTERNAL_RAM0_CGB if cgb else INTERNAL_RAM0))
        self.non_io_internal_ram0 = array('B', [0] * NON_IO_INTERNAL_RAM0)
        self.io_ports = array('B', [0] * IO_PORTS)
        self.internal_ram1 = array('B', [0] * INTERNAL_RAM1)
        self.non_io_internal_ram1 = array('B', [0] * NON_IO_INTERNAL_RAM1)
        if randomize:
            for n in range(INTERNAL_RAM0_CGB if cgb else INTERNAL_RAM0):
                self.internal_ram0[n] = getrandbits(8)
            for n in range(NON_IO_INTERNAL_RAM0):
                self.non_io_internal_ram0[n] = getrandbits(8)
            for n in range(INTERNAL_RAM1):
                self.internal_ram1[n] = getrandbits(8)
            for n in range(NON_IO_INTERNAL_RAM1):
                self.non_io_internal_ram1[n] = getrandbits(8)

    def save_state(self, f):
        if False:
            print('Hello World!')
        for n in range(INTERNAL_RAM0_CGB if self.cgb else INTERNAL_RAM0):
            f.write(self.internal_ram0[n])
        for n in range(NON_IO_INTERNAL_RAM0):
            f.write(self.non_io_internal_ram0[n])
        for n in range(IO_PORTS):
            f.write(self.io_ports[n])
        for n in range(INTERNAL_RAM1):
            f.write(self.internal_ram1[n])
        for n in range(NON_IO_INTERNAL_RAM1):
            f.write(self.non_io_internal_ram1[n])

    def load_state(self, f, state_version):
        if False:
            i = 10
            return i + 15
        for n in range(INTERNAL_RAM0_CGB if self.cgb else INTERNAL_RAM0):
            self.internal_ram0[n] = f.read()
        for n in range(NON_IO_INTERNAL_RAM0):
            self.non_io_internal_ram0[n] = f.read()
        for n in range(IO_PORTS):
            self.io_ports[n] = f.read()
        for n in range(INTERNAL_RAM1):
            self.internal_ram1[n] = f.read()
        for n in range(NON_IO_INTERNAL_RAM1):
            self.non_io_internal_ram1[n] = f.read()