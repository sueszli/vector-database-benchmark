class Timer:

    def __init__(self):
        if False:
            return 10
        self.DIV = 0
        self.TIMA = 0
        self.DIV_counter = 0
        self.TIMA_counter = 0
        self.TMA = 0
        self.TAC = 0
        self.dividers = [1024, 16, 64, 256]

    def reset(self):
        if False:
            return 10
        self.DIV_counter = 0
        self.TIMA_counter = 0
        self.DIV = 0

    def tick(self, cycles):
        if False:
            print('Hello World!')
        self.DIV_counter += cycles
        self.DIV += self.DIV_counter >> 8
        self.DIV_counter &= 255
        self.DIV &= 255
        if self.TAC & 4 == 0:
            return False
        self.TIMA_counter += cycles
        divider = self.dividers[self.TAC & 3]
        if self.TIMA_counter >= divider:
            self.TIMA_counter -= divider
            self.TIMA += 1
            if self.TIMA > 255:
                self.TIMA = self.TMA
                self.TIMA &= 255
                return True
        return False

    def cycles_to_interrupt(self):
        if False:
            i = 10
            return i + 15
        if self.TAC & 4 == 0:
            return 1 << 16
        divider = self.dividers[self.TAC & 3]
        cyclesleft = (256 - self.TIMA) * divider - self.TIMA_counter
        return cyclesleft

    def save_state(self, f):
        if False:
            return 10
        f.write(self.DIV)
        f.write(self.TIMA)
        f.write_16bit(self.DIV_counter)
        f.write_16bit(self.TIMA_counter)
        f.write(self.TMA)
        f.write(self.TAC)

    def load_state(self, f, state_version):
        if False:
            i = 10
            return i + 15
        self.DIV = f.read()
        self.TIMA = f.read()
        self.DIV_counter = f.read_16bit()
        self.TIMA_counter = f.read_16bit()
        self.TMA = f.read()
        self.TAC = f.read()