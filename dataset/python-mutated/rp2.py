from _rp2 import *
from micropython import const
_PROG_DATA = const(0)
_PROG_OFFSET_PIO0 = const(1)
_PROG_OFFSET_PIO1 = const(2)
_PROG_EXECCTRL = const(3)
_PROG_SHIFTCTRL = const(4)
_PROG_OUT_PINS = const(5)
_PROG_SET_PINS = const(6)
_PROG_SIDESET_PINS = const(7)
_PROG_MAX_FIELDS = const(8)

class PIOASMError(Exception):
    pass

class PIOASMEmit:

    def __init__(self, *, out_init=None, set_init=None, sideset_init=None, in_shiftdir=0, out_shiftdir=0, autopush=False, autopull=False, push_thresh=32, pull_thresh=32, fifo_join=0):
        if False:
            return 10
        from array import array
        self.labels = {}
        execctrl = 0
        shiftctrl = fifo_join << 30 | (pull_thresh & 31) << 25 | (push_thresh & 31) << 20 | out_shiftdir << 19 | in_shiftdir << 18 | autopull << 17 | autopush << 16
        self.prog = [array('H'), -1, -1, execctrl, shiftctrl, out_init, set_init, sideset_init]
        self.wrap_used = False
        if sideset_init is None:
            self.sideset_count = 0
        elif isinstance(sideset_init, int):
            self.sideset_count = 1
        else:
            self.sideset_count = len(sideset_init)

    def start_pass(self, pass_):
        if False:
            i = 10
            return i + 15
        if pass_ == 1:
            if not self.wrap_used and self.num_instr:
                self.wrap()
            self.delay_max = 31
            if self.sideset_count:
                self.sideset_opt = self.num_sideset != self.num_instr
                if self.sideset_opt:
                    self.prog[_PROG_EXECCTRL] |= 1 << 30
                    self.sideset_count += 1
                self.delay_max >>= self.sideset_count
        self.pass_ = pass_
        self.num_instr = 0
        self.num_sideset = 0

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        return self.delay(key)

    def delay(self, delay):
        if False:
            return 10
        if self.pass_ > 0:
            if delay > self.delay_max:
                raise PIOASMError('delay too large')
            self.prog[_PROG_DATA][-1] |= delay << 8
        return self

    def side(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.num_sideset += 1
        if self.pass_ > 0:
            if self.sideset_count == 0:
                raise PIOASMError('no sideset')
            elif value >= 1 << self.sideset_count:
                raise PIOASMError('sideset too large')
            set_bit = 13 - self.sideset_count
            self.prog[_PROG_DATA][-1] |= self.sideset_opt << 12 | value << set_bit
        return self

    def wrap_target(self):
        if False:
            return 10
        self.prog[_PROG_EXECCTRL] |= self.num_instr << 7

    def wrap(self):
        if False:
            i = 10
            return i + 15
        assert self.num_instr
        self.prog[_PROG_EXECCTRL] |= self.num_instr - 1 << 12
        self.wrap_used = True

    def label(self, label):
        if False:
            for i in range(10):
                print('nop')
        if self.pass_ == 0:
            if label in self.labels:
                raise PIOASMError('duplicate label {}'.format(label))
            self.labels[label] = self.num_instr

    def word(self, instr, label=None):
        if False:
            for i in range(10):
                print('nop')
        self.num_instr += 1
        if self.pass_ > 0:
            if label is None:
                label = 0
            else:
                if label not in self.labels:
                    raise PIOASMError('unknown label {}'.format(label))
                label = self.labels[label]
            self.prog[_PROG_DATA].append(instr | label)
        return self

    def nop(self):
        if False:
            for i in range(10):
                print('nop')
        return self.word(41026)

    def jmp(self, cond, label=None):
        if False:
            while True:
                i = 10
        if label is None:
            label = cond
            cond = 0
        return self.word(0 | cond << 5, label)

    def wait(self, polarity, src, index):
        if False:
            for i in range(10):
                print('nop')
        if src == 6:
            src = 1
        elif src != 0:
            src = 2
        return self.word(8192 | polarity << 7 | src << 5 | index)

    def in_(self, src, data):
        if False:
            for i in range(10):
                print('nop')
        if not 0 < data <= 32:
            raise PIOASMError('invalid bit count {}'.format(data))
        return self.word(16384 | src << 5 | data & 31)

    def out(self, dest, data):
        if False:
            while True:
                i = 10
        if dest == 8:
            dest = 7
        if not 0 < data <= 32:
            raise PIOASMError('invalid bit count {}'.format(data))
        return self.word(24576 | dest << 5 | data & 31)

    def push(self, value=0, value2=0):
        if False:
            while True:
                i = 10
        value |= value2
        if not value & 1:
            value |= 32
        return self.word(32768 | value & 96)

    def pull(self, value=0, value2=0):
        if False:
            for i in range(10):
                print('nop')
        value |= value2
        if not value & 1:
            value |= 32
        return self.word(32896 | value & 96)

    def mov(self, dest, src):
        if False:
            i = 10
            return i + 15
        if dest == 8:
            dest = 4
        return self.word(40960 | dest << 5 | src)

    def irq(self, mod, index=None):
        if False:
            for i in range(10):
                print('nop')
        if index is None:
            index = mod
            mod = 0
        return self.word(49152 | mod & 96 | index)

    def set(self, dest, data):
        if False:
            for i in range(10):
                print('nop')
        return self.word(57344 | dest << 5 | data)
_pio_funcs = {'gpio': 0, 'pins': 0, 'x': 1, 'y': 2, 'null': 3, 'pindirs': 4, 'pc': 5, 'status': 5, 'isr': 6, 'osr': 7, 'exec': 8, 'invert': lambda x: x | 8, 'reverse': lambda x: x | 16, 'not_x': 1, 'x_dec': 2, 'not_y': 3, 'y_dec': 4, 'x_not_y': 5, 'pin': 6, 'not_osre': 7, 'noblock': 1, 'block': 33, 'iffull': 64, 'ifempty': 64, 'clear': 64, 'rel': lambda x: x | 16, 'wrap_target': None, 'wrap': None, 'label': None, 'word': None, 'nop': None, 'jmp': None, 'wait': None, 'in_': None, 'out': None, 'push': None, 'pull': None, 'mov': None, 'irq': None, 'set': None}

def asm_pio(**kw):
    if False:
        for i in range(10):
            print('nop')
    emit = PIOASMEmit(**kw)

    def dec(f):
        if False:
            i = 10
            return i + 15
        nonlocal emit
        gl = _pio_funcs
        gl['wrap_target'] = emit.wrap_target
        gl['wrap'] = emit.wrap
        gl['label'] = emit.label
        gl['word'] = emit.word
        gl['nop'] = emit.nop
        gl['jmp'] = emit.jmp
        gl['wait'] = emit.wait
        gl['in_'] = emit.in_
        gl['out'] = emit.out
        gl['push'] = emit.push
        gl['pull'] = emit.pull
        gl['mov'] = emit.mov
        gl['irq'] = emit.irq
        gl['set'] = emit.set
        old_gl = f.__globals__.copy()
        f.__globals__.clear()
        f.__globals__.update(gl)
        emit.start_pass(0)
        f()
        emit.start_pass(1)
        f()
        f.__globals__.clear()
        f.__globals__.update(old_gl)
        return emit.prog
    return dec

def asm_pio_encode(instr, sideset_count, sideset_opt=False):
    if False:
        while True:
            i = 10
    emit = PIOASMEmit()
    emit.sideset_count = sideset_count
    emit.sideset_opt = sideset_opt != 0
    emit.delay_max = 31 >> emit.sideset_count + emit.sideset_opt
    emit.pass_ = 1
    emit.num_instr = 0
    emit.num_sideset = 0
    gl = _pio_funcs
    gl['word'] = emit.word
    gl['nop'] = emit.nop
    gl['wait'] = emit.wait
    gl['in_'] = emit.in_
    gl['out'] = emit.out
    gl['push'] = emit.push
    gl['pull'] = emit.pull
    gl['mov'] = emit.mov
    gl['irq'] = emit.irq
    gl['set'] = emit.set
    exec(instr, gl)
    if len(emit.prog[_PROG_DATA]) != 1:
        raise PIOASMError('expecting exactly 1 instruction')
    return emit.prog[_PROG_DATA][0]