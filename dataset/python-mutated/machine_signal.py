try:
    import machine
    machine.PinBase
    machine.Signal
except:
    print('SKIP')
    raise SystemExit

class Pin(machine.PinBase):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.v = 0

    def value(self, v=None):
        if False:
            return 10
        if v is None:
            return self.v
        else:
            self.v = int(v)
p = Pin()
s = machine.Signal(p)
s.value(0)
print(p.value(), s.value())
s.value(1)
print(p.value(), s.value())
p = Pin()
s = machine.Signal(p, invert=True)
s.off()
print(p.value(), s.value())
s.on()
print(p.value(), s.value())