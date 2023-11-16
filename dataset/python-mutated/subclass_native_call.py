try:
    import machine
    machine.PinBase
    machine.Signal
except:
    print('SKIP')
    raise SystemExit

class Pin(machine.PinBase):

    def value(self, v=None):
        if False:
            i = 10
            return i + 15
        return 42

class MySignal(machine.Signal):
    pass
s = MySignal(Pin())
print(s())