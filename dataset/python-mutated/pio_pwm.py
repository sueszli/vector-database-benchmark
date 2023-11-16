from machine import Pin
from rp2 import PIO, StateMachine, asm_pio
from time import sleep

@asm_pio(sideset_init=PIO.OUT_LOW)
def pwm_prog():
    if False:
        return 10
    pull(noblock).side(0)
    mov(x, osr)
    mov(y, isr)
    label('pwmloop')
    jmp(x_not_y, 'skip')
    nop().side(1)
    label('skip')
    jmp(y_dec, 'pwmloop')

class PIOPWM:

    def __init__(self, sm_id, pin, max_count, count_freq):
        if False:
            while True:
                i = 10
        self._sm = StateMachine(sm_id, pwm_prog, freq=2 * count_freq, sideset_base=Pin(pin))
        self._sm.put(max_count)
        self._sm.exec('pull()')
        self._sm.exec('mov(isr, osr)')
        self._sm.active(1)
        self._max_count = max_count

    def set(self, value):
        if False:
            return 10
        value = max(value, -1)
        value = min(value, self._max_count)
        self._sm.put(value)
pwm = PIOPWM(0, 25, max_count=(1 << 16) - 1, count_freq=10000000)
while True:
    for i in range(256):
        pwm.set(i ** 2)
        sleep(0.01)