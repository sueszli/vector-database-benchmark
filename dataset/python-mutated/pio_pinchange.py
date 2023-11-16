import time
from machine import Pin
import rp2

@rp2.asm_pio()
def wait_pin_low():
    if False:
        print('Hello World!')
    wrap_target()
    wait(0, pin, 0)
    irq(block, rel(0))
    wait(1, pin, 0)
    wrap()

def handler(sm):
    if False:
        while True:
            i = 10
    print(time.ticks_ms(), sm)
pin16 = Pin(16, Pin.IN, Pin.PULL_UP)
sm0 = rp2.StateMachine(0, wait_pin_low, in_base=pin16)
sm0.irq(handler)
pin17 = Pin(17, Pin.IN, Pin.PULL_UP)
sm1 = rp2.StateMachine(1, wait_pin_low, in_base=pin17)
sm1.irq(handler)
sm0.active(1)
sm1.active(1)