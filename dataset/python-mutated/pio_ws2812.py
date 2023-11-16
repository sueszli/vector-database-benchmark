import array, time
from machine import Pin
import rp2
NUM_LEDS = 8

@rp2.asm_pio(sideset_init=rp2.PIO.OUT_LOW, out_shiftdir=rp2.PIO.SHIFT_LEFT, autopull=True, pull_thresh=24)
def ws2812():
    if False:
        while True:
            i = 10
    T1 = 2
    T2 = 5
    T3 = 3
    wrap_target()
    label('bitloop')
    out(x, 1).side(0)[T3 - 1]
    jmp(not_x, 'do_zero').side(1)[T1 - 1]
    jmp('bitloop').side(1)[T2 - 1]
    label('do_zero')
    nop().side(0)[T2 - 1]
    wrap()
sm = rp2.StateMachine(0, ws2812, freq=8000000, sideset_base=Pin(22))
sm.active(1)
ar = array.array('I', [0 for _ in range(NUM_LEDS)])
for i in range(4 * NUM_LEDS):
    for j in range(NUM_LEDS):
        r = j * 100 // (NUM_LEDS - 1)
        b = 100 - j * 100 // (NUM_LEDS - 1)
        if j != i % NUM_LEDS:
            r >>= 3
            b >>= 3
        ar[j] = r << 16 | b
    sm.put(ar, 8)
    time.sleep_ms(50)
for i in range(24):
    for j in range(NUM_LEDS):
        ar[j] = ar[j] >> 1 & 8355711
    sm.put(ar, 8)
    time.sleep_ms(50)