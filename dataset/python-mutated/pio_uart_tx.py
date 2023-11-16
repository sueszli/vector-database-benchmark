from machine import Pin
from rp2 import PIO, StateMachine, asm_pio
UART_BAUD = 115200
PIN_BASE = 10
NUM_UARTS = 8

@asm_pio(sideset_init=PIO.OUT_HIGH, out_init=PIO.OUT_HIGH, out_shiftdir=PIO.SHIFT_RIGHT)
def uart_tx():
    if False:
        print('Hello World!')
    pull()
    set(x, 7).side(0)[7]
    label('bitloop')
    out(pins, 1)[6]
    jmp(x_dec, 'bitloop')
    nop().side(1)[6]
uarts = []
for i in range(NUM_UARTS):
    sm = StateMachine(i, uart_tx, freq=8 * UART_BAUD, sideset_base=Pin(PIN_BASE + i), out_base=Pin(PIN_BASE + i))
    sm.active(1)
    uarts.append(sm)

def pio_uart_print(sm, s):
    if False:
        print('Hello World!')
    for c in s:
        sm.put(ord(c))
for (i, u) in enumerate(uarts):
    pio_uart_print(u, 'Hello from UART {}!\n'.format(i))