import _thread
from machine import Pin, UART
from rp2 import PIO, StateMachine, asm_pio
UART_BAUD = 9600
HARD_UART_TX_PIN = Pin(4, Pin.OUT)
PIO_RX_PIN = Pin(3, Pin.IN, Pin.PULL_UP)

@asm_pio(autopush=True, push_thresh=8, in_shiftdir=rp2.PIO.SHIFT_RIGHT, fifo_join=PIO.JOIN_RX)
def uart_rx_mini():
    if False:
        print('Hello World!')
    wait(0, pin, 0)
    set(x, 7)[10]
    label('bitloop')
    in_(pins, 1)
    jmp(x_dec, 'bitloop')[6]

@asm_pio(in_shiftdir=rp2.PIO.SHIFT_RIGHT)
def uart_rx():
    if False:
        print('Hello World!')
    label('start')
    wait(0, pin, 0)
    set(x, 7)[10]
    label('bitloop')
    in_(pins, 1)
    jmp(x_dec, 'bitloop')[6]
    jmp(pin, 'good_stop')
    irq(block, 4)
    wait(1, pin, 0)
    jmp('start')
    label('good_stop')
    push(block)

def handler(sm):
    if False:
        while True:
            i = 10
    print('break', time.ticks_ms(), end=' ')

def core1_task(uart, text):
    if False:
        i = 10
        return i + 15
    uart.write(text)
uart = UART(1, UART_BAUD, tx=HARD_UART_TX_PIN)
for pio_prog in ('uart_rx_mini', 'uart_rx'):
    sm = StateMachine(0, globals()[pio_prog], freq=8 * UART_BAUD, in_base=PIO_RX_PIN, jmp_pin=PIO_RX_PIN)
    sm.irq(handler)
    sm.active(1)
    text = 'Hello, world from PIO, using {}!'.format(pio_prog)
    _thread.start_new_thread(core1_task, (uart, text))
    for i in range(len(text)):
        print(chr(sm.get() >> 24), end='')
    print()