from micropython import const
import time, machine, bluetooth
TIMEOUT_MS = 4000
_IRQ_CENTRAL_CONNECT = const(1)
_IRQ_CENTRAL_DISCONNECT = const(2)
_IRQ_PERIPHERAL_CONNECT = const(7)
_IRQ_PERIPHERAL_DISCONNECT = const(8)
waiting_events = {}

def irq(event, data):
    if False:
        for i in range(10):
            print('nop')
    if event == _IRQ_CENTRAL_CONNECT:
        print('_IRQ_CENTRAL_CONNECT')
        waiting_events[event] = data[0]
    elif event == _IRQ_CENTRAL_DISCONNECT:
        print('_IRQ_CENTRAL_DISCONNECT')
    elif event == _IRQ_PERIPHERAL_CONNECT:
        print('_IRQ_PERIPHERAL_CONNECT')
        waiting_events[event] = data[0]
    elif event == _IRQ_PERIPHERAL_DISCONNECT:
        print('_IRQ_PERIPHERAL_DISCONNECT')
    if event not in waiting_events:
        waiting_events[event] = None

def wait_for_event(event, timeout_ms):
    if False:
        while True:
            i = 10
    t0 = time.ticks_ms()
    while time.ticks_diff(time.ticks_ms(), t0) < timeout_ms:
        if event in waiting_events:
            return waiting_events.pop(event)
        machine.idle()
    raise ValueError('Timeout waiting for {}'.format(event))

def instance0():
    if False:
        for i in range(10):
            print('nop')
    multitest.globals(BDADDR=ble.config('mac'))
    print('gap_advertise')
    ble.gap_advertise(20000, b'\x02\x01\x06\x04\xffMPY')
    multitest.next()
    try:
        wait_for_event(_IRQ_CENTRAL_CONNECT, TIMEOUT_MS)
        wait_for_event(_IRQ_CENTRAL_DISCONNECT, TIMEOUT_MS)
        print('gap_advertise')
        ble.gap_advertise(20000, b'\x02\x01\x06\x04\xffMPY')
        conn_handle = wait_for_event(_IRQ_CENTRAL_CONNECT, TIMEOUT_MS)
        print('gap_disconnect:', ble.gap_disconnect(conn_handle))
        wait_for_event(_IRQ_CENTRAL_DISCONNECT, TIMEOUT_MS)
    finally:
        ble.active(0)

def instance1():
    if False:
        i = 10
        return i + 15
    multitest.next()
    try:
        print('gap_connect')
        ble.gap_connect(*BDADDR)
        conn_handle = wait_for_event(_IRQ_PERIPHERAL_CONNECT, TIMEOUT_MS)
        print('gap_disconnect:', ble.gap_disconnect(conn_handle))
        wait_for_event(_IRQ_PERIPHERAL_DISCONNECT, TIMEOUT_MS)
        print('gap_connect')
        ble.gap_connect(BDADDR[0], BDADDR[1], 5000)
        wait_for_event(_IRQ_PERIPHERAL_CONNECT, TIMEOUT_MS)
        wait_for_event(_IRQ_PERIPHERAL_DISCONNECT, TIMEOUT_MS)
    finally:
        ble.active(0)
ble = bluetooth.BLE()
ble.active(1)
ble.irq(irq)