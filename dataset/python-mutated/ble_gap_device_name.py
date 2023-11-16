from micropython import const
import time, machine, bluetooth
TIMEOUT_MS = 5000
_IRQ_CENTRAL_CONNECT = const(1)
_IRQ_CENTRAL_DISCONNECT = const(2)
_IRQ_PERIPHERAL_CONNECT = const(7)
_IRQ_PERIPHERAL_DISCONNECT = const(8)
_IRQ_GATTC_CHARACTERISTIC_RESULT = const(11)
_IRQ_GATTC_CHARACTERISTIC_DONE = const(12)
_IRQ_GATTC_READ_RESULT = const(15)
_IRQ_GATTC_READ_DONE = const(16)
GAP_DEVICE_NAME_UUID = bluetooth.UUID(10752)
waiting_events = {}

def irq(event, data):
    if False:
        return 10
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
    elif event == _IRQ_GATTC_CHARACTERISTIC_RESULT:
        if data[-1] == GAP_DEVICE_NAME_UUID:
            print('_IRQ_GATTC_CHARACTERISTIC_RESULT', data[-1])
            waiting_events[event] = data[2]
        else:
            return
    elif event == _IRQ_GATTC_CHARACTERISTIC_DONE:
        print('_IRQ_GATTC_CHARACTERISTIC_DONE')
    elif event == _IRQ_GATTC_READ_RESULT:
        print('_IRQ_GATTC_READ_RESULT', bytes(data[-1]))
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
        while True:
            i = 10
    multitest.globals(BDADDR=ble.config('mac'))
    ble.config(gap_name='GAP_NAME')
    print(ble.config('gap_name'))
    ble.gatts_register_services([])
    print('gap_advertise')
    multitest.next()
    try:
        for iteration in range(2):
            ble.config(gap_name='GAP_NAME{}'.format(iteration))
            print(ble.config('gap_name'))
            ble.gap_advertise(20000)
            multitest.broadcast('peripheral:adv:{}'.format(iteration))
            wait_for_event(_IRQ_CENTRAL_CONNECT, TIMEOUT_MS)
            wait_for_event(_IRQ_CENTRAL_DISCONNECT, 4 * TIMEOUT_MS)
    finally:
        ble.active(0)

def instance1():
    if False:
        return 10
    multitest.next()
    try:
        value_handle = None
        for iteration in range(2):
            multitest.wait('peripheral:adv:{}'.format(iteration))
            print('gap_connect')
            ble.gap_connect(*BDADDR)
            conn_handle = wait_for_event(_IRQ_PERIPHERAL_CONNECT, TIMEOUT_MS)
            if iteration == 0:
                print('gattc_discover_characteristics')
                ble.gattc_discover_characteristics(conn_handle, 1, 65535)
                value_handle = wait_for_event(_IRQ_GATTC_CHARACTERISTIC_RESULT, TIMEOUT_MS)
                wait_for_event(_IRQ_GATTC_CHARACTERISTIC_DONE, TIMEOUT_MS)
            print('gattc_read')
            ble.gattc_read(conn_handle, value_handle)
            wait_for_event(_IRQ_GATTC_READ_RESULT, TIMEOUT_MS)
            print('gap_disconnect:', ble.gap_disconnect(conn_handle))
            wait_for_event(_IRQ_PERIPHERAL_DISCONNECT, TIMEOUT_MS)
    finally:
        ble.active(0)
ble = bluetooth.BLE()
ble.active(1)
ble.irq(irq)