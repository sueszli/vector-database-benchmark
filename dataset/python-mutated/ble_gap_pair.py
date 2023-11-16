from micropython import const
import time, machine, bluetooth
if not hasattr(bluetooth.BLE, 'gap_pair'):
    print('SKIP')
    raise SystemExit
TIMEOUT_MS = 4000
_IRQ_CENTRAL_CONNECT = const(1)
_IRQ_CENTRAL_DISCONNECT = const(2)
_IRQ_GATTS_READ_REQUEST = const(4)
_IRQ_PERIPHERAL_CONNECT = const(7)
_IRQ_PERIPHERAL_DISCONNECT = const(8)
_IRQ_GATTC_CHARACTERISTIC_RESULT = const(11)
_IRQ_GATTC_CHARACTERISTIC_DONE = const(12)
_IRQ_GATTC_READ_RESULT = const(15)
_IRQ_ENCRYPTION_UPDATE = const(28)
_FLAG_READ = const(2)
_FLAG_READ_ENCRYPTED = const(512)
SERVICE_UUID = bluetooth.UUID('A5A5A5A5-FFFF-9999-1111-5A5A5A5A5A5A')
CHAR_UUID = bluetooth.UUID('00000000-1111-2222-3333-444444444444')
CHAR = (CHAR_UUID, _FLAG_READ | _FLAG_READ_ENCRYPTED)
SERVICE = (SERVICE_UUID, (CHAR,))
waiting_events = {}

def irq(event, data):
    if False:
        i = 10
        return i + 15
    if event == _IRQ_CENTRAL_CONNECT:
        print('_IRQ_CENTRAL_CONNECT')
        waiting_events[event] = data[0]
    elif event == _IRQ_CENTRAL_DISCONNECT:
        print('_IRQ_CENTRAL_DISCONNECT')
    elif event == _IRQ_GATTS_READ_REQUEST:
        print('_IRQ_GATTS_READ_REQUEST')
    elif event == _IRQ_PERIPHERAL_CONNECT:
        print('_IRQ_PERIPHERAL_CONNECT')
        waiting_events[event] = data[0]
    elif event == _IRQ_PERIPHERAL_DISCONNECT:
        print('_IRQ_PERIPHERAL_DISCONNECT')
    elif event == _IRQ_GATTC_CHARACTERISTIC_RESULT:
        if data[-1] == CHAR_UUID:
            print('_IRQ_GATTC_CHARACTERISTIC_RESULT', data[-1])
            waiting_events[event] = data[2]
        else:
            return
    elif event == _IRQ_GATTC_CHARACTERISTIC_DONE:
        print('_IRQ_GATTC_CHARACTERISTIC_DONE')
    elif event == _IRQ_GATTC_READ_RESULT:
        print('_IRQ_GATTC_READ_RESULT', bytes(data[-1]))
    elif event == _IRQ_ENCRYPTION_UPDATE:
        print('_IRQ_ENCRYPTION_UPDATE', data[1], data[2], data[3])
    if event not in waiting_events:
        waiting_events[event] = None

def wait_for_event(event, timeout_ms):
    if False:
        return 10
    t0 = time.ticks_ms()
    while time.ticks_diff(time.ticks_ms(), t0) < timeout_ms:
        if event in waiting_events:
            return waiting_events.pop(event)
        machine.idle()
    raise ValueError('Timeout waiting for {}'.format(event))

def instance0():
    if False:
        print('Hello World!')
    multitest.globals(BDADDR=ble.config('mac'))
    ((char_handle,),) = ble.gatts_register_services((SERVICE,))
    ble.gatts_write(char_handle, 'encrypted')
    print('gap_advertise')
    ble.gap_advertise(20000, b'\x02\x01\x06\x04\xffMPY')
    multitest.next()
    try:
        wait_for_event(_IRQ_CENTRAL_CONNECT, TIMEOUT_MS)
        wait_for_event(_IRQ_ENCRYPTION_UPDATE, TIMEOUT_MS)
        wait_for_event(_IRQ_GATTS_READ_REQUEST, TIMEOUT_MS)
        wait_for_event(_IRQ_CENTRAL_DISCONNECT, TIMEOUT_MS)
    finally:
        ble.active(0)

def instance1():
    if False:
        return 10
    multitest.next()
    try:
        print('gap_connect')
        ble.gap_connect(*BDADDR)
        conn_handle = wait_for_event(_IRQ_PERIPHERAL_CONNECT, TIMEOUT_MS)
        ble.gattc_discover_characteristics(conn_handle, 1, 65535)
        value_handle = wait_for_event(_IRQ_GATTC_CHARACTERISTIC_RESULT, TIMEOUT_MS)
        wait_for_event(_IRQ_GATTC_CHARACTERISTIC_DONE, TIMEOUT_MS)
        print('gap_pair')
        ble.gap_pair(conn_handle)
        wait_for_event(_IRQ_ENCRYPTION_UPDATE, TIMEOUT_MS)
        print('gattc_read')
        ble.gattc_read(conn_handle, value_handle)
        wait_for_event(_IRQ_GATTC_READ_RESULT, TIMEOUT_MS)
        print('gap_disconnect:', ble.gap_disconnect(conn_handle))
        wait_for_event(_IRQ_PERIPHERAL_DISCONNECT, TIMEOUT_MS)
    finally:
        ble.active(0)
ble = bluetooth.BLE()
ble.config(mitm=True, le_secure=True, bond=False)
ble.active(1)
ble.irq(irq)