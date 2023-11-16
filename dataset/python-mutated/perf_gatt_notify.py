from micropython import const
import time, machine, bluetooth
TIMEOUT_MS = 2000
_IRQ_CENTRAL_CONNECT = const(1)
_IRQ_CENTRAL_DISCONNECT = const(2)
_IRQ_PERIPHERAL_CONNECT = const(7)
_IRQ_PERIPHERAL_DISCONNECT = const(8)
_IRQ_GATTC_CHARACTERISTIC_RESULT = const(11)
_IRQ_GATTC_CHARACTERISTIC_DONE = const(12)
_IRQ_GATTC_NOTIFY = const(18)
_NUM_NOTIFICATIONS = const(50)
SERVICE_UUID = bluetooth.UUID('A5A5A5A5-FFFF-9999-1111-5A5A5A5A5A5A')
CHAR_UUID = bluetooth.UUID('00000000-1111-2222-3333-444444444444')
CHAR = (CHAR_UUID, bluetooth.FLAG_NOTIFY)
SERVICE = (SERVICE_UUID, (CHAR,))
SERVICES = (SERVICE,)
is_central = False
waiting_events = {}

def irq(event, data):
    if False:
        return 10
    if event == _IRQ_CENTRAL_CONNECT:
        waiting_events[event] = data[0]
    elif event == _IRQ_PERIPHERAL_CONNECT:
        waiting_events[event] = data[0]
    elif event == _IRQ_GATTC_CHARACTERISTIC_RESULT:
        if data[-1] == CHAR_UUID:
            waiting_events[event] = data[2]
        else:
            return
    elif event == _IRQ_GATTC_NOTIFY:
        if is_central:
            (conn_handle, value_handle, notify_data) = data
            ble.gatts_notify(conn_handle, value_handle, b'central' + notify_data)
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
        print('Hello World!')
    multitest.globals(BDADDR=ble.config('mac'))
    ((char_handle,),) = ble.gatts_register_services(SERVICES)
    print('gap_advertise')
    ble.gap_advertise(20000, b'\x02\x01\x06\x04\xffMPY')
    multitest.next()
    try:
        conn_handle = wait_for_event(_IRQ_CENTRAL_CONNECT, TIMEOUT_MS)
        ble.gattc_discover_characteristics(conn_handle, 1, 65535)
        value_handle = wait_for_event(_IRQ_GATTC_CHARACTERISTIC_RESULT, TIMEOUT_MS)
        wait_for_event(_IRQ_GATTC_CHARACTERISTIC_DONE, TIMEOUT_MS)
        time.sleep_ms(500)
        ticks_start = time.ticks_ms()
        for i in range(_NUM_NOTIFICATIONS):
            ble.gatts_notify(conn_handle, value_handle, 'peripheral' + str(i))
            wait_for_event(_IRQ_GATTC_NOTIFY, TIMEOUT_MS)
        ticks_end = time.ticks_ms()
        ticks_total = time.ticks_diff(ticks_end, ticks_start)
        multitest.output_metric('Acknowledged {} notifications in {} ms. {} ms/notification.'.format(_NUM_NOTIFICATIONS, ticks_total, ticks_total // _NUM_NOTIFICATIONS))
        print('gap_disconnect:', ble.gap_disconnect(conn_handle))
        wait_for_event(_IRQ_CENTRAL_DISCONNECT, TIMEOUT_MS)
    finally:
        ble.active(0)

def instance1():
    if False:
        i = 10
        return i + 15
    global is_central
    is_central = True
    ((char_handle,),) = ble.gatts_register_services(SERVICES)
    multitest.next()
    try:
        print('gap_connect')
        ble.gap_connect(*BDADDR)
        conn_handle = wait_for_event(_IRQ_PERIPHERAL_CONNECT, TIMEOUT_MS)
        ble.gattc_discover_characteristics(conn_handle, 1, 65535)
        value_handle = wait_for_event(_IRQ_GATTC_CHARACTERISTIC_RESULT, TIMEOUT_MS)
        wait_for_event(_IRQ_GATTC_CHARACTERISTIC_DONE, TIMEOUT_MS)
        wait_for_event(_IRQ_PERIPHERAL_DISCONNECT, 20000)
    finally:
        ble.active(0)
ble = bluetooth.BLE()
ble.active(1)
ble.irq(irq)