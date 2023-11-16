from micropython import const
import time, machine, bluetooth
TIMEOUT_MS = 5000
_IRQ_CENTRAL_CONNECT = const(1)
_IRQ_CENTRAL_DISCONNECT = const(2)
_IRQ_GATTS_WRITE = const(3)
_IRQ_PERIPHERAL_CONNECT = const(7)
_IRQ_PERIPHERAL_DISCONNECT = const(8)
_IRQ_GATTC_CHARACTERISTIC_RESULT = const(11)
_IRQ_GATTC_CHARACTERISTIC_DONE = const(12)
_IRQ_GATTC_DESCRIPTOR_RESULT = const(13)
_IRQ_GATTC_DESCRIPTOR_DONE = const(14)
_IRQ_GATTC_READ_RESULT = const(15)
_IRQ_GATTC_READ_DONE = const(16)
_IRQ_GATTC_WRITE_DONE = const(17)
_IRQ_GATTC_NOTIFY = const(18)
_IRQ_GATTC_INDICATE = const(19)
_CCCD_UUID = bluetooth.UUID(const(10498))
SERVICE_UUID = bluetooth.UUID('A5A5A5A5-FFFF-9999-1111-5A5A5A5A5A5A')
CHAR_UUID = bluetooth.UUID('00000000-1111-2222-3333-444444444444')
CHAR = (CHAR_UUID, bluetooth.FLAG_READ | bluetooth.FLAG_WRITE | bluetooth.FLAG_NOTIFY | bluetooth.FLAG_INDICATE)
SERVICE = (SERVICE_UUID, (CHAR,))
SERVICES = (SERVICE,)
waiting_events = {}

def irq(event, data):
    if False:
        print('Hello World!')
    if event == _IRQ_CENTRAL_CONNECT:
        print('_IRQ_CENTRAL_CONNECT')
        waiting_events[event] = data[0]
    elif event == _IRQ_CENTRAL_DISCONNECT:
        print('_IRQ_CENTRAL_DISCONNECT')
    elif event == _IRQ_GATTS_WRITE:
        print('_IRQ_GATTS_WRITE', ble.gatts_read(data[-1]))
    elif event == _IRQ_PERIPHERAL_CONNECT:
        print('_IRQ_PERIPHERAL_CONNECT')
        waiting_events[event] = data[0]
    elif event == _IRQ_PERIPHERAL_DISCONNECT:
        print('_IRQ_PERIPHERAL_DISCONNECT')
    elif event == _IRQ_GATTC_CHARACTERISTIC_RESULT:
        if data[-1] == CHAR_UUID:
            print('_IRQ_GATTC_CHARACTERISTIC_RESULT', data[-1])
            waiting_events[event] = (data[2], data[1])
        else:
            return
    elif event == _IRQ_GATTC_CHARACTERISTIC_DONE:
        print('_IRQ_GATTC_CHARACTERISTIC_DONE')
    elif event == _IRQ_GATTC_DESCRIPTOR_RESULT:
        if data[-1] == _CCCD_UUID:
            print('_IRQ_GATTC_DESCRIPTOR_RESULT', data[-1])
            waiting_events[event] = data[1]
        else:
            return
    elif event == _IRQ_GATTC_DESCRIPTOR_DONE:
        print('_IRQ_GATTC_DESCRIPTOR_DONE')
    elif event == _IRQ_GATTC_READ_RESULT:
        print('_IRQ_GATTC_READ_RESULT', bytes(data[-1]))
    elif event == _IRQ_GATTC_READ_DONE:
        print('_IRQ_GATTC_READ_DONE', data[-1])
    elif event == _IRQ_GATTC_WRITE_DONE:
        print('_IRQ_GATTC_WRITE_DONE', data[-1])
    elif event == _IRQ_GATTC_NOTIFY:
        print('_IRQ_GATTC_NOTIFY', bytes(data[-1]))
    elif event == _IRQ_GATTC_INDICATE:
        print('_IRQ_GATTC_NOTIFY', bytes(data[-1]))
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
        i = 10
        return i + 15
    multitest.globals(BDADDR=ble.config('mac'))
    ((char_handle,),) = ble.gatts_register_services(SERVICES)
    print('gap_advertise')
    ble.gap_advertise(20000, b'\x02\x01\x06\x04\xffMPY')
    multitest.next()
    try:
        ble.gatts_write(char_handle, 'periph0')
        conn_handle = wait_for_event(_IRQ_CENTRAL_CONNECT, TIMEOUT_MS * 10)
        wait_for_event(_IRQ_GATTS_WRITE, TIMEOUT_MS)
        print('sync A')
        ble.gatts_write(char_handle, 'periph1')
        time.sleep_ms(100)
        ble.gatts_write(char_handle, 'periph2')
        ble.gatts_notify(conn_handle, char_handle)
        time.sleep_ms(100)
        ble.gatts_write(char_handle, 'periph3', True)
        time.sleep_ms(100)
        multitest.broadcast('A')
        wait_for_event(_IRQ_GATTS_WRITE, TIMEOUT_MS)
        print('sync B')
        ble.gatts_write(char_handle, 'periph4', False)
        time.sleep_ms(100)
        ble.gatts_write(char_handle, 'periph5', True)
        time.sleep_ms(100)
        multitest.broadcast('B')
        wait_for_event(_IRQ_GATTS_WRITE, TIMEOUT_MS)
        print('sync C')
        ble.gatts_write(char_handle, 'periph6', False)
        time.sleep_ms(100)
        ble.gatts_write(char_handle, 'periph7', True)
        time.sleep_ms(100)
        multitest.broadcast('C')
        wait_for_event(_IRQ_GATTS_WRITE, TIMEOUT_MS)
        print('sync D')
        ble.gatts_write(char_handle, 'periph8', False)
        time.sleep_ms(100)
        ble.gatts_write(char_handle, 'periph9', True)
        time.sleep_ms(100)
        ble.gatts_write(char_handle, 'periph10')
        ble.gatts_notify(conn_handle, char_handle)
        time.sleep_ms(100)
        multitest.broadcast('D')
        wait_for_event(_IRQ_CENTRAL_DISCONNECT, TIMEOUT_MS)
    finally:
        ble.active(0)

def instance1():
    if False:
        for i in range(10):
            print('nop')
    multitest.next()
    try:
        print('gap_connect')
        ble.gap_connect(*BDADDR)
        conn_handle = wait_for_event(_IRQ_PERIPHERAL_CONNECT, TIMEOUT_MS)
        ble.gattc_discover_characteristics(conn_handle, 1, 65535)
        (value_handle, end_handle) = wait_for_event(_IRQ_GATTC_CHARACTERISTIC_RESULT, TIMEOUT_MS)
        wait_for_event(_IRQ_GATTC_CHARACTERISTIC_DONE, TIMEOUT_MS)
        ble.gattc_discover_descriptors(conn_handle, value_handle, end_handle)
        cccd_handle = wait_for_event(_IRQ_GATTC_DESCRIPTOR_RESULT, TIMEOUT_MS)
        wait_for_event(_IRQ_GATTC_DESCRIPTOR_DONE, TIMEOUT_MS)
        print('gattc_read')
        ble.gattc_read(conn_handle, value_handle)
        wait_for_event(_IRQ_GATTC_READ_RESULT, TIMEOUT_MS)
        print('gattc_write')
        ble.gattc_write(conn_handle, value_handle, 'central0', 1)
        wait_for_event(_IRQ_GATTC_WRITE_DONE, TIMEOUT_MS)
        multitest.wait('A')
        ble.gattc_write(conn_handle, cccd_handle, b'\x01\x00', 1)
        wait_for_event(_IRQ_GATTC_WRITE_DONE, TIMEOUT_MS)
        print('gattc_write')
        ble.gattc_write(conn_handle, value_handle, 'central1', 1)
        wait_for_event(_IRQ_GATTC_WRITE_DONE, TIMEOUT_MS)
        multitest.wait('B')
        ble.gattc_write(conn_handle, cccd_handle, b'\x02\x00', 1)
        wait_for_event(_IRQ_GATTC_WRITE_DONE, TIMEOUT_MS)
        print('gattc_write')
        ble.gattc_write(conn_handle, value_handle, 'central2', 1)
        wait_for_event(_IRQ_GATTC_WRITE_DONE, TIMEOUT_MS)
        multitest.wait('C')
        ble.gattc_write(conn_handle, cccd_handle, b'\x00\x00', 1)
        wait_for_event(_IRQ_GATTC_WRITE_DONE, TIMEOUT_MS)
        print('gattc_write')
        ble.gattc_write(conn_handle, value_handle, 'central3', 1)
        wait_for_event(_IRQ_GATTC_WRITE_DONE, TIMEOUT_MS)
        multitest.wait('D')
        print('gap_disconnect:', ble.gap_disconnect(conn_handle))
        wait_for_event(_IRQ_PERIPHERAL_DISCONNECT, TIMEOUT_MS)
    finally:
        ble.active(0)
ble = bluetooth.BLE()
ble.active(1)
ble.irq(irq)