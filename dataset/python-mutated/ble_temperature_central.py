import bluetooth
import random
import struct
import time
import micropython
from ble_advertising import decode_services, decode_name
from micropython import const
_IRQ_CENTRAL_CONNECT = const(1)
_IRQ_CENTRAL_DISCONNECT = const(2)
_IRQ_GATTS_WRITE = const(3)
_IRQ_GATTS_READ_REQUEST = const(4)
_IRQ_SCAN_RESULT = const(5)
_IRQ_SCAN_DONE = const(6)
_IRQ_PERIPHERAL_CONNECT = const(7)
_IRQ_PERIPHERAL_DISCONNECT = const(8)
_IRQ_GATTC_SERVICE_RESULT = const(9)
_IRQ_GATTC_SERVICE_DONE = const(10)
_IRQ_GATTC_CHARACTERISTIC_RESULT = const(11)
_IRQ_GATTC_CHARACTERISTIC_DONE = const(12)
_IRQ_GATTC_DESCRIPTOR_RESULT = const(13)
_IRQ_GATTC_DESCRIPTOR_DONE = const(14)
_IRQ_GATTC_READ_RESULT = const(15)
_IRQ_GATTC_READ_DONE = const(16)
_IRQ_GATTC_WRITE_DONE = const(17)
_IRQ_GATTC_NOTIFY = const(18)
_IRQ_GATTC_INDICATE = const(19)
_ADV_IND = const(0)
_ADV_DIRECT_IND = const(1)
_ADV_SCAN_IND = const(2)
_ADV_NONCONN_IND = const(3)
_ENV_SENSE_UUID = bluetooth.UUID(6170)
_TEMP_UUID = bluetooth.UUID(10862)
_ADV_APPEARANCE_GENERIC_THERMOMETER = const(768)

class BLETemperatureCentral:

    def __init__(self, ble):
        if False:
            for i in range(10):
                print('nop')
        self._ble = ble
        self._ble.active(True)
        self._ble.irq(self._irq)
        self._reset()

    def _reset(self):
        if False:
            i = 10
            return i + 15
        self._name = None
        self._addr_type = None
        self._addr = None
        self._value = None
        self._scan_callback = None
        self._conn_callback = None
        self._read_callback = None
        self._notify_callback = None
        self._conn_handle = None
        self._start_handle = None
        self._end_handle = None
        self._value_handle = None

    def _irq(self, event, data):
        if False:
            i = 10
            return i + 15
        if event == _IRQ_SCAN_RESULT:
            (addr_type, addr, adv_type, rssi, adv_data) = data
            if adv_type in (_ADV_IND, _ADV_DIRECT_IND) and _ENV_SENSE_UUID in decode_services(adv_data):
                self._addr_type = addr_type
                self._addr = bytes(addr)
                self._name = decode_name(adv_data) or '?'
                self._ble.gap_scan(None)
        elif event == _IRQ_SCAN_DONE:
            if self._scan_callback:
                if self._addr:
                    self._scan_callback(self._addr_type, self._addr, self._name)
                    self._scan_callback = None
                else:
                    self._scan_callback(None, None, None)
        elif event == _IRQ_PERIPHERAL_CONNECT:
            (conn_handle, addr_type, addr) = data
            if addr_type == self._addr_type and addr == self._addr:
                self._conn_handle = conn_handle
                self._ble.gattc_discover_services(self._conn_handle)
        elif event == _IRQ_PERIPHERAL_DISCONNECT:
            (conn_handle, _, _) = data
            if conn_handle == self._conn_handle:
                self._reset()
        elif event == _IRQ_GATTC_SERVICE_RESULT:
            (conn_handle, start_handle, end_handle, uuid) = data
            if conn_handle == self._conn_handle and uuid == _ENV_SENSE_UUID:
                (self._start_handle, self._end_handle) = (start_handle, end_handle)
        elif event == _IRQ_GATTC_SERVICE_DONE:
            if self._start_handle and self._end_handle:
                self._ble.gattc_discover_characteristics(self._conn_handle, self._start_handle, self._end_handle)
            else:
                print('Failed to find environmental sensing service.')
        elif event == _IRQ_GATTC_CHARACTERISTIC_RESULT:
            (conn_handle, def_handle, value_handle, properties, uuid) = data
            if conn_handle == self._conn_handle and uuid == _TEMP_UUID:
                self._value_handle = value_handle
        elif event == _IRQ_GATTC_CHARACTERISTIC_DONE:
            if self._value_handle:
                if self._conn_callback:
                    self._conn_callback()
            else:
                print('Failed to find temperature characteristic.')
        elif event == _IRQ_GATTC_READ_RESULT:
            (conn_handle, value_handle, char_data) = data
            if conn_handle == self._conn_handle and value_handle == self._value_handle:
                self._update_value(char_data)
                if self._read_callback:
                    self._read_callback(self._value)
                    self._read_callback = None
        elif event == _IRQ_GATTC_READ_DONE:
            (conn_handle, value_handle, status) = data
        elif event == _IRQ_GATTC_NOTIFY:
            (conn_handle, value_handle, notify_data) = data
            if conn_handle == self._conn_handle and value_handle == self._value_handle:
                self._update_value(notify_data)
                if self._notify_callback:
                    self._notify_callback(self._value)

    def is_connected(self):
        if False:
            return 10
        return self._conn_handle is not None and self._value_handle is not None

    def scan(self, callback=None):
        if False:
            i = 10
            return i + 15
        self._addr_type = None
        self._addr = None
        self._scan_callback = callback
        self._ble.gap_scan(2000, 30000, 30000)

    def connect(self, addr_type=None, addr=None, callback=None):
        if False:
            print('Hello World!')
        self._addr_type = addr_type or self._addr_type
        self._addr = addr or self._addr
        self._conn_callback = callback
        if self._addr_type is None or self._addr is None:
            return False
        self._ble.gap_connect(self._addr_type, self._addr)
        return True

    def disconnect(self):
        if False:
            print('Hello World!')
        if self._conn_handle is None:
            return
        self._ble.gap_disconnect(self._conn_handle)
        self._reset()

    def read(self, callback):
        if False:
            return 10
        if not self.is_connected():
            return
        self._read_callback = callback
        self._ble.gattc_read(self._conn_handle, self._value_handle)

    def on_notify(self, callback):
        if False:
            while True:
                i = 10
        self._notify_callback = callback

    def _update_value(self, data):
        if False:
            return 10
        self._value = struct.unpack('<h', data)[0] / 100
        return self._value

    def value(self):
        if False:
            i = 10
            return i + 15
        return self._value

def demo():
    if False:
        while True:
            i = 10
    ble = bluetooth.BLE()
    central = BLETemperatureCentral(ble)
    not_found = False

    def on_scan(addr_type, addr, name):
        if False:
            i = 10
            return i + 15
        if addr_type is not None:
            print('Found sensor:', addr_type, addr, name)
            central.connect()
        else:
            nonlocal not_found
            not_found = True
            print('No sensor found.')
    central.scan(callback=on_scan)
    while not central.is_connected():
        time.sleep_ms(100)
        if not_found:
            return
    print('Connected')
    while central.is_connected():
        central.read(callback=print)
        time.sleep_ms(2000)
    print('Disconnected')
if __name__ == '__main__':
    demo()