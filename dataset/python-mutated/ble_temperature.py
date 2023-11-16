import bluetooth
import random
import struct
import time
from ble_advertising import advertising_payload
from micropython import const
_IRQ_CENTRAL_CONNECT = const(1)
_IRQ_CENTRAL_DISCONNECT = const(2)
_IRQ_GATTS_INDICATE_DONE = const(20)
_FLAG_READ = const(2)
_FLAG_NOTIFY = const(16)
_FLAG_INDICATE = const(32)
_ENV_SENSE_UUID = bluetooth.UUID(6170)
_TEMP_CHAR = (bluetooth.UUID(10862), _FLAG_READ | _FLAG_NOTIFY | _FLAG_INDICATE)
_ENV_SENSE_SERVICE = (_ENV_SENSE_UUID, (_TEMP_CHAR,))
_ADV_APPEARANCE_GENERIC_THERMOMETER = const(768)

class BLETemperature:

    def __init__(self, ble, name='mpy-temp'):
        if False:
            print('Hello World!')
        self._ble = ble
        self._ble.active(True)
        self._ble.irq(self._irq)
        ((self._handle,),) = self._ble.gatts_register_services((_ENV_SENSE_SERVICE,))
        self._connections = set()
        self._payload = advertising_payload(name=name, services=[_ENV_SENSE_UUID], appearance=_ADV_APPEARANCE_GENERIC_THERMOMETER)
        self._advertise()

    def _irq(self, event, data):
        if False:
            while True:
                i = 10
        if event == _IRQ_CENTRAL_CONNECT:
            (conn_handle, _, _) = data
            self._connections.add(conn_handle)
        elif event == _IRQ_CENTRAL_DISCONNECT:
            (conn_handle, _, _) = data
            self._connections.remove(conn_handle)
            self._advertise()
        elif event == _IRQ_GATTS_INDICATE_DONE:
            (conn_handle, value_handle, status) = data

    def set_temperature(self, temp_deg_c, notify=False, indicate=False):
        if False:
            print('Hello World!')
        self._ble.gatts_write(self._handle, struct.pack('<h', int(temp_deg_c * 100)))
        if notify or indicate:
            for conn_handle in self._connections:
                if notify:
                    self._ble.gatts_notify(conn_handle, self._handle)
                if indicate:
                    self._ble.gatts_indicate(conn_handle, self._handle)

    def _advertise(self, interval_us=500000):
        if False:
            print('Hello World!')
        self._ble.gap_advertise(interval_us, adv_data=self._payload)

def demo():
    if False:
        i = 10
        return i + 15
    ble = bluetooth.BLE()
    temp = BLETemperature(ble)
    t = 25
    i = 0
    while True:
        i = (i + 1) % 10
        temp.set_temperature(t, notify=i == 0, indicate=False)
        t += random.uniform(-0.5, 0.5)
        time.sleep_ms(1000)
if __name__ == '__main__':
    demo()