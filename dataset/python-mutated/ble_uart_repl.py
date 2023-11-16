import bluetooth
import io
import os
import micropython
from micropython import const
import machine
from ble_uart_peripheral import BLEUART
_MP_STREAM_POLL = const(3)
_MP_STREAM_POLL_RD = const(1)
if hasattr(machine, 'Timer'):
    _timer = machine.Timer(-1)
else:
    _timer = None

def schedule_in(handler, delay_ms):
    if False:
        while True:
            i = 10

    def _wrap(_arg):
        if False:
            i = 10
            return i + 15
        handler()
    if _timer:
        _timer.init(mode=machine.Timer.ONE_SHOT, period=delay_ms, callback=_wrap)
    else:
        micropython.schedule(_wrap, None)

class BLEUARTStream(io.IOBase):

    def __init__(self, uart):
        if False:
            i = 10
            return i + 15
        self._uart = uart
        self._tx_buf = bytearray()
        self._uart.irq(self._on_rx)

    def _on_rx(self):
        if False:
            i = 10
            return i + 15
        if hasattr(os, 'dupterm_notify'):
            os.dupterm_notify(None)

    def read(self, sz=None):
        if False:
            return 10
        return self._uart.read(sz)

    def readinto(self, buf):
        if False:
            i = 10
            return i + 15
        avail = self._uart.read(len(buf))
        if not avail:
            return None
        for i in range(len(avail)):
            buf[i] = avail[i]
        return len(avail)

    def ioctl(self, op, arg):
        if False:
            i = 10
            return i + 15
        if op == _MP_STREAM_POLL:
            if self._uart.any():
                return _MP_STREAM_POLL_RD
        return 0

    def _flush(self):
        if False:
            for i in range(10):
                print('nop')
        data = self._tx_buf[0:100]
        self._tx_buf = self._tx_buf[100:]
        self._uart.write(data)
        if self._tx_buf:
            schedule_in(self._flush, 50)

    def write(self, buf):
        if False:
            while True:
                i = 10
        empty = not self._tx_buf
        self._tx_buf += buf
        if empty:
            schedule_in(self._flush, 50)

def start():
    if False:
        return 10
    ble = bluetooth.BLE()
    uart = BLEUART(ble, name='mpy-repl')
    stream = BLEUARTStream(uart)
    os.dupterm(stream)