"""The settings for RS485 are stored in a dedicated object that can be applied to
serial ports (where supported).
NOTE: Some implementations may only support a subset of the settings.
"""
from __future__ import absolute_import
import time
import serial

class RS485Settings(object):

    def __init__(self, rts_level_for_tx=True, rts_level_for_rx=False, loopback=False, delay_before_tx=None, delay_before_rx=None):
        if False:
            print('Hello World!')
        self.rts_level_for_tx = rts_level_for_tx
        self.rts_level_for_rx = rts_level_for_rx
        self.loopback = loopback
        self.delay_before_tx = delay_before_tx
        self.delay_before_rx = delay_before_rx

class RS485(serial.Serial):
    """    A subclass that replaces the write method with one that toggles RTS
    according to the RS485 settings.

    NOTE: This may work unreliably on some serial ports (control signals not
          synchronized or delayed compared to data). Using delays may be
          unreliable (varying times, larger than expected) as the OS may not
          support very fine grained delays (no smaller than in the order of
          tens of milliseconds).

    NOTE: Some implementations support this natively. Better performance
          can be expected when the native version is used.

    NOTE: The loopback property is ignored by this implementation. The actual
          behavior depends on the used hardware.

    Usage:

        ser = RS485(...)
        ser.rs485_mode = RS485Settings(...)
        ser.write(b'hello')
    """

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(RS485, self).__init__(*args, **kwargs)
        self._alternate_rs485_settings = None

    def write(self, b):
        if False:
            i = 10
            return i + 15
        'Write to port, controlling RTS before and after transmitting.'
        if self._alternate_rs485_settings is not None:
            self.setRTS(self._alternate_rs485_settings.rts_level_for_tx)
            if self._alternate_rs485_settings.delay_before_tx is not None:
                time.sleep(self._alternate_rs485_settings.delay_before_tx)
            super(RS485, self).write(b)
            super(RS485, self).flush()
            if self._alternate_rs485_settings.delay_before_rx is not None:
                time.sleep(self._alternate_rs485_settings.delay_before_rx)
            self.setRTS(self._alternate_rs485_settings.rts_level_for_rx)
        else:
            super(RS485, self).write(b)

    @property
    def rs485_mode(self):
        if False:
            print('Hello World!')
        '        Enable RS485 mode and apply new settings, set to None to disable.\n        See serial.rs485.RS485Settings for more info about the value.\n        '
        return self._alternate_rs485_settings

    @rs485_mode.setter
    def rs485_mode(self, rs485_settings):
        if False:
            for i in range(10):
                print('nop')
        self._alternate_rs485_settings = rs485_settings