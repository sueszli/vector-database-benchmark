import logging
import os
import struct
import time
from pyboy.utils import STATE_VERSION, IntIOWrapper
logger = logging.getLogger(__name__)

class RTC:

    def __init__(self, filename):
        if False:
            return 10
        self.filename = filename + '.rtc'
        if not os.path.exists(self.filename):
            logger.info('No RTC file found. Skipping.')
        else:
            with open(self.filename, 'rb') as f:
                self.load_state(IntIOWrapper(f), STATE_VERSION)
        self.latch_enabled = False
        self.timezero = time.time()
        self.sec_latch = 0
        self.min_latch = 0
        self.hour_latch = 0
        self.day_latch_low = 0
        self.day_latch_high = 0
        self.day_carry = 0
        self.halt = 0

    def stop(self):
        if False:
            i = 10
            return i + 15
        with open(self.filename, 'wb') as f:
            self.save_state(IntIOWrapper(f))

    def save_state(self, f):
        if False:
            i = 10
            return i + 15
        for b in struct.pack('f', self.timezero):
            f.write(b)
        f.write(self.halt)
        f.write(self.day_carry)

    def load_state(self, f, state_version):
        if False:
            i = 10
            return i + 15
        self.timezero = struct.unpack('f', bytes([f.read() for _ in range(4)]))[0]
        self.halt = f.read()
        self.day_carry = f.read()

    def latch_rtc(self):
        if False:
            return 10
        t = time.time() - self.timezero
        self.sec_latch = int(t % 60)
        self.min_latch = int(t / 60 % 60)
        self.hour_latch = int(t / 3600 % 24)
        days = int(t / 3600 / 24)
        self.day_latch_low = days & 255
        self.day_latch_high = days >> 8
        if self.day_latch_high > 1:
            self.day_carry = 1
            self.day_latch_high &= 1
            self.timezero += 512 * 3600 * 24

    def writecommand(self, value):
        if False:
            return 10
        if value == 0:
            self.latch_enabled = False
        elif value == 1:
            if not self.latch_enabled:
                self.latch_rtc()
            self.latch_enabled = True
        else:
            logger.warning('Invalid RTC command: %0.2x' % value)

    def getregister(self, register):
        if False:
            print('Hello World!')
        if not self.latch_enabled:
            logger.debug('RTC: Get register, but nothing is latched! 0x%0.2x' % register)
        if register == 8:
            return self.sec_latch
        elif register == 9:
            return self.min_latch
        elif register == 10:
            return self.hour_latch
        elif register == 11:
            return self.day_latch_low
        elif register == 12:
            day_high = self.day_latch_high & 1
            halt = self.halt << 6
            day_carry = self.day_carry << 7
            return day_high + halt + day_carry
        else:
            logger.warning('Invalid RTC register: %0.4x' % register)

    def setregister(self, register, value):
        if False:
            return 10
        if not self.latch_enabled:
            logger.debug('RTC: Set register, but nothing is latched! 0x%0.4x, 0x%0.2x' % (register, value))
        t = time.time() - self.timezero
        if register == 8:
            self.timezero -= int(t % 60) - value
        elif register == 9:
            self.timezero -= int(t / 60 % 60) - value
        elif register == 10:
            self.timezero -= int(t / 3600 % 24) - value
        elif register == 11:
            self.timezero -= int(t / 3600 / 24) - value
        elif register == 12:
            day_high = value & 1
            halt = (value & 64) >> 6
            day_carry = (value & 128) >> 7
            self.halt = halt
            if self.halt == 0:
                pass
            else:
                logger.warning('Stopping RTC is not implemented!')
            self.timezero -= int(t / 3600 / 24) - (day_high << 8)
            self.day_carry = day_carry
        else:
            logger.warning('Invalid RTC register: %0.4x %0.2x' % (register, value))