import array
import logging
import os
from pyboy.logger import logger
from pyboy.utils import IntIOWrapper
from .rtc import RTC
logger = logging.getLogger(__name__)

class BaseMBC:

    def __init__(self, filename, rombanks, external_ram_count, carttype, sram, battery, rtc_enabled):
        if False:
            i = 10
            return i + 15
        self.filename = filename + '.ram'
        self.rombanks = rombanks
        self.carttype = carttype
        self.battery = battery
        self.rtc_enabled = rtc_enabled
        if self.rtc_enabled:
            self.rtc = RTC(filename)
        self.rambank_initialized = False
        self.external_rom_count = len(rombanks)
        self.external_ram_count = external_ram_count
        self.init_rambanks(external_ram_count)
        self.gamename = self.getgamename(rombanks)
        self.memorymodel = 0
        self.rambank_enabled = False
        self.rambank_selected = 0
        self.rombank_selected = 1
        self.cgb = bool(self.getitem(323) >> 7)
        if not os.path.exists(self.filename):
            logger.debug('No RAM file found. Skipping.')
        else:
            with open(self.filename, 'rb') as f:
                self.load_ram(IntIOWrapper(f))

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        with open(self.filename, 'wb') as f:
            self.save_ram(IntIOWrapper(f))
        if self.rtc_enabled:
            self.rtc.stop()

    def save_state(self, f):
        if False:
            i = 10
            return i + 15
        f.write(self.rombank_selected)
        f.write(self.rambank_selected)
        f.write(self.rambank_enabled)
        f.write(self.memorymodel)
        self.save_ram(f)
        if self.rtc_enabled:
            self.rtc.save_state(f)

    def load_state(self, f, state_version):
        if False:
            return 10
        self.rombank_selected = f.read()
        self.rambank_selected = f.read()
        self.rambank_enabled = f.read()
        self.memorymodel = f.read()
        self.load_ram(f)
        if self.rtc_enabled:
            self.rtc.load_state(f, state_version)

    def save_ram(self, f):
        if False:
            print('Hello World!')
        if not self.rambank_initialized:
            logger.warning('Saving RAM is not supported on {}'.format(self.carttype))
            return
        for bank in range(self.external_ram_count):
            for byte in range(8 * 1024):
                f.write(self.rambanks[bank, byte])
        logger.debug('RAM saved.')

    def load_ram(self, f):
        if False:
            while True:
                i = 10
        if not self.rambank_initialized:
            logger.warning('Loading RAM is not supported on {}'.format(self.carttype))
            return
        for bank in range(self.external_ram_count):
            for byte in range(8 * 1024):
                self.rambanks[bank, byte] = f.read()
        logger.debug('RAM loaded.')

    def init_rambanks(self, n):
        if False:
            for i in range(10):
                print('nop')
        self.rambank_initialized = True
        self.rambanks = memoryview(array.array('B', [0] * (8 * 1024 * 16))).cast('B', shape=(16, 8 * 1024))

    def getgamename(self, rombanks):
        if False:
            i = 10
            return i + 15
        return ''.join([chr(rombanks[0, x]) for x in range(308, 322)]).split('\x00')[0]

    def setitem(self, address, value):
        if False:
            return 10
        raise Exception('Cannot set item in MBC')

    def overrideitem(self, rom_bank, address, value):
        if False:
            for i in range(10):
                print('nop')
        if 0 <= address < 16384:
            logger.debug('Performing overwrite on address: %s:%s. New value: %s Old value: %s' % (hex(rom_bank), hex(address), hex(value), self.rombanks[rom_bank, address]))
            self.rombanks[rom_bank, address] = value
        else:
            logger.error('Invalid override address: %s' % hex(address))

    def getitem(self, address):
        if False:
            while True:
                i = 10
        if 0 <= address < 16384:
            return self.rombanks[0, address]
        elif 16384 <= address < 32768:
            return self.rombanks[self.rombank_selected, address - 16384]
        elif 40960 <= address < 49152:
            if not self.rambank_enabled:
                return 255
            if self.rtc_enabled and 8 <= self.rambank_selected <= 12:
                return self.rtc.getregister(self.rambank_selected)
            else:
                return self.rambanks[self.rambank_selected, address - 40960]

    def __repr__(self):
        if False:
            return 10
        return '\n'.join(['Cartridge:', 'Filename: %s' % self.filename, 'Game name: %s' % self.gamename, 'GB Color: %s' % str(self.ROMBanks[0, 323] == 128), 'Cartridge type: %s' % hex(self.cartType), 'Number of ROM banks: %s' % self.external_rom_count, 'Active ROM bank: %s' % self.rombank_selected, 'Number of RAM banks: %s' % len(self.rambanks), 'Active RAM bank: %s' % self.rambank_selected, 'Battery: %s' % self.battery, 'RTC: %s' % self.rtc])

class ROMOnly(BaseMBC):

    def setitem(self, address, value):
        if False:
            for i in range(10):
                print('nop')
        if 8192 <= address < 16384:
            if value == 0:
                value = 1
            self.rombank_selected = value & 1
            logger.debug('Switching bank 0x%0.4x, 0x%0.2x' % (address, value))
        elif 40960 <= address < 49152:
            self.rambanks[self.rambank_selected, address - 40960] = value