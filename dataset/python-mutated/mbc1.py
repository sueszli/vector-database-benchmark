import logging
from .base_mbc import BaseMBC
logger = logging.getLogger(__name__)

class MBC1(BaseMBC):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        self.bank_select_register1 = 1
        self.bank_select_register2 = 0

    def setitem(self, address, value):
        if False:
            return 10
        if 0 <= address < 8192:
            self.rambank_enabled = value & 15 == 10
        elif 8192 <= address < 16384:
            value &= 31
            if value == 0:
                value = 1
            self.bank_select_register1 = value
        elif 16384 <= address < 24576:
            self.bank_select_register2 = value & 3
        elif 24576 <= address < 32768:
            self.memorymodel = value & 1
        elif 40960 <= address < 49152:
            if self.rambank_enabled:
                self.rambank_selected = self.bank_select_register2 if self.memorymodel == 1 else 0
                self.rambanks[self.rambank_selected % self.external_ram_count, address - 40960] = value

    def getitem(self, address):
        if False:
            print('Hello World!')
        if 0 <= address < 16384:
            if self.memorymodel == 1:
                self.rombank_selected = (self.bank_select_register2 << 5) % self.external_rom_count
            else:
                self.rombank_selected = 0
            return self.rombanks[self.rombank_selected, address]
        elif 16384 <= address < 32768:
            self.rombank_selected = (self.bank_select_register2 << 5 | self.bank_select_register1) % self.external_rom_count
            return self.rombanks[self.rombank_selected, address - 16384]
        elif 40960 <= address < 49152:
            if not self.rambank_initialized:
                logger.error('RAM banks not initialized: %s' % hex(address))
            if not self.rambank_enabled:
                return 255
            if self.memorymodel == 1:
                self.rambank_selected = self.bank_select_register2 % self.external_ram_count
            else:
                self.rambank_selected = 0
            return self.rambanks[self.rambank_selected, address - 40960]

    def save_state(self, f):
        if False:
            i = 10
            return i + 15
        BaseMBC.save_state(self, f)
        f.write(self.bank_select_register1)
        f.write(self.bank_select_register2)

    def load_state(self, f, state_version):
        if False:
            i = 10
            return i + 15
        BaseMBC.load_state(self, f, state_version)
        if state_version >= 3:
            self.bank_select_register1 = f.read()
            self.bank_select_register2 = f.read()
        else:
            self.bank_select_register1 = self.rombank_selected & 31
            self.bank_select_register2 = (self.rombank_selected & 96) >> 5
            self.rambank_selected = self.bank_select_register2