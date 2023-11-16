import logging
from .base_mbc import BaseMBC
logger = logging.getLogger(__name__)

class MBC3(BaseMBC):

    def setitem(self, address, value):
        if False:
            while True:
                i = 10
        if 0 <= address < 8192:
            if value & 15 == 10:
                self.rambank_enabled = True
            elif value == 0:
                self.rambank_enabled = False
            else:
                self.rambank_enabled = False
        elif 8192 <= address < 16384:
            value &= 127
            if value == 0:
                value = 1
            self.rombank_selected = value % self.external_rom_count
        elif 16384 <= address < 24576:
            self.rambank_selected = value % self.external_ram_count
        elif 24576 <= address < 32768:
            if self.rtc_enabled:
                self.rtc.writecommand(value)
        elif 40960 <= address < 49152:
            if self.rambank_enabled:
                if self.rambank_selected <= 3:
                    self.rambanks[self.rambank_selected, address - 40960] = value
                elif 8 <= self.rambank_selected <= 12:
                    self.rtc.setregister(self.rambank_selected, value)