import logging
from .base_mbc import BaseMBC
logger = logging.getLogger(__name__)

class MBC2(BaseMBC):

    def setitem(self, address, value):
        if False:
            for i in range(10):
                print('nop')
        if 0 <= address < 16384:
            value &= 15
            if address & 256 == 0:
                self.rambank_enabled = value == 10
            else:
                if value == 0:
                    value = 1
                self.rombank_selected = value % self.external_rom_count
        elif 40960 <= address < 49152:
            if self.rambank_enabled:
                self.rambanks[0, address % 512] = value | 240

    def getitem(self, address):
        if False:
            return 10
        if 0 <= address < 16384:
            return self.rombanks[0, address]
        elif 16384 <= address < 32768:
            return self.rombanks[self.rombank_selected, address - 16384]
        elif 40960 <= address < 49152:
            if not self.rambank_initialized:
                logger.error('RAM banks not initialized: %s' % hex(address))
            if not self.rambank_enabled:
                return 255
            else:
                return self.rambanks[0, address % 512] | 240