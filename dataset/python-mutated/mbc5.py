import logging
from .base_mbc import BaseMBC
logger = logging.getLogger(__name__)

class MBC5(BaseMBC):

    def setitem(self, address, value):
        if False:
            print('Hello World!')
        if 0 <= address < 8192:
            self.rambank_enabled = value == 10
        elif 8192 <= address < 12288:
            self.rombank_selected = (self.rombank_selected & 256 | value) % self.external_rom_count
        elif 12288 <= address < 16384:
            self.rombank_selected = ((value & 1) << 8 | self.rombank_selected & 255) % self.external_rom_count
        elif 16384 <= address < 24576:
            self.rambank_selected = (value & 15) % self.external_ram_count
        elif 40960 <= address < 49152:
            if self.rambank_enabled:
                self.rambanks[self.rambank_selected, address - 40960] = value
        else:
            logger.debug('Unexpected write to 0x%0.4x, value: 0x%0.2x' % (address, value))