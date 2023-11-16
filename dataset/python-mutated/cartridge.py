import logging
from array import array
from .base_mbc import ROMOnly
from .mbc1 import MBC1
from .mbc2 import MBC2
from .mbc3 import MBC3
from .mbc5 import MBC5
logger = logging.getLogger(__name__)

def load_cartridge(filename):
    if False:
        i = 10
        return i + 15
    rombanks = load_romfile(filename)
    if not validate_checksum(rombanks):
        raise Exception('Cartridge header checksum mismatch!')
    external_ram_count = int(EXTERNAL_RAM_TABLE[rombanks[0, 329]])
    carttype = rombanks[0, 327]
    cartinfo = CARTRIDGE_TABLE.get(carttype, None)
    if cartinfo is None:
        raise Exception('Catridge type invalid: %s' % carttype)
    cartdata = (carttype, cartinfo[0].__name__, ', '.join([x for (x, y) in zip(['SRAM', 'Battery', 'RTC'], cartinfo[1:]) if y]))
    logger.debug('Cartridge type: 0x%0.2x - %s, %s' % cartdata)
    logger.debug('Cartridge size: %d ROM banks of 16KB, %s RAM banks of 8KB' % (len(rombanks), external_ram_count))
    cartmeta = CARTRIDGE_TABLE[carttype]
    return cartmeta[0](filename, rombanks, external_ram_count, carttype, *cartmeta[1:])

def validate_checksum(rombanks):
    if False:
        for i in range(10):
            print('nop')
    x = 0
    for m in range(308, 333):
        x = x - rombanks[0, m] - 1
        x &= 255
    return rombanks[0, 333] == x

def load_romfile(filename):
    if False:
        i = 10
        return i + 15
    with open(filename, 'rb') as romfile:
        romdata = array('B', romfile.read())
    logger.debug(f'Loading ROM file: {len(romdata)} bytes')
    if len(romdata) == 0:
        logger.error('ROM file is empty!')
        raise Exception('Empty ROM file')
    banksize = 16 * 1024
    if len(romdata) % banksize != 0:
        logger.error('Unexpected ROM file length')
        raise Exception('Bad ROM file size')
    return memoryview(romdata).cast('B', shape=(len(romdata) // banksize, banksize))
CARTRIDGE_TABLE = {0: (ROMOnly, False, False, False), 1: (MBC1, False, False, False), 2: (MBC1, True, False, False), 3: (MBC1, True, True, False), 5: (MBC2, False, False, False), 6: (MBC2, False, True, False), 8: (ROMOnly, True, False, False), 9: (ROMOnly, True, True, False), 15: (MBC3, False, True, True), 16: (MBC3, True, True, True), 17: (MBC3, False, False, False), 18: (MBC3, True, False, False), 19: (MBC3, True, True, False), 25: (MBC5, False, False, False), 26: (MBC5, True, False, False), 27: (MBC5, True, True, False), 28: (MBC5, False, False, False), 29: (MBC5, True, False, False), 30: (MBC5, True, True, False)}
EXTERNAL_RAM_TABLE = {0: 1, 1: 1, 2: 1, 3: 4, 4: 16, 5: 8}