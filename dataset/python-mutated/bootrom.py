import array
import os
import struct

class BootROM:

    def __init__(self, bootrom_file, cgb):
        if False:
            return 10
        if bootrom_file == 'pyboy_fast':
            self.bootrom = array.array('B', [0] * 256)
            self.bootrom[0] = 49
            self.bootrom[1] = 254
            self.bootrom[2] = 255
            self.bootrom[3] = 195
            self.bootrom[4] = 252
            self.bootrom[5] = 0
            self.bootrom[252] = 62
            self.bootrom[253] = 1
            self.bootrom[254] = 224
            self.bootrom[255] = 80
            return
        if bootrom_file is None:
            rom = '/bootrom_cgb.bin' if cgb else '/bootrom_dmg.bin'
            bootrom_file = os.path.dirname(os.path.realpath(__file__)) + rom
        with open(bootrom_file, 'rb') as f:
            rom = f.read()
        self.bootrom = array.array('B', struct.unpack('%iB' % len(rom), rom))

    def getitem(self, addr):
        if False:
            return 10
        return self.bootrom[addr]