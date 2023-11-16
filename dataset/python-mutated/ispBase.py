"""
General interface for Isp based AVR programmers.
The ISP AVR programmer can load firmware into AVR chips. Which are commonly used on 3D printers.

 Needs to be subclassed to support different programmers.
 Currently only the stk500v2 subclass exists.
 This is a python 3 conversion of the code created by David Braam for the Cura project.
"""
from . import chipDB
from UM.Logger import Logger

class IspBase:
    """
    Base class for ISP based AVR programmers.
    Functions in this class raise an IspError when something goes wrong.
    """

    def programChip(self, flash_data):
        if False:
            print('Hello World!')
        ' Program a chip with the given flash data. '
        self.cur_ext_addr = -1
        self.chip = chipDB.getChipFromDB(self.getSignature())
        if not self.chip:
            raise IspError('Chip with signature: ' + str(self.getSignature()) + 'not found')
        self.chipErase()
        Logger.log('d', 'Flashing %i bytes', len(flash_data))
        self.writeFlash(flash_data)
        Logger.log('d', 'Verifying %i bytes', len(flash_data))
        self.verifyFlash(flash_data)
        Logger.log('d', 'Completed')

    def getSignature(self):
        if False:
            print('Hello World!')
        '\n        Get the AVR signature from the chip. This is a 3 byte array which describes which chip we are connected to.\n        This is important to verify that we are programming the correct type of chip and that we use proper flash block sizes.\n        '
        sig = []
        sig.append(self.sendISP([48, 0, 0, 0])[3])
        sig.append(self.sendISP([48, 0, 1, 0])[3])
        sig.append(self.sendISP([48, 0, 2, 0])[3])
        return sig

    def chipErase(self):
        if False:
            return 10
        '\n        Do a full chip erase, clears all data, and lockbits.\n        '
        self.sendISP([172, 128, 0, 0])

    def writeFlash(self, flash_data):
        if False:
            print('Hello World!')
        '\n        Write the flash data, needs to be implemented in a subclass.\n        '
        raise IspError('Called undefined writeFlash')

    def verifyFlash(self, flash_data):
        if False:
            return 10
        '\n        Verify the flash data, needs to be implemented in a subclass.\n        '
        raise IspError('Called undefined verifyFlash')

class IspError(Exception):

    def __init__(self, value):
        if False:
            while True:
                i = 10
        self.value = value

    def __str__(self):
        if False:
            print('Hello World!')
        return repr(self.value)