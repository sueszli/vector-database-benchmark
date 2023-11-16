from bottles.backend.logger import Logger
from bottles.backend.wine.wineprogram import WineProgram
logging = Logger()

class Eject(WineProgram):
    program = 'Wine Eject CLI'
    command = 'eject'

    def cdrom(self, drive: str, unmount_only: bool=False):
        if False:
            for i in range(10):
                print('nop')
        args = drive
        if unmount_only:
            args += ' -u'
        return self.launch(args=args, communicate=True, action_name='cdrom')

    def all(self):
        if False:
            return 10
        args = '-a'
        return self.launch(args=args, communicate=True, action_name='all')