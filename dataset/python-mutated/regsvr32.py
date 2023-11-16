from bottles.backend.logger import Logger
from bottles.backend.wine.wineprogram import WineProgram
logging = Logger()

class Regsvr32(WineProgram):
    program = 'Wine DLL Registration Server'
    command = 'regsvr32'

    def register(self, dll: str):
        if False:
            for i in range(10):
                print('nop')
        args = f'/s {dll}'
        return self.launch(args=args, communicate=True, action_name='register')

    def unregister(self, dll: str):
        if False:
            return 10
        args = f'/s /u {dll}'
        return self.launch(args=args, communicate=True, action_name='unregister')

    def register_all(self, dlls: list):
        if False:
            for i in range(10):
                print('nop')
        for dll in dlls:
            self.register(dll)

    def unregister_all(self, dlls: list):
        if False:
            print('Hello World!')
        for dll in dlls:
            self.unregister(dll)