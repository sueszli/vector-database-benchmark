from typing import Optional
from bottles.backend.logger import Logger
from bottles.backend.wine.wineprogram import WineProgram
logging = Logger()

class Uninstaller(WineProgram):
    program = 'Wine Uninstaller'
    command = 'uninstaller'

    def get_uuid(self, name: Optional[str]=None):
        if False:
            for i in range(10):
                print('nop')
        args = ' --list'
        if name is not None:
            args = f"--list | grep -i '{name}' | cut -f1 -d\\|"
        return self.launch(args=args, communicate=True, action_name='get_uuid')

    def from_uuid(self, uuid: Optional[str]=None):
        if False:
            print('Hello World!')
        args = ''
        if uuid not in [None, '']:
            args = f'--remove {uuid}'
        return self.launch(args=args, action_name='from_uuid')

    def from_name(self, name: str):
        if False:
            return 10
        res = self.get_uuid(name)
        if not res.ready:
            '\n            No UUID found, at this point it is safe to assume that the\n            program is not installed\n            ref: <https://github.com/bottlesdevs/Bottles/issues/2237>\n            '
            return
        uuid = res.data.strip()
        for _uuid in uuid.splitlines():
            self.from_uuid(_uuid)