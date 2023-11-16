import os
from bottles.backend.logger import Logger
from bottles.backend.wine.wineprogram import WineProgram
from bottles.backend.wine.wineserver import WineServer
logging = Logger()

class WineBridge(WineProgram):
    program = 'Wine Bridge'
    command = 'WineBridge.exe'
    is_internal = True
    internal_path = 'winebridge'

    def __wineserver_status(self):
        if False:
            while True:
                i = 10
        return WineServer(self.config).is_alive()

    def is_available(self):
        if False:
            return 10
        if os.path.isfile(self.get_command()):
            logging.info(f'{self.program} is available.')
            return True
        return False

    def get_procs(self):
        if False:
            for i in range(10):
                print('nop')
        args = 'getProcs'
        processes = []
        if not self.__wineserver_status:
            return processes
        res = self.launch(args=args, communicate=True, action_name='get_procs')
        if not res.ready:
            return processes
        lines = res.data.split('\n')
        for r in lines:
            if r in ['', '\r']:
                continue
            r = r.split('|')
            if len(r) < 3:
                continue
            processes.append({'pid': r[1], 'threads': r[2], 'name': r[0]})
        return processes

    def kill_proc(self, pid: str):
        if False:
            print('Hello World!')
        args = f'killProc {pid}'
        return self.launch(args=args, communicate=True, action_name='kill_proc')

    def kill_proc_by_name(self, name: str):
        if False:
            for i in range(10):
                print('nop')
        args = f'killProcByName {name}'
        return self.launch(args=args, communicate=True, action_name='kill_proc_by_name')

    def run_exe(self, exec_path: str):
        if False:
            print('Hello World!')
        args = f'runExe {exec_path}'
        return self.launch(args=args, communicate=True, action_name='run_exe')