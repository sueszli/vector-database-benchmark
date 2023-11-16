import os
import subprocess
import time
from bottles.backend.logger import Logger
from bottles.backend.utils.manager import ManagerUtils
from bottles.backend.utils.proc import ProcUtils
from bottles.backend.utils.steam import SteamUtils
from bottles.backend.wine.wineprogram import WineProgram
logging = Logger()

class WineServer(WineProgram):
    program = 'Wine Server'
    command = 'wineserver'

    def is_alive(self):
        if False:
            i = 10
            return i + 15
        config = self.config
        if not config.Runner:
            return False
        res = subprocess.Popen(['pgrep', 'wineserver'], stdout=subprocess.PIPE)
        if res.stdout.read() == b'':
            return False
        bottle = ManagerUtils.get_bottle_path(config)
        runner = ManagerUtils.get_runner_path(config.Runner)
        if config.Environment == 'Steam':
            bottle = config.Path
            runner = config.RunnerPath
        if SteamUtils.is_proton(runner):
            runner = SteamUtils.get_dist_directory(runner)
        env = os.environ.copy()
        env['WINEPREFIX'] = bottle
        env['PATH'] = f"{runner}/bin:{env['PATH']}"
        res = subprocess.Popen('wineserver -w', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, cwd=bottle, env=env)
        time.sleep(0.5)
        if res.poll() is None:
            res.kill()
            return True
        return False

    def wait(self):
        if False:
            for i in range(10):
                print('nop')
        config = self.config
        bottle = ManagerUtils.get_bottle_path(config)
        runner = ManagerUtils.get_runner_path(config.Runner)
        if config.Environment == 'Steam':
            bottle = config.Path
            runner = config.RunnerPath
        if SteamUtils.is_proton(runner):
            runner = SteamUtils.get_dist_directory(runner)
        env = os.environ.copy()
        env['WINEPREFIX'] = bottle
        env['PATH'] = f"{runner}/bin:{env['PATH']}"
        subprocess.Popen('wineserver -w', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, cwd=bottle, env=env).wait()

    def kill(self, signal: int=-1):
        if False:
            while True:
                i = 10
        args = '-k'
        if signal != -1:
            args += str(signal)
        self.launch(args=args, communicate=True, action_name='sending signal to the wine server')

    def force_kill(self):
        if False:
            while True:
                i = 10
        bottle = ManagerUtils.get_bottle_path(self.config)
        procs = ProcUtils.get_by_env(f'WINEPREFIX={bottle}')
        for proc in procs:
            proc.kill()
        if len(procs) == 0:
            self.kill(9)