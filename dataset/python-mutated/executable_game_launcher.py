from serpent.game_launcher import GameLauncher, GameLauncherException
from serpent.utilities import is_linux, is_windows
import shlex
import subprocess

class ExecutableGameLauncher(GameLauncher):

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)

    def launch(self, **kwargs):
        if False:
            while True:
                i = 10
        executable_path = kwargs.get('executable_path')
        if executable_path is None:
            raise GameLauncherException("An 'executable_path' kwarg is required...")
        if is_linux():
            subprocess.Popen(shlex.split(executable_path))
        elif is_windows():
            subprocess.Popen(shlex.split(executable_path))