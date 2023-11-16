import os, runpy
import shutil
from setup import Command

class XWin(Command):
    description = 'Install the Windows headers for cross compilation'

    def run(self, opts):
        if False:
            while True:
                i = 10
        if not shutil.which('msiextract'):
            raise SystemExit('No msiextract found in PATH you may need to install msitools')
        base = os.path.dirname(self.SRC)
        m = runpy.run_path(os.path.join(base, 'setup', 'wincross.py'))
        cache_dir = os.path.join(base, '.build-cache', 'xwin')
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir)
        m['main'](['--dest', cache_dir])
        for x in os.listdir(cache_dir):
            if x != 'root':
                shutil.rmtree(os.path.join(cache_dir, x))