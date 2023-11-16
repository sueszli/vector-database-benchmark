import sys
import subprocess
from .test_public_api import PUBLIC_MODULES

def test_public_modules_importable():
    if False:
        print('Hello World!')
    pids = [subprocess.Popen([sys.executable, '-c', f'import {module}']) for module in PUBLIC_MODULES]
    for (i, pid) in enumerate(pids):
        assert pid.wait() == 0, f'Failed to import {PUBLIC_MODULES[i]}'