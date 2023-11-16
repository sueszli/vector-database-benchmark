from pathlib import Path
import sys
import subprocess
import atexit
from PyInstaller.utils.tests import skip

@skip(reason='')
def test_hook_order(pyi_builder):
    if False:
        i = 10
        return i + 15
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', str(Path.with_name(__file__, 'hook_order_hooks'))])
    args = [sys.executable, '-m', 'pip', 'uninstall', 'pyi_example_package', '--yes', '-q', '-q', '-q']
    atexit.register(lambda : subprocess.run(args))
    pyi_builder.test_source('\n        try:\n            import pyi_example_package\n        except:\n            pass\n        ', pyi_args=['--additional-hooks-dir={}'.format(Path(__file__).with_name('hook_order_hooks'))])