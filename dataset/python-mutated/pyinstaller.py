"""
This is an example of using PyInstaller packager to build
executable from one of CEF Python's examples (wxpython.py).

See README-pyinstaller.md for installing required packages.

To package example type: `python pyinstaller.py`.
"""
import os
import platform
import shutil
import sys
from subprocess import Popen
try:
    import PyInstaller
except ImportError:
    PyInstaller = None
    raise SystemExit('Error: PyInstaller package missing. To install type: pip install --upgrade pyinstaller')
EXE_EXT = ''
if platform.system() == 'Windows':
    EXE_EXT = '.exe'
elif platform.system() == 'Darwin':
    EXE_EXT = '.app'
elif platform.system() == 'Linux':
    EXE_EXT = ''

def main():
    if False:
        return 10
    if platform.system() not in ['Windows', 'Darwin', 'Linux']:
        raise SystemExit('Error: Only Windows, Linux and Darwin platforms are currently supported. See Issue #135 for details.')
    if os.path.exists('build/'):
        shutil.rmtree('build/')
    if os.path.exists('dist/'):
        shutil.rmtree('dist/')
    env = os.environ
    if '--debug' in sys.argv:
        env['CEFPYTHON_PYINSTALLER_DEBUG'] = '1'
    sub = Popen(['pyinstaller', '--clean', 'pyinstaller.spec'], env=env)
    sub.communicate()
    rcode = sub.returncode
    if rcode != 0:
        print('Error: PyInstaller failed, code=%s' % rcode)
        if os.path.exists('dist/'):
            shutil.rmtree('dist/')
        sys.exit(1)
    curdir = os.path.dirname(os.path.abspath(__file__))
    cefapp_dir = os.path.join(curdir, 'dist', 'cefapp')
    executable = os.path.join(cefapp_dir, 'cefapp' + EXE_EXT)
    if not os.path.exists(executable):
        print('Error: PyInstaller failed, main executable is missing: %s' % executable)
        sys.exit(1)
    print('OK. Created dist/ directory.')
    if platform.system() == 'Windows':
        if '--debug' in sys.argv:
            os.system('start cmd /k "%s"' % executable)
        else:
            os.system('%s/explorer.exe /n,/e,%s' % (os.environ['SYSTEMROOT'], cefapp_dir))
if __name__ == '__main__':
    main()