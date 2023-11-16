"""
This package contains the runtime hooks support code for when Salt is pacakged with PyInstaller.
"""
import io
import logging
import os
import subprocess
import sys
import salt.utils.vt
log = logging.getLogger(__name__)

def clean_pyinstaller_vars(environ):
    if False:
        for i in range(10):
            print('nop')
    '\n    Restore or cleanup PyInstaller specific environent variable behavior.\n    '
    if environ is None:
        environ = dict(os.environ)
    for varname in ('LD_LIBRARY_PATH', 'LIBPATH'):
        original_varname = '{}_ORIG'.format(varname)
        if varname in environ and environ[varname] == sys._MEIPASS:
            log.debug("User provided environment variable %r with value %r which is the value that PyInstaller set's. Removing it", varname, environ[varname])
            environ.pop(varname)
        if original_varname in environ and varname not in environ:
            log.debug('The %r variable was found in the passed environment, renaming it to %r', original_varname, varname)
            environ[varname] = environ.pop(original_varname)
        if varname not in environ:
            if original_varname in os.environ:
                log.debug('Renaming environment variable %r to %r', original_varname, varname)
                environ[varname] = os.environ[original_varname]
            elif varname in os.environ:
                log.debug('Setting environment variable %r to an empty string', varname)
                environ[varname] = ''
    return environ

class PyinstallerPopen(subprocess.Popen):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        kwargs['env'] = clean_pyinstaller_vars(kwargs.pop('env', None))
        super().__init__(*args, **kwargs)
    if sys.platform == 'win32' and (not isinstance(sys.stdout, io.IOBase)):

        def _get_handles(self, stdin, stdout, stderr):
            if False:
                print('Hello World!')
            (stdin, stdout, stderr) = (subprocess.DEVNULL if pipe is None else pipe for pipe in (stdin, stdout, stderr))
            return super()._get_handles(stdin, stdout, stderr)

class PyinstallerTerminal(salt.utils.vt.Terminal):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        kwargs['env'] = clean_pyinstaller_vars(kwargs.pop('env', None))
        super().__init__(*args, **kwargs)