import os
import zipfile
from thefuck.utils import for_app
from thefuck.shells import shell

def _is_bad_zip(file):
    if False:
        while True:
            i = 10
    try:
        with zipfile.ZipFile(file, 'r') as archive:
            return len(archive.namelist()) > 1
    except Exception:
        return False

def _zip_file(command):
    if False:
        while True:
            i = 10
    for c in command.script_parts[1:]:
        if not c.startswith('-'):
            if c.endswith('.zip'):
                return c
            else:
                return u'{}.zip'.format(c)

@for_app('unzip')
def match(command):
    if False:
        print('Hello World!')
    if '-d' in command.script:
        return False
    zip_file = _zip_file(command)
    if zip_file:
        return _is_bad_zip(zip_file)
    else:
        return False

def get_new_command(command):
    if False:
        print('Hello World!')
    return u'{} -d {}'.format(command.script, shell.quote(_zip_file(command)[:-4]))

def side_effect(old_cmd, command):
    if False:
        for i in range(10):
            print('nop')
    with zipfile.ZipFile(_zip_file(old_cmd), 'r') as archive:
        for file in archive.namelist():
            if not os.path.abspath(file).startswith(os.getcwd()):
                continue
            try:
                os.remove(file)
            except OSError:
                pass
requires_output = False