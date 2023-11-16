import tarfile
import os
from thefuck.utils import for_app
from thefuck.shells import shell
tar_extensions = ('.tar', '.tar.Z', '.tar.bz2', '.tar.gz', '.tar.lz', '.tar.lzma', '.tar.xz', '.taz', '.tb2', '.tbz', '.tbz2', '.tgz', '.tlz', '.txz', '.tz')

def _is_tar_extract(cmd):
    if False:
        for i in range(10):
            print('nop')
    if '--extract' in cmd:
        return True
    cmd = cmd.split()
    return len(cmd) > 1 and 'x' in cmd[1]

def _tar_file(cmd):
    if False:
        for i in range(10):
            print('nop')
    for c in cmd:
        for ext in tar_extensions:
            if c.endswith(ext):
                return (c, c[0:len(c) - len(ext)])

@for_app('tar')
def match(command):
    if False:
        for i in range(10):
            print('nop')
    return '-C' not in command.script and _is_tar_extract(command.script) and (_tar_file(command.script_parts) is not None)

def get_new_command(command):
    if False:
        for i in range(10):
            print('nop')
    dir = shell.quote(_tar_file(command.script_parts)[1])
    return shell.and_('mkdir -p {dir}', '{cmd} -C {dir}').format(dir=dir, cmd=command.script)

def side_effect(old_cmd, command):
    if False:
        while True:
            i = 10
    with tarfile.TarFile(_tar_file(old_cmd.script_parts)[0]) as archive:
        for file in archive.getnames():
            if not os.path.abspath(file).startswith(os.getcwd()):
                continue
            try:
                os.remove(file)
            except OSError:
                pass