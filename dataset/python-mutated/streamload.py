"""Read all test files for a particular file format using a single
importer instance. Read them again in reversed order. This is used
to verify that a loader does proper cleanup and can be called
repeatedly."""
import sys
import os
import subprocess
sys.path.append(os.path.join('..', 'regression'))
import utils
import settings

def process_dir(thisdir):
    if False:
        return 10
    'Process /thisdir/ recursively'
    res = []
    shellparams = {'stdin': subprocess.PIPE, 'stdout': sys.stdout, 'shell': True}
    command = [utils.assimp_bin_path, 'testbatchload']
    for f in os.listdir(thisdir):
        if os.path.splitext(f)[-1] in settings.exclude_extensions:
            continue
        fullpath = os.path.join(thisdir, f)
        if os.path.isdir(fullpath):
            if f != '.svn':
                res += process_dir(fullpath)
            continue
        command.append(fullpath)
        command.append(fullpath)
    if len(command) > 2:
        command += reversed(command[2:])
        if subprocess.call(command, **shellparams):
            res.append(thisdir)
    return res

def main():
    if False:
        i = 10
        return i + 15
    'Run the test on all registered test repositories'
    utils.find_assimp_or_die()
    res = []
    for tp in settings.model_directories:
        res += process_dir(tp)
    [print(f) for f in res]
    return 0
if __name__ == '__main__':
    res = main()
    input('All done, waiting for keystroke ')
    sys.exit(res)