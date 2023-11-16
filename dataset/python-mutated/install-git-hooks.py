""" Launcher for git hook installer tool.

"""
import os
import sys
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), '..')))
import stat
from nuitka.tools.Basics import goHome
from nuitka.Tracing import tools_logger
from nuitka.utils.Execution import getExecutablePath
from nuitka.utils.FileOperations import getFileContents, getWindowsShortPathName

def main():
    if False:
        return 10
    goHome()
    if os.name == 'nt':
        git_path = getExecutablePath('git')
        if git_path is None:
            git_path = 'C:\\Program Files\\Git\\bin\\sh.exe'
            if not os.path.exists(git_path):
                git_path = None
        if git_path is None:
            tools_logger.sysexit("Error, cannot locate 'git.exe' which we need to install git hooks. Add it to\nPATH while executing this will be sufficient.")
        for candidate in ('sh.exe', os.path.join('..', 'bin', 'sh.exe'), os.path.join('..', '..', 'bin', 'sh.exe')):
            sh_path = os.path.normpath(os.path.join(os.path.dirname(git_path), candidate))
            if os.path.exists(sh_path):
                break
        else:
            tools_logger.sysexit("Error, cannot locate 'sh.exe' near 'git.exe' which we need to install git hooks,\nplease improve this script.")
        sh_path = getWindowsShortPathName(sh_path)
    for hook in os.listdir('.githooks'):
        full_path = os.path.join('.githooks', hook)
        hook_contents = getFileContents(full_path)
        if hook_contents.startswith('#!/bin/sh'):
            if os.name == 'nt':
                hook_contents = '#!%s\n%s' % (sh_path.replace('\\', '/').replace(' ', '\\ '), hook_contents[10:])
            hook_contents = hook_contents.replace('./bin/autoformat-nuitka-source', "'%s' ./bin/autoformat-nuitka-source" % sys.executable)
        else:
            sys.exit('Error, unknown hook contents.')
        hook_target = os.path.join('.git/hooks/', hook)
        with open(hook_target, 'wb') as out_file:
            out_file.write(hook_contents.encode('utf8'))
        st = os.stat(hook_target)
        os.chmod(hook_target, st.st_mode | stat.S_IEXEC)
if __name__ == '__main__':
    main()