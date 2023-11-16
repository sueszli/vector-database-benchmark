"""Data used by setup.py and the PyInstaller qutebrowser.spec."""
import sys
import os
import os.path
import subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
if sys.hexversion >= 50331648:
    open_file = open
else:
    import codecs
    open_file = codecs.open
BASEDIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir)

def _call_git(gitpath, *args):
    if False:
        while True:
            i = 10
    'Call a git subprocess.'
    return subprocess.run(['git'] + list(args), cwd=gitpath, check=True, stdout=subprocess.PIPE, text=True).stdout.strip()

def _git_str():
    if False:
        for i in range(10):
            print('nop')
    "Try to find out git version.\n\n    Return:\n        string containing the git commit ID and timestamp.\n        None if there was an error or we're not in a git repo.\n    "
    if BASEDIR is None:
        return None
    if not os.path.isdir(os.path.join(BASEDIR, '.git')):
        return None
    try:
        commit_hash = _call_git(BASEDIR, 'describe', '--match=NeVeRmAtCh', '--always', '--dirty')
        date = _call_git(BASEDIR, 'show', '-s', '--format=%ci', 'HEAD')
        branch = _call_git(BASEDIR, 'rev-parse', '--abbrev-ref', 'HEAD')
        return '{} on {} ({})'.format(commit_hash, branch, date)
    except (subprocess.CalledProcessError, OSError):
        return None

def write_git_file():
    if False:
        while True:
            i = 10
    'Write the git-commit-id file with the current commit.'
    gitstr = _git_str()
    if gitstr is None:
        gitstr = ''
    path = os.path.join(BASEDIR, 'qutebrowser', 'git-commit-id')
    with open_file(path, 'w', encoding='ascii') as f:
        f.write(gitstr)