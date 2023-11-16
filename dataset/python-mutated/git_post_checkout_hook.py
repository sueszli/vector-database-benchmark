__license__ = 'GPL v3'
__copyright__ = '2013, Kovid Goyal <kovid at kovidgoyal.net>'
import os, subprocess, sys
(prev_rev, current_rev, flags) = sys.argv[1:]

def get_branch_name(rev):
    if False:
        for i in range(10):
            print('nop')
    return subprocess.check_output(['git', 'name-rev', '--name-only', '--refs=refs/heads/*', rev]).decode('utf-8').strip()
base = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
os.chdir(base)
if flags == '1':
    (prev_branch, cur_branch) = list(map(get_branch_name, (prev_rev, current_rev)))
    rebase_in_progress = os.path.exists('.git/rebase-apply') or os.path.exists('.git/rebase-merge')
    subprocess.check_call('./setup.py gui --summary'.split())
    for (dirpath, dirnames, filenames) in os.walk('.'):
        for f in filenames:
            fpath = os.path.join(dirpath, f)
            if f.endswith('.pyc') and '/chroot/' not in fpath:
                os.remove(fpath)