import subprocess
from ..utils import RepoStats, ThreadedSegment, get_git_subprocess_env

def get_stash_count():
    if False:
        i = 10
        return i + 15
    try:
        p = subprocess.Popen(['git', 'stash', 'list'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=get_git_subprocess_env())
    except OSError:
        return 0
    pdata = p.communicate()
    if p.returncode != 0:
        return 0
    return pdata[0].count(b'\n')

class Segment(ThreadedSegment):

    def run(self):
        if False:
            print('Hello World!')
        self.stash_count = get_stash_count()

    def add_to_powerline(self):
        if False:
            for i in range(10):
                print('nop')
        self.join()
        if not self.stash_count:
            return
        bg = self.powerline.theme.GIT_STASH_BG
        fg = self.powerline.theme.GIT_STASH_FG
        sc = self.stash_count if self.stash_count > 1 else ''
        stash_str = u' {}{} '.format(sc, RepoStats.symbols['stash'])
        self.powerline.append(stash_str, fg, bg)