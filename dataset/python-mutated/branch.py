import re
from visidata import vd, Column, VisiData, ItemColumn, AttrColumn, Path, AttrDict, RowColorizer, date, Progress
from .gitsheet import GitSheet
vd.theme_option('color_git_current_branch', 'underline', 'color of current branch on branches sheet')
vd.theme_option('color_git_remote_branch', 'cyan', 'color of remote branches on branches sheet')

@VisiData.api
def git_branch(vd, p, args):
    if False:
        print('Hello World!')
    nonListArgs = '--track --no-track --set-upstream-to -u --unset-upstream -m -M -c -C -d -D --edit-description'.split()
    if any((x in args for x in nonListArgs)):
        return
    return GitBranch('git-branch-list', source=p, git_args=args)

def _remove_prefix(text, prefix):
    if False:
        while True:
            i = 10
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

class GitBranchNameColumn(Column):

    def calcValue(self, row):
        if False:
            for i in range(10):
                print('nop')
        return _remove_prefix(row.localbranch, 'remotes/')

    def putValue(self, row, val):
        if False:
            print('Hello World!')
        self.sheet.loggit('branch', '-v', '--move', row.localbranch, val)

class GitBranch(GitSheet):
    help = '\n        # git branch\n        List of all branches, including relevant metadata.\n\n        - `d` to mark a branch for deletion\n        - `e` on the _branch_ column to rename the branch\n        - `z Ctrl+S` to commit changes\n    '
    defer = True
    rowtype = 'branches'
    columns = [GitBranchNameColumn('branch', width=20), ItemColumn('head_commitid', 'refid', width=0), ItemColumn('tracking', 'remotebranch'), ItemColumn('upstream'), ItemColumn('merge_base', 'merge_name', width=20), ItemColumn('extra', width=0), ItemColumn('head_commitmsg', 'msg', width=50), ItemColumn('last_commit', type=date), ItemColumn('last_author')]
    colorizers = [RowColorizer(10, 'color_git_current_branch', lambda s, c, r, v: r and r['current']), RowColorizer(10, 'color_git_remote_branch', lambda s, c, r, v: r and r['localbranch'].startswith('remotes/'))]
    nKeys = 1

    def iterload(self):
        if False:
            print('Hello World!')
        branches_lines = self.git_lines('branch', '--list', '--format', ' '.join(('%(if)%(symref)%(then)yes%(else)no%(end)', '%(HEAD) %(refname:short) %(objectname:short)', '%(if)%(upstream)%(then)[%(upstream:short)', '%(if)%(upstream:track)%(then): %(upstream:track,nobracket)%(end)]', '%(end)', '%(contents:subject)')), '-vv', '--no-color', *self.git_args)
        for line in branches_lines:
            m = re.match('(?P<is_symref>(yes|no)?)\\s+\n                             (?P<current>\\*?)\\s+\n                             (?P<localbranch>\\S+)\\s+\n                             (?P<refid>\\w+)\\s+\n                             (?:\\[\n                               (?P<remotebranch>[^\\s\\]:]+):?\n                               \\s*(?P<extra>.*?)\n                             \\])?\n                             \\s*(?P<msg>.*)', line, re.VERBOSE)
            if not m:
                continue
            branch_details = AttrDict(m.groupdict())
            if branch_details.is_symref == 'yes':
                continue
            yield branch_details
        branch_stats = self.gitRootSheet.gitBranchStatuses
        for row in Progress(self.rows):
            merge_base = self.git_all('show-branch', '--merge-base', row.localbranch, self.gitRootSheet.branch, _ok_code=[0, 1]).strip()
            row.update(dict(merge_name=self.git_all('name-rev', '--name-only', merge_base).strip() if merge_base else '', upstream=branch_stats.get(row.localbranch), last_commit=self.git_all('show', '--no-patch', '--pretty=%ai', row.localbranch).strip(), last_author=self.git_all('show', '--no-patch', '--pretty=%an', row.localbranch).strip()))

    def commitAddRow(self, row):
        if False:
            return 10
        self.loggit('branch', row.localbranch)

    def commitDeleteRow(self, row):
        if False:
            for i in range(10):
                print('nop')
        self.loggit('branch', '--delete', _remove_prefix(row.localbranch, 'remotes/'))

@GitSheet.lazy_property
def gitBranchStatuses(sheet):
    if False:
        for i in range(10):
            print('nop')
    ret = {}
    for branch_status in sheet.git_lines('for-each-ref', '--format=%(refname:short) %(upstream:short) %(upstream:track)', 'refs/heads'):
        m = re.search('(\\S+)\\s*\n                          (\\S+)?\\s*\n                          (\\[\n                          (ahead.(\\d+)),?\\s*\n                          (behind.(\\d+))?\n                          \\])?', branch_status, re.VERBOSE)
        if not m:
            vd.status('unmatched branch status: ' + branch_status)
            continue
        (localb, remoteb, _, _, nahead, _, nbehind) = m.groups()
        if nahead:
            r = '+%s' % nahead
        else:
            r = ''
        if nbehind:
            if r:
                r += '/'
            r += '-%s' % nbehind
        ret[localb] = r
    return ret
GitSheet.addCommand('', 'git-open-branches', 'vd.push(git_branch(source, []))', 'push branches sheet')
GitSheet.addCommand('', 'git-branch-create', 'git("branch", input("create branch: ", type="branch"))', 'create a new branch off the current checkout')
GitBranch.addCommand('', 'git-branch-checkout', 'git("checkout", cursorRow.localbranch)', 'checkout this branch')
vd.addMenuItems('\n    Git > Branch > add > git-branch-create\n    Git > Branch > delete > git-branch-delete\n    Git > Branch > rename > git-branch-rename\n    Git > Branch > checkout > git-branch-checkout\n    Git > Open > branches > git-open-branches\n')