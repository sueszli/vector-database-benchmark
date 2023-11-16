from visidata import vd, Column, VisiData, ItemColumn, Path, AttrDict, BaseSheet, IndexSheet
from visidata import RowColorizer, CellColorizer
from visidata import filesize, modtime, date
from .gitsheet import GitSheet
vd.option('vgit_show_ignored', False, 'show ignored files on git status')
vd.theme_option('color_git_staged_mod', 'green', 'color of files staged with modifications')
vd.theme_option('color_git_staged_add', 'green', 'color of files staged for addition')
vd.theme_option('color_git_staged_del', 'red', 'color of files staged for deletion')
vd.theme_option('color_git_unstaged_del', 'on 88', 'color of files deleted but unstaged')
vd.theme_option('color_git_untracked', '243 blue', 'color of ignored/untracked files')

@VisiData.api
def git_status(vd, p, args, **kwargs):
    if False:
        i = 10
        return i + 15
    vs = GitStatus('/'.join(p.parts[-2:]), source=p)
    if not vs.gitRootPath:
        return vd.git_repos(p, [])
    return vs

class GitFile:

    def __init__(self, path, gitsrc):
        if False:
            print('Hello World!')
        self.path = path
        self.filename = path.relative_to(gitsrc)
        self.is_dir = self.path.is_dir()

    def __str__(self):
        if False:
            print('Hello World!')
        return str(self.filename) + (self.is_dir and '/' or '')

class GitStatus(GitSheet):
    rowtype = 'files'
    help = '\n        # git status\n        An overview of the local git checkout.\n\n        - `Enter` to open diff of file (`git diff`)\n        - `a` to stage changes in file (`git add`)\n        - `r` to unstage changes in file (`git reset`)\n        - `c` to revert all unstaged changes in file (`git checkout`)\n        - `d` to stage the entire file for deletion (`git rm`)\n        - `z Ctrl+S` to commit staged changes (`git commit`)\n    '
    columns = [Column('path', width=40, getter=lambda c, r: str(r)), Column('status', getter=lambda c, r: c.sheet.statusText(c.sheet.git_status(r)), width=8), Column('status_raw', getter=lambda c, r: c.sheet.git_status(r), width=0), Column('staged', getter=lambda c, r: c.sheet.git_status(r).dels), Column('unstaged', getter=lambda c, r: c.sheet.git_status(r).adds), Column('type', getter=lambda c, r: r.is_dir() and '/' or r.suffix, width=0), Column('size', type=int, getter=lambda c, r: filesize(r)), Column('modtime', type=date, getter=lambda c, r: modtime(r))]
    nKeys = 1
    colorizers = [CellColorizer(3, 'color_git_staged_mod', lambda s, c, r, v: r and c and (c.name == 'staged') and (s.git_status(r).status[0] == 'M')), CellColorizer(1, 'color_git_staged_del', lambda s, c, r, v: r and c and (c.name == 'staged') and (s.git_status(r).status == 'D ')), RowColorizer(1, 'color_git_staged_add', lambda s, c, r, v: r and s.git_status(r).status in ['A ', 'M ']), RowColorizer(1, 'color_git_unstaged_del', lambda s, c, r, v: r and s.git_status(r).status[1] == 'D'), RowColorizer(3, 'color_git_untracked', lambda s, c, r, v: r and s.git_status(r).status == '!!'), RowColorizer(1, 'color_git_untracked', lambda s, c, r, v: r and s.git_status(r).status == '??')]

    def statusText(self, st):
        if False:
            for i in range(10):
                print('nop')
        vmod = {'A': 'add', 'D': 'rm', 'M': 'mod', 'T': 'chmod', '?': '', '!': 'ignored', 'U': 'unmerged'}
        (x, y) = st.status
        if st == '??':
            return 'new'
        elif st == '!!':
            return 'ignored'
        elif x != ' ' and y == ' ':
            return vmod.get(x, x)
        elif y != ' ':
            return vmod.get(y, y)
        else:
            return ''

    @property
    def workdir(self):
        if False:
            i = 10
            return i + 15
        return str(self.source)

    def git_status(self, r):
        if False:
            i = 10
            return i + 15
        'return tuple of (status, adds, dels).\n        status like !! ??\n        adds and dels are lists of additions and deletions.\n        '
        if not r:
            return None
        fn = str(r)
        ret = self._cachedStatus.get(fn, None)
        if not ret:
            ret = AttrDict(status='??')
            self._cachedStatus[fn] = ret
        return ret

    def ignored(self, fn):
        if False:
            print('Hello World!')
        if self.options.vgit_show_ignored:
            return False
        if fn in self._cachedStatus:
            return self._cachedStatus[fn].status == '!!'
        return False

    @property
    def remotediff(self):
        if False:
            while True:
                i = 10
        return self.gitBranchStatuses.get(self.branch, 'no branch')

    def iterload(self):
        if False:
            i = 10
            return i + 15
        files = [GitFile(p, self.source) for p in self.source.iterdir() if p.name not in '.git']
        filenames = dict(((gf.filename, gf) for gf in files))
        self._cachedStatus.clear()
        for fn in self.git_iter('ls-files', '-z'):
            self._cachedStatus[fn] = AttrDict(status='  ')
        for line in self.git_iter('status', '-z', '-unormal', '--ignored'):
            if not line:
                continue
            if line[2:3] == ' ':
                (st, fn) = (line[:2], line[3:])
            else:
                fn = line
                st = '??'
            self._cachedStatus[fn] = AttrDict(status=st)
            if not self.ignored(fn):
                yield Path(fn)
        for line in self.git_iter('diff-files', '--numstat', '-z'):
            if not line:
                continue
            (adds, dels, fn) = line.split('\t')
            if fn not in self._cachedStatus:
                self._cachedStatus[fn] = AttrDict(status='##')
            cs = self._cachedStatus[fn]
            cs.adds = '+%s/-%s' % (adds, dels)
        for line in self.git_iter('diff-index', '--cached', '--numstat', '-z', 'HEAD'):
            if not line:
                continue
            (adds, dels, fn) = line.split('\t')
            if fn not in self._cachedStatus:
                self._cachedStatus[fn] = AttrDict(status='$$')
            cs = self._cachedStatus[fn]
            cs.dels = '+%s/-%s' % (adds, dels)
        self.orderBy(None, self.columns[-1], reverse=True)
        self.recalc()

    def openRow(self, row):
        if False:
            while True:
                i = 10
        'Open unstaged diffs for this file, or dive into directory'
        if row.is_dir:
            return GitStatus(row.path)
        else:
            return DifferSheet(row, 'HEAD', 'index', 'working', source=sheet)

    def openRows(self, rows):
        if False:
            while True:
                i = 10
        'Open unstaged hunks for selected rows'
        return getHunksSheet(sheet, *rows)

@GitStatus.lazy_property
def _cachedStatus(self):
    if False:
        while True:
            i = 10
    return {}
GitStatus.addCommand('a', 'git-add', 'loggit("add", cursorRow.filename)', 'add this new file or modified file to staging')
GitStatus.addCommand('d', 'git-rm', 'loggit("rm", cursorRow.filename)', 'stage this file for deletion')
GitStatus.addCommand('r', 'git-reset', 'loggit("reset", "HEAD", cursorRow.filename)', 'reset/unstage this file')
GitStatus.addCommand('c', 'git-checkout', 'loggit("checkout", cursorRow.filename)', 'checkout this file')
GitStatus.addCommand('ga', 'git-add-selected', 'loggit("add", *[r for r in selectedRows])', 'add all selected files to staging')
GitStatus.addCommand('gd', 'git-rm-selected', 'loggit("rm", *[r for r in selectedRows])', 'delete all selected files')
GitStatus.addCommand(None, 'git-commit', 'loggit("commit", "-m", input("commit message: "))', 'commit changes')
GitStatus.addCommand(None, 'git-ignore-file', 'open(rootPath/".gitignore", "a").write(cursorRow.filename+"\\n"); reload()', 'add file to toplevel .gitignore')
GitStatus.addCommand(None, 'git-ignore-wildcard', 'open(rootPath/.gitignore, "a").write(input("add wildcard to .gitignore: "))', 'add input line to toplevel .gitignore')
vd.addMenuItems('\n    Git > View staged changes > current file > diff-file-staged\n    Git > View staged changes > selected files > staged changes > diff-selected-staged\n    Git > Stage > current file > git-add\n    Git > Stage > selected files > git-add-selected\n    Git > Unstage > current file > git-reset\n    Git > Unstage > selected files > git-reset-selected\n    Git > Rename file > git-mv\n    Git > Delete > file > git-rm\n    Git > Delete > selected files > git-rm-selected\n    Git > Ignore > file > ignore-file\n    Git > Ignore > wildcard > ignore-wildcard\n    Git > Commit staged changes > git-commit\n    Git > Revert unstaged changes > current file > git-checkout\n')