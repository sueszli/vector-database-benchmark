from visidata import vd, VisiData, ItemColumn, AttrDict, RowColorizer, Path
from .gitsheet import GitSheet

@VisiData.api
def git_remote(vd, p, args):
    if False:
        while True:
            i = 10
    if not args or 'show' in args:
        return GitRemotes('remotes', source=p)

class GitRemotes(GitSheet):
    help = '\n        # git remote\n        Manage the set of repositories ("remotes") whose branches you track.\n\n        - `a` to add a remote\n        - `d` to mark a remote for deletion\n        - `e` to edit the _remote_ or _url_\n        - `z Ctrl+S` to commit the changes.\n    '
    rowtypes = 'remotes'
    columns = [ItemColumn('remote', setter=lambda c, r, v: c.sheet.set_remote(c, r, v)), ItemColumn('type'), ItemColumn('url', width=40, setter=lambda c, r, v: c.sheet.set_url(c, r, v))]
    nKeys = 1
    defer = True

    def set_remote(self, col, row, val):
        if False:
            return 10
        self.loggit('remote', 'rename', self.column('remote').getSourceValue(row), val)

    def set_url(self, col, row, val):
        if False:
            return 10
        self.loggit('remote', 'set-url', row.remote, val)

    def iterload(self):
        if False:
            print('Hello World!')
        for line in self.git_lines('remote', '-v', 'show'):
            (name, url, paren_type) = line.split()
            yield AttrDict(remote=name, url=url, type=paren_type[1:-1])

    def commitDeleteRow(self, row):
        if False:
            return 10
        self.loggit('remote', 'remove', row.remote)

    def commitAddRow(self, row):
        if False:
            return 10
        row.remote = self.column('remote').getValue(row)
        row.url = self.column('url').getValue(row)
        self.loggit('remote', 'add', row.remote, row.url)

    def newRow(self):
        if False:
            for i in range(10):
                print('nop')
        return AttrDict()
GitSheet.addCommand('', 'git-open-remotes', 'vd.push(git_remote(Path("."), ""))', 'open git remotes sheet')
vd.addMenuItems('\n    Git > Open > remotes > git-open-remotes\n')