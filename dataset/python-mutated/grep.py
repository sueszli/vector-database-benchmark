from visidata import vd, VisiData, Path, ColumnItem, ESC
from .gitsheet import GitSheet

@VisiData.api
def git_grep(vd, p, args):
    if False:
        while True:
            i = 10
    return GitGrep(args[0], regex=args[0], source=p)

class GitGrep(GitSheet):
    rowtype = 'results'
    help = '\n        # vgit grep\n        Each row on this sheet is a line matching the regex pattern `{sheet.regex}` in the tracked files of the current directory.\n\n        - `Ctrl+O` to open _{sheet.cursorRow[0]}:{sheet.cursorRow[1]}_ in the system editor; saved changes will be reflected automatically.\n    '
    columns = [ColumnItem('file', 0, help='filename of the match'), ColumnItem('line', 1, help='line number within file'), ColumnItem('text', 2, width=120, help='matching line of text')]
    nKeys = 2

    def iterload(self):
        if False:
            print('Hello World!')
        tmp = (self.topRowIndex, self.cursorRowIndex)
        for line in self.git_lines('grep', '--no-color', '-z', '--line-number', '--ignore-case', self.regex):
            yield list(line.split('\x00'))
        (self.topRowIndex, self.cursorRowIndex) = tmp
GitSheet.addCommand('g/', 'git-grep', 'rex=input("git grep: "); vd.push(GitGrep(rex, regex=rex, source=sheet))', 'find in all files in this repo')
GitGrep.addCommand('Ctrl+O', 'sysopen-row', 'launchExternalEditorPath(Path(cursorRow[0]), linenum=cursorRow[1]); reload()', 'open this file in $EDITOR')
GitGrep.bindkey('Enter', 'sysopen-row')