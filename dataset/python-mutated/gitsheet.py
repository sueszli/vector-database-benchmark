import io
from visidata import AttrDict, vd, Path, asyncthread, Sheet

class GitSheet(Sheet):

    @property
    def gitargstr(self):
        if False:
            return 10
        return ' '.join(self.gitargs)

    def git(self, subcmd, *args, **kwargs):
        if False:
            print('Hello World!')
        'For non-modifying commands; not logged except in debug mode'
        sh = vd.importExternal('sh')
        args = list(subcmd.split()) + list(args)
        vd.debug('git ' + ' '.join((str(x) for x in args)))
        return sh.git(*args, _cwd=self.gitRootPath, **kwargs)

    def loggit(self, subcmd, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Run git command with *args*, and post a status message.'
        import sh
        args = list(subcmd.split()) + list(args)
        vd.warning('git ' + ' '.join((str(x) for x in args)))
        return sh.git(*args, _cwd=self.gitRootPath, **kwargs)

    def git_all(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Return entire output of git command.'
        sh = vd.importExternal('sh')
        try:
            vd.debug('git ' + ' '.join((str(x) for x in args)))
            out = self.git('--no-pager', *args, _decode_errors='replace', _bg_exc=False, **kwargs)
        except sh.ErrorReturnCode as e:
            vd.warning('git ' + ' '.join((str(x) for x in args)), 'error=%s' % e.exit_code)
            out = e.stdout
        return out

    def git_lines(self, subcmd, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Generator of stdout lines from given git command'
        sh = vd.importExternal('sh')
        err = io.StringIO()
        args = list(subcmd.split()) + list(args)
        try:
            vd.debug('git ' + ' '.join((str(x) for x in args)))
            for line in self.git('--no-pager', *args, _decode_errors='replace', _iter=True, _bg_exc=False, _err=err, **kwargs):
                yield line[:-1]
        except sh.ErrorReturnCode as e:
            vd.warning('git ' + ' '.join((str(x) for x in args)), 'error=%s' % e.exit_code)
        errlines = err.getvalue().splitlines()
        if errlines:
            vd.warning('git stderr: ' + '\n'.join(errlines))

    def git_iter(self, subcmd, *args, sep='\x00', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Generator of chunks of stdout from given git command *subcmd*, delineated by sep character.'
        sh = vd.importExternal('sh')
        import sh
        err = io.StringIO()
        args = list(subcmd.split()) + list(args)
        bufsize = 512
        chunks = []
        try:
            vd.debug('git ' + ' '.join((str(x) for x in args)))
            for data in self.git('--no-pager', *args, _decode_errors='replace', _out_bufsize=bufsize, _iter=True, _bg_exc=False, _err=err, **kwargs):
                while True:
                    i = data.find(sep)
                    if i < 0:
                        break
                    chunks.append(data[:i])
                    data = data[i + 1:]
                    yield ''.join(chunks)
                    chunks.clear()
                chunks.append(data)
        except sh.ErrorReturnCode as e:
            vd.warning('git ' + ' '.join((str(x) for x in args)), 'error=%s' % e.exit_code)
        if chunks:
            yield ''.join(chunks)
        errlines = err.getvalue().splitlines()
        if errlines:
            vd.warning('git stderr: ' + '\n'.join(errlines))

    @asyncthread
    def modifyGit(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Run git command that modifies the repo'
        vd.warning('git ' + ' '.join((str(x) for x in args)))
        ret = self.git_all(*args, **kwargs)
        vd.status(ret)
        if isinstance(self.source, GitSheet):
            self.source.reload()
        self.reload()

    @property
    def gitRootSheet(self):
        if False:
            i = 10
            return i + 15
        if isinstance(self.source, GitSheet):
            return self.source.gitRootSheet
        return self

    def iterload(self):
        if False:
            i = 10
            return i + 15
        for line in self.git_lines(*self.gitargs):
            yield AttrDict(line=line)

@GitSheet.lazy_property
def gitRootPath(self):
    if False:
        print('Hello World!')
    'Return Path of git root (nearest ancestor directory with a .git/)'

    def _getRepoPath(p):
        if False:
            while True:
                i = 10
        'Return path at p or above which has .git subdir'
        if p.joinpath('.git').exists():
            return p
        if str(p) in ['/', '']:
            return None
        return _getRepoPath(p.resolve().parent)
    p = _getRepoPath(self.gitRootSheet.source)
    if p:
        return p

@GitSheet.lazy_property
def branch(self):
    if False:
        return 10
    return self.git('rev-parse', '--abbrev-ref', 'HEAD').strip()
GitSheet.options.disp_note_none = ''
GitSheet.options.disp_status_fmt = '{sheet.progressStatus}‹{sheet.branchStatus}› {sheet.name}| '
GitSheet.addCommand('gi', 'git-exec', 'cmdstr=input("gi", type="git"); vd.push(GitSheet(cmdstr, gitargs=cmdstr.split()))', 'execute git command')
GitSheet.addCommand('Alt+g', 'menu-git', 'pressMenu("Git")', '')
vd.addMenuItems('\n    Git > Execute command > git-exec\n')