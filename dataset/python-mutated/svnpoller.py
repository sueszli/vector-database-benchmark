import os
import xml.dom.minidom
from urllib.parse import quote_plus as urlquote_plus
from twisted.internet import defer
from twisted.python import log
from buildbot import util
from buildbot.changes import base
from buildbot.util import bytes2unicode
from buildbot.util import runprocess

def split_file_alwaystrunk(path):
    if False:
        for i in range(10):
            print('nop')
    return {'path': path}

def split_file_branches(path):
    if False:
        return 10
    pieces = path.split('/')
    if len(pieces) > 1 and pieces[0] == 'trunk':
        return (None, '/'.join(pieces[1:]))
    elif len(pieces) > 2 and pieces[0] == 'branches':
        return ('/'.join(pieces[0:2]), '/'.join(pieces[2:]))
    return None

def split_file_projects_branches(path):
    if False:
        for i in range(10):
            print('nop')
    if '/' not in path:
        return None
    (project, path) = path.split('/', 1)
    f = split_file_branches(path)
    if f:
        info = {'project': project, 'path': f[1]}
        if f[0]:
            info['branch'] = f[0]
        return info
    return f

class SVNPoller(base.ReconfigurablePollingChangeSource, util.ComparableMixin):
    """
    Poll a Subversion repository for changes and submit them to the change
    master.
    """
    compare_attrs = ('repourl', 'split_file', 'svnuser', 'svnpasswd', 'project', 'pollInterval', 'histmax', 'svnbin', 'category', 'cachepath', 'pollAtLaunch', 'pollRandomDelayMin', 'pollRandomDelayMax')
    secrets = ('svnuser', 'svnpasswd')
    parent = None
    last_change = None
    loop = None

    def __init__(self, repourl, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        name = kwargs.get('name', None)
        if name is None:
            kwargs['name'] = repourl
        super().__init__(repourl, **kwargs)

    def checkConfig(self, repourl, split_file=None, svnuser=None, svnpasswd=None, pollInterval=10 * 60, histmax=100, svnbin='svn', revlinktmpl='', category=None, project='', cachepath=None, pollinterval=-2, extra_args=None, name=None, pollAtLaunch=False, pollRandomDelayMin=0, pollRandomDelayMax=0):
        if False:
            for i in range(10):
                print('nop')
        if pollinterval != -2:
            pollInterval = pollinterval
        if name is None:
            name = repourl
        super().checkConfig(name=name, pollInterval=pollInterval, pollAtLaunch=pollAtLaunch, pollRandomDelayMin=pollRandomDelayMin, pollRandomDelayMax=pollRandomDelayMax)

    @defer.inlineCallbacks
    def reconfigService(self, repourl, split_file=None, svnuser=None, svnpasswd=None, pollInterval=10 * 60, histmax=100, svnbin='svn', revlinktmpl='', category=None, project='', cachepath=None, pollinterval=-2, extra_args=None, name=None, pollAtLaunch=False, pollRandomDelayMin=0, pollRandomDelayMax=0):
        if False:
            return 10
        if pollinterval != -2:
            pollInterval = pollinterval
        if name is None:
            name = repourl
        if repourl.endswith('/'):
            repourl = repourl[:-1]
        self.repourl = repourl
        self.extra_args = extra_args
        self.split_file = split_file or split_file_alwaystrunk
        self.svnuser = svnuser
        self.svnpasswd = svnpasswd
        self.revlinktmpl = revlinktmpl
        self.environ = os.environ.copy()
        self.svnbin = svnbin
        self.histmax = histmax
        self._prefix = None
        self.category = category if callable(category) else util.bytes2unicode(category)
        self.project = util.bytes2unicode(project)
        self.cachepath = cachepath
        if self.cachepath and os.path.exists(self.cachepath):
            try:
                with open(self.cachepath, 'r', encoding='utf-8') as f:
                    self.last_change = int(f.read().strip())
                    log.msg(f'SVNPoller: SVNPoller({self.repourl}) setting last_change to {self.last_change}')
                with open(self.cachepath, 'w', encoding='utf-8') as f:
                    f.write(str(self.last_change))
            except Exception:
                self.cachepath = None
                log.msg(('SVNPoller: SVNPoller({}) cache file corrupt or unwriteable; ' + 'skipping and not using').format(self.repourl))
                log.err()
        yield super().reconfigService(name=name, pollInterval=pollInterval, pollAtLaunch=pollAtLaunch, pollRandomDelayMin=pollRandomDelayMin, pollRandomDelayMax=pollRandomDelayMax)

    def describe(self):
        if False:
            while True:
                i = 10
        return f'SVNPoller: watching {self.repourl}'

    def poll(self):
        if False:
            i = 10
            return i + 15
        if self.project:
            log.msg('SVNPoller: polling ' + self.project)
        else:
            log.msg('SVNPoller: polling')
        d = defer.succeed(None)
        if not self._prefix:
            d.addCallback(lambda _: self.get_prefix())

            @d.addCallback
            def set_prefix(prefix):
                if False:
                    return 10
                self._prefix = prefix
        d.addCallback(self.get_logs)
        d.addCallback(self.parse_logs)
        d.addCallback(self.get_new_logentries)
        d.addCallback(self.create_changes)
        d.addCallback(self.submit_changes)
        d.addCallback(self.finished_ok)
        d.addErrback(log.err, 'SVNPoller: Error in  while polling')
        return d

    @defer.inlineCallbacks
    def get_prefix(self):
        if False:
            for i in range(10):
                print('nop')
        command = [self.svnbin, 'info', '--xml', '--non-interactive', self.repourl]
        if self.svnuser:
            command.append(f'--username={self.svnuser}')
        if self.svnpasswd is not None:
            command.append(f'--password={self.svnpasswd}')
        if self.extra_args:
            command.extend(self.extra_args)
        (rc, output) = (yield runprocess.run_process(self.master.reactor, command, env=self.environ, collect_stderr=False, stderr_is_error=True))
        if rc != 0:
            raise EnvironmentError(f'{self}: Got error when retrieving svn prefix')
        try:
            doc = xml.dom.minidom.parseString(output)
        except xml.parsers.expat.ExpatError:
            log.msg(f"SVNPoller: SVNPoller.get_prefix: ExpatError in '{output}'")
            raise
        rootnodes = doc.getElementsByTagName('root')
        if not rootnodes:
            self._prefix = ''
            return self._prefix
        rootnode = rootnodes[0]
        root = ''.join([c.data for c in rootnode.childNodes])
        if not self.repourl.startswith(root):
            log.msg(format='Got root %(root)r from `svn info`, but it is not a prefix of the configured repourl', repourl=self.repourl, root=root)
            raise RuntimeError("Configured repourl doesn't match svn root")
        prefix = self.repourl[len(root):]
        if prefix.startswith('/'):
            prefix = prefix[1:]
        log.msg(f'SVNPoller: repourl={self.repourl}, root={root}, so prefix={prefix}')
        return prefix

    @defer.inlineCallbacks
    def get_logs(self, _):
        if False:
            for i in range(10):
                print('nop')
        command = [self.svnbin, 'log', '--xml', '--verbose', '--non-interactive']
        if self.svnuser:
            command.extend([f'--username={self.svnuser}'])
        if self.svnpasswd is not None:
            command.extend([f'--password={self.svnpasswd}'])
        if self.extra_args:
            command.extend(self.extra_args)
        command.extend([f'--limit={self.histmax}', self.repourl])
        (rc, output) = (yield runprocess.run_process(self.master.reactor, command, env=self.environ, collect_stderr=False, stderr_is_error=True))
        if rc != 0:
            raise EnvironmentError(f'{self}: Got error when retrieving svn logs')
        return output

    def parse_logs(self, output):
        if False:
            return 10
        try:
            doc = xml.dom.minidom.parseString(output)
        except xml.parsers.expat.ExpatError:
            log.msg(f"SVNPoller: SVNPoller.parse_logs: ExpatError in '{output}'")
            raise
        logentries = doc.getElementsByTagName('logentry')
        return logentries

    def get_new_logentries(self, logentries):
        if False:
            for i in range(10):
                print('nop')
        last_change = old_last_change = self.last_change
        new_last_change = None
        new_logentries = []
        if logentries:
            new_last_change = int(logentries[0].getAttribute('revision'))
            if last_change is None:
                log.msg(f'SVNPoller: starting at change {new_last_change}')
            elif last_change == new_last_change:
                log.msg('SVNPoller: no changes')
            else:
                for el in logentries:
                    if last_change == int(el.getAttribute('revision')):
                        break
                    new_logentries.append(el)
                new_logentries.reverse()
        self.last_change = new_last_change
        log.msg(f'SVNPoller: _process_changes {old_last_change} .. {new_last_change}')
        return new_logentries

    def _get_text(self, element, tag_name):
        if False:
            while True:
                i = 10
        try:
            child_nodes = element.getElementsByTagName(tag_name)[0].childNodes
            text = ''.join([t.data for t in child_nodes])
        except IndexError:
            text = 'unknown'
        return text

    def _transform_path(self, path):
        if False:
            i = 10
            return i + 15
        if not path.startswith(self._prefix):
            log.msg(format="SVNPoller: ignoring path '%(path)s' which doesn'tstart with prefix '%(prefix)s'", path=path, prefix=self._prefix)
            return None
        relative_path = path[len(self._prefix):]
        if relative_path.startswith('/'):
            relative_path = relative_path[1:]
        where = self.split_file(relative_path)
        if not where:
            return None
        if isinstance(where, tuple):
            where = {'branch': where[0], 'path': where[1]}
        return where

    def create_changes(self, new_logentries):
        if False:
            while True:
                i = 10
        changes = []
        for el in new_logentries:
            revision = str(el.getAttribute('revision'))
            revlink = ''
            if self.revlinktmpl and revision:
                revlink = self.revlinktmpl % urlquote_plus(revision)
                revlink = str(revlink)
            log.msg(f'Adding change revision {revision}')
            author = self._get_text(el, 'author')
            comments = self._get_text(el, 'msg')
            branches = {}
            try:
                pathlist = el.getElementsByTagName('paths')[0]
            except IndexError:
                log.msg('ignoring commit with no paths')
                continue
            for p in pathlist.getElementsByTagName('path'):
                kind = p.getAttribute('kind')
                action = p.getAttribute('action')
                path = ''.join([t.data for t in p.childNodes])
                if path.startswith('/'):
                    path = path[1:]
                if kind == 'dir' and (not path.endswith('/')):
                    path += '/'
                where = self._transform_path(path)
                if where:
                    branch = where.get('branch', None)
                    filename = where['path']
                    if branch not in branches:
                        branches[branch] = {'files': [], 'number_of_directories': 0}
                    if filename == '':
                        branches[branch]['files'].append(filename)
                        branches[branch]['number_of_directories'] += 1
                    elif filename.endswith('/'):
                        branches[branch]['files'].append(filename[:-1])
                        branches[branch]['number_of_directories'] += 1
                    else:
                        branches[branch]['files'].append(filename)
                    if 'action' not in branches[branch]:
                        branches[branch]['action'] = action
                    for key in ('repository', 'project', 'codebase'):
                        if key in where:
                            branches[branch][key] = where[key]
            for (branch, info) in branches.items():
                action = info['action']
                files = info['files']
                number_of_directories_changed = info['number_of_directories']
                number_of_files_changed = len(files)
                if action == 'D' and number_of_directories_changed == 1 and (number_of_files_changed == 1) and (files[0] == ''):
                    log.msg(f"Ignoring deletion of branch '{branch}'")
                else:
                    chdict = {'author': author, 'committer': None, 'files': [bytes2unicode(f, 'utf-8', 'replace') for f in files], 'comments': comments, 'revision': revision, 'branch': util.bytes2unicode(branch), 'revlink': revlink, 'category': self.category, 'repository': util.bytes2unicode(info.get('repository', self.repourl)), 'project': util.bytes2unicode(info.get('project', self.project)), 'codebase': util.bytes2unicode(info.get('codebase', None))}
                    changes.append(chdict)
        return changes

    @defer.inlineCallbacks
    def submit_changes(self, changes):
        if False:
            print('Hello World!')
        for chdict in changes:
            yield self.master.data.updates.addChange(src='svn', **chdict)

    def finished_ok(self, res):
        if False:
            for i in range(10):
                print('nop')
        if self.cachepath:
            with open(self.cachepath, 'w', encoding='utf-8') as f:
                f.write(str(self.last_change))
        log.msg(f'SVNPoller: finished polling {res}')
        return res