from __future__ import absolute_import
import os
from bzrlib.commands import Command

class ExternalCommand(Command):
    """Class to wrap external commands."""

    @classmethod
    def find_command(cls, cmd):
        if False:
            print('Hello World!')
        import os.path
        bzrpath = os.environ.get('BZRPATH', '')
        for dir in bzrpath.split(os.pathsep):
            if not dir:
                continue
            path = os.path.join(dir, cmd)
            if os.path.isfile(path):
                return ExternalCommand(path)
        return None

    def __init__(self, path):
        if False:
            print('Hello World!')
        self.path = path

    def name(self):
        if False:
            while True:
                i = 10
        return os.path.basename(self.path)

    def run(self, *args, **kwargs):
        if False:
            return 10
        raise NotImplementedError('should not be called on %r' % self)

    def run_argv_aliases(self, argv, alias_argv=None):
        if False:
            i = 10
            return i + 15
        return os.spawnv(os.P_WAIT, self.path, [self.path] + argv)

    def help(self):
        if False:
            return 10
        m = 'external command from %s\n\n' % self.path
        pipe = os.popen('%s --help' % self.path)
        return m + pipe.read()