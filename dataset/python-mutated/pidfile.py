import errno
import os
import tempfile

class Pidfile(object):
    """    Manage a PID file. If a specific name is provided
    it and '"%s.oldpid" % name' will be used. Otherwise
    we create a temp file using os.mkstemp.
    """

    def __init__(self, fname):
        if False:
            while True:
                i = 10
        self.fname = fname
        self.pid = None

    def create(self, pid):
        if False:
            i = 10
            return i + 15
        oldpid = self.validate()
        if oldpid:
            if oldpid == os.getpid():
                return
            msg = "Already running on PID %s (or pid file '%s' is stale)"
            raise RuntimeError(msg % (oldpid, self.fname))
        self.pid = pid
        fdir = os.path.dirname(self.fname)
        if fdir and (not os.path.isdir(fdir)):
            raise RuntimeError("%s doesn't exist. Can't create pidfile." % fdir)
        (fd, fname) = tempfile.mkstemp(dir=fdir)
        os.write(fd, ('%s\n' % self.pid).encode('utf-8'))
        if self.fname:
            os.rename(fname, self.fname)
        else:
            self.fname = fname
        os.close(fd)
        os.chmod(self.fname, 420)

    def rename(self, path):
        if False:
            for i in range(10):
                print('nop')
        self.unlink()
        self.fname = path
        self.create(self.pid)

    def unlink(self):
        if False:
            print('Hello World!')
        ' delete pidfile'
        try:
            with open(self.fname, 'r') as f:
                pid1 = int(f.read() or 0)
            if pid1 == self.pid:
                os.unlink(self.fname)
        except Exception:
            pass

    def validate(self):
        if False:
            return 10
        ' Validate pidfile and make it stale if needed'
        if not self.fname:
            return
        try:
            with open(self.fname, 'r') as f:
                try:
                    wpid = int(f.read())
                except ValueError:
                    return
                try:
                    os.kill(wpid, 0)
                    return wpid
                except OSError as e:
                    if e.args[0] == errno.EPERM:
                        return wpid
                    if e.args[0] == errno.ESRCH:
                        return
                    raise
        except IOError as e:
            if e.args[0] == errno.ENOENT:
                return
            raise