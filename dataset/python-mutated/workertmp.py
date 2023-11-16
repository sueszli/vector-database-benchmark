import os
import platform
import tempfile
from gunicorn import util
PLATFORM = platform.system()
IS_CYGWIN = PLATFORM.startswith('CYGWIN')

class WorkerTmp(object):

    def __init__(self, cfg):
        if False:
            return 10
        old_umask = os.umask(cfg.umask)
        fdir = cfg.worker_tmp_dir
        if fdir and (not os.path.isdir(fdir)):
            raise RuntimeError("%s doesn't exist. Can't create workertmp." % fdir)
        (fd, name) = tempfile.mkstemp(prefix='wgunicorn-', dir=fdir)
        os.umask(old_umask)
        if cfg.uid != os.geteuid() or cfg.gid != os.getegid():
            util.chown(name, cfg.uid, cfg.gid)
        try:
            if not IS_CYGWIN:
                util.unlink(name)
            self._tmp = os.fdopen(fd, 'w+b', 0)
        except Exception:
            os.close(fd)
            raise
        self.spinner = 0

    def notify(self):
        if False:
            i = 10
            return i + 15
        self.spinner = (self.spinner + 1) % 2
        os.fchmod(self._tmp.fileno(), self.spinner)

    def last_update(self):
        if False:
            i = 10
            return i + 15
        return os.fstat(self._tmp.fileno()).st_ctime

    def fileno(self):
        if False:
            while True:
                i = 10
        return self._tmp.fileno()

    def close(self):
        if False:
            i = 10
            return i + 15
        return self._tmp.close()