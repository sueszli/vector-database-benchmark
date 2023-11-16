import atexit
import errno
import os
import tempfile
import time
from calibre.constants import cache_dir, iswindows
from calibre.ptempfile import remove_dir
from calibre.utils.monotonic import monotonic
TDIR_LOCK = 'tdir-lock'
if iswindows:
    from calibre.utils.lock import windows_open

    def lock_tdir(path):
        if False:
            for i in range(10):
                print('nop')
        return windows_open(os.path.join(path, TDIR_LOCK))

    def unlock_file(fobj):
        if False:
            while True:
                i = 10
        fobj.close()

    def remove_tdir(path, lock_file):
        if False:
            for i in range(10):
                print('nop')
        lock_file.close()
        remove_dir(path)

    def is_tdir_locked(path):
        if False:
            print('Hello World!')
        try:
            with windows_open(os.path.join(path, TDIR_LOCK)):
                pass
        except OSError:
            return True
        return False
else:
    import fcntl
    from calibre.utils.ipc import eintr_retry_call

    def lock_tdir(path):
        if False:
            while True:
                i = 10
        lf = os.path.join(path, TDIR_LOCK)
        f = open(lf, 'w')
        eintr_retry_call(fcntl.lockf, f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return f

    def unlock_file(fobj):
        if False:
            return 10
        from calibre.utils.ipc import eintr_retry_call
        eintr_retry_call(fcntl.lockf, fobj.fileno(), fcntl.LOCK_UN)
        fobj.close()

    def remove_tdir(path, lock_file):
        if False:
            i = 10
            return i + 15
        lock_file.close()
        remove_dir(path)

    def is_tdir_locked(path):
        if False:
            print('Hello World!')
        lf = os.path.join(path, TDIR_LOCK)
        f = open(lf, 'w')
        try:
            eintr_retry_call(fcntl.lockf, f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            eintr_retry_call(fcntl.lockf, f.fileno(), fcntl.LOCK_UN)
            return False
        except OSError:
            return True
        finally:
            f.close()

def tdirs_in(b):
    if False:
        while True:
            i = 10
    try:
        tdirs = os.listdir(b)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise
        tdirs = ()
    for x in tdirs:
        x = os.path.join(b, x)
        if os.path.isdir(x):
            yield x

def clean_tdirs_in(b):
    if False:
        for i in range(10):
            print('nop')
    for q in tdirs_in(b):
        if not is_tdir_locked(q):
            remove_dir(q)

def retry_lock_tdir(path, timeout=30, sleep=0.1):
    if False:
        print('Hello World!')
    st = monotonic()
    while True:
        try:
            return lock_tdir(path)
        except Exception:
            if monotonic() - st > timeout:
                raise
            time.sleep(sleep)

def tdir_in_cache(base):
    if False:
        while True:
            i = 10
    ' Create a temp dir inside cache_dir/base. The created dir is robust\n    against application crashes. i.e. it will be cleaned up the next time the\n    application starts, even if it was left behind by a previous crash. '
    b = os.path.join(os.path.realpath(cache_dir()), base)
    try:
        os.makedirs(b)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    global_lock = retry_lock_tdir(b)
    try:
        if b not in tdir_in_cache.scanned:
            tdir_in_cache.scanned.add(b)
            try:
                clean_tdirs_in(b)
            except Exception:
                import traceback
                traceback.print_exc()
        tdir = tempfile.mkdtemp(dir=b)
        lock_data = lock_tdir(tdir)
        atexit.register(remove_tdir, tdir, lock_data)
        tdir = os.path.join(tdir, 'a')
        os.mkdir(tdir)
        return tdir
    finally:
        unlock_file(global_lock)
tdir_in_cache.scanned = set()