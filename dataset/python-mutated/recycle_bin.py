__license__ = 'GPL v3'
__copyright__ = '2010, Kovid Goyal <kovid@kovidgoyal.net>'
__docformat__ = 'restructuredtext en'
import os, shutil, time, sys
from calibre import isbytestring
from calibre.constants import iswindows, ismacos, filesystem_encoding, islinux
recycle = None
if iswindows:
    from calibre.utils.ipc import eintr_retry_call
    from threading import Lock
    from calibre_extensions import winutil
    recycler = None
    rlock = Lock()

    def start_recycler():
        if False:
            for i in range(10):
                print('nop')
        global recycler
        if recycler is None:
            from calibre.utils.ipc.simple_worker import start_pipe_worker
            recycler = start_pipe_worker('from calibre.utils.recycle_bin import recycler_main; recycler_main()')

    def recycle_path(path):
        if False:
            for i in range(10):
                print('nop')
        winutil.move_to_trash(str(path))

    def recycler_main():
        if False:
            return 10
        stdin = getattr(sys.stdin, 'buffer', sys.stdin)
        stdout = getattr(sys.stdout, 'buffer', sys.stdout)
        while True:
            path = eintr_retry_call(stdin.readline)
            if not path:
                break
            try:
                path = path.decode('utf-8').rstrip()
            except (ValueError, TypeError):
                break
            try:
                recycle_path(path)
            except:
                eintr_retry_call(stdout.write, b'KO\n')
                stdout.flush()
                try:
                    import traceback
                    traceback.print_exc()
                except Exception:
                    pass
            else:
                eintr_retry_call(stdout.write, b'OK\n')
                stdout.flush()

    def delegate_recycle(path):
        if False:
            print('Hello World!')
        if '\n' in path:
            raise ValueError('Cannot recycle paths that have newlines in them (%r)' % path)
        with rlock:
            start_recycler()
            recycler.stdin.write(path.encode('utf-8'))
            recycler.stdin.write(b'\n')
            recycler.stdin.flush()
            result = eintr_retry_call(recycler.stdout.readline)
            if result.rstrip() != b'OK':
                raise RuntimeError('recycler failed to recycle: %r' % path)

    def recycle(path):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(path, bytes):
            path = path.decode(filesystem_encoding)
        path = os.path.abspath(path)
        return delegate_recycle(path)
elif ismacos:
    from calibre_extensions.cocoa import send2trash

    def osx_recycle(path):
        if False:
            for i in range(10):
                print('nop')
        if isbytestring(path):
            path = path.decode(filesystem_encoding)
        send2trash(path)
    recycle = osx_recycle
elif islinux:
    from calibre.utils.linux_trash import send2trash

    def fdo_recycle(path):
        if False:
            for i in range(10):
                print('nop')
        if isbytestring(path):
            path = path.decode(filesystem_encoding)
        path = os.path.abspath(path)
        send2trash(path)
    recycle = fdo_recycle
can_recycle = callable(recycle)

def nuke_recycle():
    if False:
        return 10
    global can_recycle
    can_recycle = False

def restore_recyle():
    if False:
        return 10
    global can_recycle
    can_recycle = callable(recycle)

def delete_file(path, permanent=False):
    if False:
        i = 10
        return i + 15
    if not permanent and can_recycle:
        try:
            recycle(path)
            return
        except:
            import traceback
            traceback.print_exc()
    os.remove(path)

def delete_tree(path, permanent=False):
    if False:
        i = 10
        return i + 15
    if permanent:
        try:
            shutil.rmtree(path)
        except:
            import traceback
            traceback.print_exc()
            time.sleep(1)
            shutil.rmtree(path)
    else:
        if can_recycle:
            try:
                recycle(path)
                return
            except:
                import traceback
                traceback.print_exc()
        delete_tree(path, permanent=True)