import errno
import os

def silentremove(filename):
    if False:
        i = 10
        return i + 15
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise

def remove_ps_flag(pid):
    if False:
        while True:
            i = 10
    silentremove('/tmp/paddle.%d.port' % pid)