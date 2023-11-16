import os
import sys

def close_fds(keep_fds):
    if False:
        i = 10
        return i + 15
    'Close all the file descriptors except those in keep_fds.'
    keep_fds = {*keep_fds, 1, 2}
    try:
        open_fds = {int(fd) for fd in os.listdir('/proc/self/fd')}
    except FileNotFoundError:
        import resource
        max_nfds = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        open_fds = {*range(max_nfds)}
    for i in open_fds - keep_fds:
        try:
            os.close(i)
        except OSError:
            pass

def fork_exec(cmd, keep_fds, env=None):
    if False:
        while True:
            i = 10
    env = env or {}
    child_env = {**os.environ, **env}
    pid = os.fork()
    if pid == 0:
        close_fds(keep_fds)
        os.execve(sys.executable, cmd, child_env)
    else:
        return pid