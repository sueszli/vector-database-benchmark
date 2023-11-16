import atexit
import fcntl
import logging
import os
import signal
import sys
from types import FrameType, TracebackType
from typing import NoReturn, Optional, Type

def daemonize_process(pid_file: str, logger: logging.Logger, chdir: str='/') -> None:
    if False:
        while True:
            i = 10
    'daemonize the current process\n\n    This calls fork(), and has the main process exit. When it returns we will be\n    running in the child process.\n    '
    if os.path.isfile(pid_file):
        with open(pid_file) as pid_fh:
            old_pid = pid_fh.read()
    try:
        lock_fh = open(pid_file, 'w')
    except OSError:
        print('Unable to create the pidfile.')
        sys.exit(1)
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print('Unable to lock on the pidfile.')
        with open(pid_file, 'w') as pid_fh:
            pid_fh.write(old_pid)
        sys.exit(1)
    process_id = os.fork()
    if process_id != 0:
        os._exit(0)
    os.setsid()
    devnull = '/dev/null'
    if hasattr(os, 'devnull'):
        devnull = os.devnull
    devnull_fd = os.open(devnull, os.O_RDWR)
    os.dup2(devnull_fd, 0)
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)

    def excepthook(type_: Type[BaseException], value: BaseException, traceback: Optional[TracebackType]) -> None:
        if False:
            print('Hello World!')
        logger.critical('Unhanded exception', exc_info=(type_, value, traceback))
    sys.excepthook = excepthook
    os.umask(23)
    os.chdir(chdir)
    try:
        lock_fh.write('%s' % os.getpid())
        lock_fh.flush()
    except OSError:
        logger.error('Unable to write pid to the pidfile.')
        print('Unable to write pid to the pidfile.')
        sys.exit(1)

    def sigterm(signum: int, frame: Optional[FrameType]) -> NoReturn:
        if False:
            while True:
                i = 10
        logger.warning('Caught signal %s. Stopping daemon.' % signum)
        sys.exit(0)
    signal.signal(signal.SIGTERM, sigterm)

    def exit() -> None:
        if False:
            return 10
        logger.warning('Stopping daemon.')
        os.remove(pid_file)
        sys.exit(0)
    atexit.register(exit)
    logger.warning('Starting daemon.')