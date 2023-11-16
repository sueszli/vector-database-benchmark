import os
import sys
import atexit
import logging
from subprocess import PIPE
CREATE_NEW_PROCESS_GROUP = 512
DETACHED_PROCESS = 8
REGISTERED = []

def start_detached(executable, *args):
    if False:
        for i in range(10):
            print('nop')
    'Starts a fully independent subprocess with no parent.\n    :param executable: executable\n    :param args: arguments to the executable,\n        eg: ["--param1_key=param1_val", "-vvv"]\n    :return: pid of the grandchild process '
    import multiprocessing
    (reader, writer) = multiprocessing.Pipe(False)
    multiprocessing.Process(target=_start_detached, args=(executable, *args), kwargs={'writer': writer}, daemon=True).start()
    pid = reader.recv()
    REGISTERED.append(pid)
    writer.close()
    reader.close()
    return pid

def _start_detached(executable, *args, writer=None):
    if False:
        for i in range(10):
            print('nop')
    kwargs = {}
    import platform
    from subprocess import Popen
    if platform.system() == 'Windows':
        kwargs.update(creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP)
    else:
        kwargs.update(start_new_session=True)
    p = Popen([executable, *args], stdin=PIPE, stdout=PIPE, stderr=PIPE, **kwargs)
    writer.send(p.pid)
    sys.exit()

def _cleanup():
    if False:
        i = 10
        return i + 15
    import signal
    for pid in REGISTERED:
        try:
            logging.getLogger(__name__).debug('cleaning up pid %d ' % pid)
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass
atexit.register(_cleanup)