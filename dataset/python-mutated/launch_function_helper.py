import os
import socket
import sys
import time
from contextlib import closing
from multiprocessing import Process

def launch_func(func, env_dict):
    if False:
        i = 10
        return i + 15
    for key in env_dict:
        os.environ[key] = env_dict[key]
    proc = Process(target=func)
    return proc

def wait(procs, timeout=30):
    if False:
        print('Hello World!')
    error = False
    begin = time.time()
    while True:
        alive = False
        for p in procs:
            p.join(timeout=10)
            if p.exitcode is None:
                alive = True
                continue
            elif p.exitcode != 0:
                error = True
                break
        if not alive:
            break
        if error:
            break
        if timeout is not None and time.time() - begin >= timeout:
            error = True
            break
    for p in procs:
        if p.is_alive():
            p.terminate()
    if error:
        sys.exit(1)

def _find_free_port(port_set):
    if False:
        while True:
            i = 10

    def __free_port():
        if False:
            while True:
                i = 10
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            return s.getsockname()[1]
    while True:
        port = __free_port()
        if port not in port_set:
            port_set.add(port)
            return port