import itertools
import os
import select
import signal
import socket
import threading
import time
if os.name == 'nt':
    import cffi
    _ffi = cffi.FFI()
    _ffi.cdef('int raise(int);')
    _lib = _ffi.dlopen('api-ms-win-crt-runtime-l1-1-0.dll')
    signal_raise = getattr(_lib, 'raise')
else:

    def signal_raise(signum):
        if False:
            print('Hello World!')
        signal.pthread_kill(threading.get_ident(), signum)

def raise_SIGINT_soon():
    if False:
        return 10
    time.sleep(1)
    signal_raise(signal.SIGINT)

def drain(sock):
    if False:
        i = 10
        return i + 15
    total = 0
    try:
        while True:
            total += len(sock.recv(1024))
    except BlockingIOError:
        pass
    return total

def main():
    if False:
        while True:
            i = 10
    (writer, reader) = socket.socketpair()
    writer.setblocking(False)
    reader.setblocking(False)
    signal.set_wakeup_fd(writer.fileno())
    for attempt in itertools.count():
        print(f'Attempt {attempt}: start')
        drained = drain(reader)
        if drained:
            print(f'Attempt {attempt}: ({drained} residual bytes discarded)')
        thread = threading.Thread(target=raise_SIGINT_soon)
        thread.start()
        start = time.perf_counter()
        target = start + 10
        try:
            select_calls = 0
            drained = 0
            while True:
                now = time.perf_counter()
                if now > target:
                    break
                select_calls += 1
                (r, _, _) = select.select([reader], [], [], target - now)
                if r:
                    drained += drain(reader)
        except KeyboardInterrupt:
            pass
        else:
            print(f'Attempt {attempt}: no KeyboardInterrupt?!')
        duration = time.perf_counter() - start
        if duration < 2:
            print(f'Attempt {attempt}: OK, trying again (select_calls = {select_calls}, drained = {drained})')
        else:
            print(f'Attempt {attempt}: FAILED, took {duration} seconds')
            print(f'select_calls = {select_calls}, drained = {drained}')
            break
        thread.join()
if __name__ == '__main__':
    main()