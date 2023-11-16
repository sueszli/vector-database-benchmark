"""Script to wait for ports to start listening.

Replaces logic previously implemented in a shell script that needed
tools that are not available on Windows.
"""
import argparse
import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple
import psutil

def _pid_exists(pid):
    if False:
        print('Hello World!')
    return pid in psutil.pids()

def _wait_for_port(port, server_pid, timeout) -> Tuple[bool, str]:
    if False:
        while True:
            i = 10
    start = time.time()
    print(f'Waiting for up to {timeout} seconds for port {port} to start listening.')
    while True:
        if not _pid_exists(server_pid):
            return (False, f'Server PID {server_pid} is not running.')
        try:
            socket.create_connection(('localhost', port), timeout=0.5)
            return (True, f'Port {port} is listening after {time.time() - start} seconds')
        except Exception:
            if time.time() - start > timeout:
                return (False, f'Port {port} still not listening after {timeout} seconds.')
            time.sleep(5)

def main():
    if False:
        while True:
            i = 10
    'Wait for ports to start listening.'
    parser = argparse.ArgumentParser(description='Wait for ports to start listening.')
    parser.add_argument('port', type=int, nargs='+')
    parser.add_argument('--timeout', type=int, required=True)
    parser.add_argument('--server-pid', type=int)
    args = parser.parse_args()
    executor = ThreadPoolExecutor(max_workers=len(args.port))
    futures = []
    for p in args.port:
        futures.append(executor.submit(_wait_for_port, p, args.server_pid, args.timeout))
    for f in as_completed(futures):
        (ok, msg) = f.result()
        if ok:
            print(f'OK: {msg}')
        else:
            print(f'FAIL: {msg}')
            exit(1)
if __name__ == '__main__':
    main()