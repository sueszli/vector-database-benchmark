"""
Tests related to inter-process communication (for services).

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import contextlib
import pickle
import random
import socket
import sys
import threading
import time
import psutil
import pytest
import retrying
from fiftyone.service.ipc import IPCServer, send_request
from fiftyone.service.util import find_processes_by_args, get_listening_tcp_ports, normalize_wrapper_process, send_ipc_message
current_process = psutil.Process()

def list_current_ports():
    if False:
        print('Hello World!')
    return list(get_listening_tcp_ports(current_process))

@contextlib.contextmanager
def SingleRequestHandler(server):
    if False:
        return 10
    t = threading.Thread(target=server.handle_request)
    t.start()
    try:
        yield
    finally:
        server.stop()
        t.join()

@contextlib.contextmanager
def MultiRequestHandler(server):
    if False:
        print('Hello World!')
    t = threading.Thread(target=server.serve_forever)
    t.start()
    try:
        yield
    finally:
        server.stop()
        t.join()

def test_one_request():
    if False:
        i = 10
        return i + 15
    with IPCServer(lambda x: x * 2) as server, SingleRequestHandler(server):
        assert send_request(server.port, 5) == 10

def test_multiple_requests():
    if False:
        while True:
            i = 10
    with IPCServer(lambda x: x * 2) as server, MultiRequestHandler(server):
        assert send_request(server.port, 5) == 10
        assert send_request(server.port, 'a') == 'aa'

def test_bad_request():
    if False:
        i = 10
        return i + 15
    with IPCServer(lambda _: None) as server, SingleRequestHandler(server):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('localhost', server.port))
        s.send(b'foo')
        res = pickle.loads(s.recv(2048))
        assert isinstance(res, pickle.UnpicklingError)

def test_large_request():
    if False:
        while True:
            i = 10
    with IPCServer(lambda x: x) as server, SingleRequestHandler(server):
        data = list(range(10000))
        assert send_request(server.port, data) == data

def test_timeout():
    if False:
        i = 10
        return i + 15
    with IPCServer(lambda _: None) as server:
        server.timeout = 1
        timeout_called = threading.Event()
        server.handle_timeout = timeout_called.set
        with SingleRequestHandler(server):
            time.sleep(server.timeout + 0.5)
        assert timeout_called.is_set()

def test_stop_single():
    if False:
        return 10
    requests = []
    with IPCServer(requests.append) as server, SingleRequestHandler(server):
        server.timeout = 1
        server.stop()
        with pytest.raises(socket.error):
            send_request(server.port, 5)
    assert not requests

def test_stop_multi():
    if False:
        while True:
            i = 10
    requests = []
    with IPCServer(requests.append) as server, MultiRequestHandler(server):
        send_request(server.port, 1)
        assert requests == [1]
        server.stop()
        with pytest.raises(socket.error):
            send_request(server.port, 2)
    assert requests == [1]

def test_run_in_background():
    if False:
        print('Hello World!')
    requests = []
    with IPCServer.run_in_background(requests.append) as server:
        send_request(server.port, 2)
        send_request(server.port, 3)
    assert requests == [2, 3]

def test_find_processes_by_args():
    if False:
        return 10
    assert current_process in list(find_processes_by_args(current_process.cmdline()))
    random_arg = str(5 + random.random())
    p = psutil.Popen([sys.executable, '-c', 'import sys, time; time.sleep(float(sys.argv[1]))', random_arg])

    @retrying.retry(stop_max_delay=2000)
    def _check():
        if False:
            print('Hello World!')
        assert normalize_wrapper_process(p) in list(find_processes_by_args([random_arg]))
    try:
        _check()
    finally:
        p.kill()

def test_get_listening_tcp_ports():
    if False:
        for i in range(10):
            print('nop')
    assert not list_current_ports()
    with IPCServer(lambda _: None) as server:
        assert list_current_ports() == [server.port]
    assert not list_current_ports()

def test_send_ipc_message():
    if False:
        for i in range(10):
            print('nop')
    with IPCServer.run_in_background(lambda x: x) as server:
        assert send_ipc_message(current_process, 6) == 6
    with pytest.raises(IOError):
        send_ipc_message(current_process, 7)