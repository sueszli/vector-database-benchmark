import os
import time
from multiprocessing import Process
from random import random
import pytest
from arctic import Arctic, VERSION_STORE
from arctic.exceptions import LibraryNotFoundException
from arctic.hooks import register_get_auth_hook
from arctic.store.version_store import VersionStore
MY_ARCTIC = None
AUTH_COUNT = 0

def f(library_name, total_writes, do_reset):
    if False:
        i = 10
        return i + 15
    my_pid = os.getpid()
    data = [str(my_pid)] * 100
    if do_reset:
        global AUTH_COUNT
        AUTH_COUNT = 0
        MY_ARCTIC.reset()
        assert AUTH_COUNT > 0
    while True:
        try:
            vstore = MY_ARCTIC[library_name]
            break
        except LibraryNotFoundException:
            pass
        time.sleep(random() * 0.2)
    for i in range(total_writes):
        if i % 20 == 0:
            time.sleep(random())
        key = '{}_{}'.format(my_pid, i)
        vstore.write(key, data + [key])
    for i in range(total_writes):
        key = '{}_{}'.format(my_pid, i)
        assert vstore.read(key).data == data + [key]

def my_auth_hook(host, app_name, database_name):
    if False:
        while True:
            i = 10
    global AUTH_COUNT
    AUTH_COUNT += 1

@pytest.mark.timeout(600)
def test_multiprocessing_safety(mongo_host, library_name):
    if False:
        i = 10
        return i + 15
    total_processes = 64
    total_writes_per_child = 100
    register_get_auth_hook(my_auth_hook)
    global MY_ARCTIC
    MY_ARCTIC = Arctic(mongo_host=mongo_host)
    MY_ARCTIC.initialize_library(library_name, VERSION_STORE)
    assert isinstance(MY_ARCTIC.get_library(library_name), VersionStore)
    processes = [Process(target=f, args=(library_name, total_writes_per_child, True)) for _ in range(total_processes)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    for p in processes:
        assert p.exitcode == 0
    assert isinstance(MY_ARCTIC.get_library(library_name), VersionStore)

@pytest.mark.timeout(600)
def test_multiprocessing_safety_parent_children_race(mongo_host, library_name):
    if False:
        for i in range(10):
            print('nop')
    total_iterations = 12
    total_processes = 6
    total_writes_per_child = 20
    global MY_ARCTIC
    for i in range(total_iterations):
        processes = list()
        MY_ARCTIC = Arctic(mongo_host=mongo_host)
        for j in range(total_processes):
            p = Process(target=f, args=(library_name, total_writes_per_child, False))
            p.start()
            processes.append(p)
        MY_ARCTIC.initialize_library(library_name, VERSION_STORE)
        for p in processes:
            p.join()
        for p in processes:
            assert p.exitcode == 0
        MY_ARCTIC.reset()
    assert isinstance(MY_ARCTIC.get_library(library_name), VersionStore)