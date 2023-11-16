import os
import psutil
import weakref

def capture_processes(pool):
    if False:
        while True:
            i = 10
    'Need to be called to register the processes created by the pool. It is later used\n    by the teardown_function to check if no process stayed alive after the test finished.\n    '
    global pool_processes
    global pool_threads
    global pools
    if pool is not None:
        pools.append(weakref.ref(pool))
        pool_processes.extend(pool.pids())
        proc_pool = pool.pool
        pool_threads.append(proc_pool._observer.thread)
    check_shm_for_dali('All shm chunks should be closed after initial pool setup, found {}')

def setup_function():
    if False:
        for i in range(10):
            print('nop')
    'Prepare for the check if all started processes are no longer children of current process\n    '
    global pool_processes
    global pool_threads
    global pools
    pool_processes = []
    pool_threads = []
    pools = []

def teardown_function():
    if False:
        for i in range(10):
            print('nop')
    'Check if there are no children processes started by the test after it ended.\n\n    Be sure to call `capture_processes` in the test.\n    '
    global pool_processes
    global pool_threads
    global pools
    assert len(pool_processes), 'No processes where tracked - did the test call capture_processes?'
    pools_not_collected = [pool_ref() is not None for pool_ref in pools]
    current_process = psutil.Process()
    children_pids = [process.pid for process in current_process.children()]
    left = set(pool_processes).intersection(children_pids)
    assert len(left) == 0, f'Pipeline-started processes left after test is finished, pids alive: {left},\npids started during tests: {pool_processes}.\nPools not collected: {sum(pools_not_collected)}'
    alive_threads = [thread.is_alive() for thread in pool_threads]
    assert sum(alive_threads) == 0, 'Some pool related threads are left after the test finished. Started in test suite: {}, still active: {}. Active threads map in the order of creation {}'.format(len(pool_threads), sum(alive_threads), alive_threads)

def check_shm_for_dali(msg):
    if False:
        return 10
    shm_paths = ['/dev/shm/', '/run/shm/']
    for shm_path in shm_paths:
        if os.path.isdir(shm_path):
            shm_handles = os.listdir(shm_path)
            for handle in shm_handles:
                assert 'nvidia_dali_' not in handle, msg.format(shm_path + handle)

def setup_module():
    if False:
        i = 10
        return i + 15
    check_shm_for_dali('Expected clear shared mem environment before starting tests, found old DALI file handle: {}')

def teardown_module():
    if False:
        i = 10
        return i + 15
    check_shm_for_dali('Test left opened shared memory file handle: {}')