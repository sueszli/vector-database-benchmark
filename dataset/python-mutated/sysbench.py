"""
The 'sysbench' module is used to analyze the
performance of the minions, right from the master!
It measures various system parameters such as
CPU, Memory, File I/O, Threads and Mutex.
"""
import re
import salt.utils.path

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    loads the module, if only sysbench is installed\n    '
    if salt.utils.path.which('sysbench'):
        return 'sysbench'
    return (False, 'The sysbench execution module failed to load: the sysbench binary is not in the path.')

def _parser(result):
    if False:
        for i in range(10):
            print('nop')
    '\n    parses the output into a dictionary\n    '
    _total_time = re.compile('total time:\\s*(\\d*.\\d*s)')
    _total_execution = re.compile('event execution:\\s*(\\d*.\\d*s?)')
    _min_response_time = re.compile('min:\\s*(\\d*.\\d*ms)')
    _max_response_time = re.compile('max:\\s*(\\d*.\\d*ms)')
    _avg_response_time = re.compile('avg:\\s*(\\d*.\\d*ms)')
    _per_response_time = re.compile('95 percentile:\\s*(\\d*.\\d*ms)')
    total_time = re.search(_total_time, result).group(1)
    total_execution = re.search(_total_execution, result).group(1)
    min_response_time = re.search(_min_response_time, result).group(1)
    max_response_time = re.search(_max_response_time, result).group(1)
    avg_response_time = re.search(_avg_response_time, result).group(1)
    per_response_time = re.search(_per_response_time, result)
    if per_response_time is not None:
        per_response_time = per_response_time.group(1)
    return {'total time': total_time, 'total execution time': total_execution, 'minimum response time': min_response_time, 'maximum response time': max_response_time, 'average response time': avg_response_time, '95 percentile': per_response_time}

def cpu():
    if False:
        i = 10
        return i + 15
    "\n    Tests for the CPU performance of minions.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' sysbench.cpu\n    "
    max_primes = [500, 1000, 2500, 5000]
    test_command = 'sysbench --test=cpu --cpu-max-prime={0} run'
    result = None
    ret_val = {}
    for primes in max_primes:
        key = 'Prime numbers limit: {}'.format(primes)
        run_command = test_command.format(primes)
        result = __salt__['cmd.run'](run_command)
        ret_val[key] = _parser(result)
    return ret_val

def threads():
    if False:
        while True:
            i = 10
    "\n    This tests the performance of the processor's scheduler\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' sysbench.threads\n    "
    thread_yields = [100, 200, 500, 1000]
    thread_locks = [2, 4, 8, 16]
    test_command = 'sysbench --num-threads=64 --test=threads '
    test_command += '--thread-yields={0} --thread-locks={1} run '
    result = None
    ret_val = {}
    for (yields, locks) in zip(thread_yields, thread_locks):
        key = 'Yields: {} Locks: {}'.format(yields, locks)
        run_command = test_command.format(yields, locks)
        result = __salt__['cmd.run'](run_command)
        ret_val[key] = _parser(result)
    return ret_val

def mutex():
    if False:
        while True:
            i = 10
    "\n    Tests the implementation of mutex\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' sysbench.mutex\n    "
    mutex_num = [50, 50, 50, 500, 500, 500, 1000, 1000, 1000]
    locks = [10000, 25000, 50000, 10000, 25000, 50000, 10000, 25000, 50000]
    mutex_locks = []
    mutex_locks.extend(locks)
    mutex_loops = [2500, 5000, 10000, 10000, 2500, 5000, 5000, 10000, 2500]
    test_command = 'sysbench --num-threads=250 --test=mutex '
    test_command += '--mutex-num={0} --mutex-locks={1} --mutex-loops={2} run '
    result = None
    ret_val = {}
    for (num, locks, loops) in zip(mutex_num, mutex_locks, mutex_loops):
        key = 'Mutex: {} Locks: {} Loops: {}'.format(num, locks, loops)
        run_command = test_command.format(num, locks, loops)
        result = __salt__['cmd.run'](run_command)
        ret_val[key] = _parser(result)
    return ret_val

def memory():
    if False:
        while True:
            i = 10
    "\n    This tests the memory for read and write operations.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' sysbench.memory\n    "
    memory_oper = ['read', 'write']
    memory_scope = ['local', 'global']
    test_command = 'sysbench --num-threads=64 --test=memory '
    test_command += '--memory-oper={0} --memory-scope={1} '
    test_command += '--memory-block-size=1K --memory-total-size=32G run '
    result = None
    ret_val = {}
    for oper in memory_oper:
        for scope in memory_scope:
            key = 'Operation: {} Scope: {}'.format(oper, scope)
            run_command = test_command.format(oper, scope)
            result = __salt__['cmd.run'](run_command)
            ret_val[key] = _parser(result)
    return ret_val

def fileio():
    if False:
        while True:
            i = 10
    "\n    This tests for the file read and write operations\n    Various modes of operations are\n\n    * sequential write\n    * sequential rewrite\n    * sequential read\n    * random read\n    * random write\n    * random read and write\n\n    The test works with 32 files with each file being 1Gb in size\n    The test consumes a lot of time. Be patient!\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' sysbench.fileio\n    "
    test_modes = ['seqwr', 'seqrewr', 'seqrd', 'rndrd', 'rndwr', 'rndrw']
    test_command = 'sysbench --num-threads=16 --test=fileio '
    test_command += '--file-num=32 --file-total-size=1G --file-test-mode={0} '
    result = None
    ret_val = {}
    for mode in test_modes:
        key = 'Mode: {}'.format(mode)
        run_command = (test_command + 'prepare').format(mode)
        __salt__['cmd.run'](run_command)
        run_command = (test_command + 'run').format(mode)
        result = __salt__['cmd.run'](run_command)
        ret_val[key] = _parser(result)
        run_command = (test_command + 'cleanup').format(mode)
        __salt__['cmd.run'](run_command)
    return ret_val

def ping():
    if False:
        i = 10
        return i + 15
    return True