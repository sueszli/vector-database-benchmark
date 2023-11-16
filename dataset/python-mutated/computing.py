"""Functions used in the computation of subtasks of the dummy task"""
import hashlib
import random
import time

def check_pow(proof, input_data, difficulty):
    if False:
        return 10
    '\n    :param long proof:\n    :param str input_data:\n    :param int difficulty:\n    :rtype bool:\n    '
    sha = hashlib.sha256()
    sha.update(input_data.encode())
    sha.update(('%x' % proof).encode())
    h = int(sha.hexdigest()[0:8], 16)
    return h >= difficulty

def find_pow(input_data, difficulty, result_size):
    if False:
        for i in range(10):
            print('nop')
    '\n    :param str input_data:\n    :param int difficulty:\n    :param int result_size:\n    :rtype long:\n    '
    num_bits = result_size * 4
    solution = 1 << num_bits - 1 | random.getrandbits(num_bits - 1)
    while True:
        if check_pow(solution, input_data, difficulty):
            return solution
        solution += 1

def run_dummy_task(data_file, subtask_string, difficulty, result_size):
    if False:
        return 10
    'Find a string S of result_size bytes such that the hash of the contents\n    of the data_file, subtask_data and S produce sha256 hash H such that\n    4 leftmost bytes of H is less or equal difficulty.\n    :param str data_file: file with shared task data\n    :param str subtask_string: subtask-specific part of data\n    :param int difficulty: required difficulty\n    :param int result_size: size of the solution string S\n    :rtype DummyTaskResult    '
    print('[DUMMY TASK] computation started, data_file = ', data_file, ', result_size = ', result_size, ', difficulty = 0x%08x' % difficulty)
    t0 = time.clock()
    with open(data_file, 'rU') as f:
        shared_input = f.read()
    all_input = shared_input + subtask_string
    solution = find_pow(all_input, difficulty, result_size)
    assert check_pow(solution, all_input, difficulty)
    result = '%x' % solution
    assert len(result) == result_size
    print('[DUMMY TASK] computation finished, time =', time.clock() - t0, 'sec')
    sha = hashlib.sha256()
    sha.update(all_input.encode())
    print('[DUMMY TASK] computation finished, hash of input =', sha.hexdigest())
    return result