import time
import sys
import numpy as np
import cloudpickle
import tensorflow as tf
import os
from bigdl.nano.tf.keras import InferenceOptimizer

def throughput_calculate_helper(iterrun, func, model, input_sample):
    if False:
        for i in range(10):
            print('nop')
    '\n    A simple helper to calculate average latency\n    '
    time_list = []
    for i in range(iterrun):
        st = time.perf_counter()
        func(model, input_sample)
        end = time.perf_counter()
        time_list.append(end - st)
    time_list.sort()
    time_list = time_list[int(0.1 * iterrun):int(0.9 * iterrun)]
    return np.mean(time_list) * 1000
if __name__ == '__main__':
    param_file = sys.argv[1]
    thread_num = sys.argv[2]
    if thread_num != 'None':
        thread_num = int(thread_num)
        tf.config.threading.set_inter_op_parallelism_threads(0)
        tf.config.threading.set_intra_op_parallelism_threads(thread_num)
    params = cloudpickle.load(open(param_file, 'rb'))
    if params['method'] != 'original':
        model_dir = os.path.dirname(param_file)
        qmodel = InferenceOptimizer.load(model_dir, params['model'])
        params['model'] = qmodel
    del params['method']
    latency = throughput_calculate_helper(**params)
    print(latency)