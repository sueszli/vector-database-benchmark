import os
import platform
import warnings
import cv2
from torch import multiprocessing as mp

def setup_multi_processes(cfg):
    if False:
        for i in range(10):
            print('nop')
    'Setup multi-processing environment variables.'
    if platform.system() != 'Windows':
        mp_start_method = cfg.get('mp_start_method', 'fork')
        current_method = mp.get_start_method(allow_none=True)
        if current_method is not None and current_method != mp_start_method:
            warnings.warn(f'Multi-processing start method `{mp_start_method}` is different from the previous setting `{current_method}`.It will be force set to `{mp_start_method}`. You can change this behavior by changing `mp_start_method` in your config.')
        mp.set_start_method(mp_start_method, force=True)
    opencv_num_threads = cfg.get('opencv_num_threads', 0)
    cv2.setNumThreads(opencv_num_threads)
    workers_per_gpu = cfg.data.get('workers_per_gpu', 1)
    if 'train_dataloader' in cfg.data:
        workers_per_gpu = max(cfg.data.train_dataloader.get('workers_per_gpu', 1), workers_per_gpu)
    if 'OMP_NUM_THREADS' not in os.environ and workers_per_gpu > 1:
        omp_num_threads = 1
        warnings.warn(f'Setting OMP_NUM_THREADS environment variable for each process to be {omp_num_threads} in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.')
        os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)
    if 'MKL_NUM_THREADS' not in os.environ and workers_per_gpu > 1:
        mkl_num_threads = 1
        warnings.warn(f'Setting MKL_NUM_THREADS environment variable for each process to be {mkl_num_threads} in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.')
        os.environ['MKL_NUM_THREADS'] = str(mkl_num_threads)