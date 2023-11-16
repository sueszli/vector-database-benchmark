"""
@author jennakwon06
"""
import argparse
import json
import logging
import math
import os
import random
import time
from copy import copy, deepcopy
from typing import List, Tuple
import dask.array
import numpy as np
import xarray
import ray
from ray._private.test_utils import monitor_memory_usage
from ray.util.dask import ray_dask_get
"\nWe simulate a real-life usecase where we process a time-series\ndata of 1 month, using Dask/Xarray on a Ray cluster.\n\nProcessing is as follows:\n(1) Load the 1-month Xarray lazily.\nPerform decimation to reduce data size.\n\n(2) Compute FFT on the 1-year Xarray lazily.\nPerform decimation to reduce data size.\n\n(3) Segment the Xarray from (2) into 30-minute Xarrays;\nat this point, we have 4380 / 30 = 146 Xarrays.\n\n(4) Trigger save to disk for each of the 30-minute Xarrays.\nThis triggers Dask computations; there will be 146 graphs.\nSince 1460 graphs is too much to process at once,\nwe determine the batch_size based on script parameters.\n(e.g. if batch_size is 100, we'll have 15 batches).\n\n"
MINUTES_IN_A_MONTH = 500
NUM_MINS_PER_OUTPUT_FILE = 30
SAMPLING_RATE = 1000000
SECONDS_IN_A_MIN = 60
INPUT_SHAPE = (3, SAMPLING_RATE * SECONDS_IN_A_MIN)
PEAK_MEMORY_CONSUMPTION_IN_GB = 6
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

class TestSpec:

    def __init__(self, num_workers: int, worker_obj_store_size_in_gb: int, trigger_object_spill: bool, error_rate: float):
        if False:
            for i in range(10):
                print('nop')
        '\n        `batch_size` is the # of Dask graphs sent to the cluster\n        simultaneously for processing.\n\n        One element in the batch represents 1 Dask graph.\n        The Dask graph involves reading 30 arrays (one is 1.44GB)\n        and concatenating them into a Dask array.\n        Then, it does FFT computations across chunks of the Dask array.\n        It saves the FFT-ed version of the Dask array as an output file.\n\n        If `trigger_object_spill` is True, then we send work to\n        the cluster such that each worker gets the number of graphs\n        that would exceed the worker memory, triggering object spills.\n        We use the estimated peak memory consumption to determine\n        how many graphs should be sent.\n\n        If `error_rate` is True, we throw an exception at the Data\n        load layer as per error rate.\n        '
        self.error_rate = error_rate
        if trigger_object_spill:
            num_graphs_per_worker = int(math.floor(worker_obj_store_size_in_gb / PEAK_MEMORY_CONSUMPTION_IN_GB)) + 1
        else:
            num_graphs_per_worker = int(math.floor(worker_obj_store_size_in_gb / PEAK_MEMORY_CONSUMPTION_IN_GB))
        self.batch_size = num_graphs_per_worker * num_workers

    def __str__(self):
        if False:
            print('Hello World!')
        return 'Error rate = {}, Batch Size = {}'.format(self.error_rate, self.batch_size)

class LoadRoutines:

    @staticmethod
    def lazy_load_xarray_one_month(test_spec: TestSpec) -> xarray.Dataset:
        if False:
            while True:
                i = 10
        "\n        Lazily load an Xarray representing 1 month of data.\n\n        The Xarray's data variable is a dask.array that's lazily constructed.\n        Therefore, creating the Xarray object doesn't consume any memory.\n        But computing the Xarray will.\n        "
        dask_array_lists = list()
        array_dtype = np.float32
        rechunk_size = 2 << 23
        for i in range(0, MINUTES_IN_A_MONTH):
            dask_arr = dask.array.from_delayed(dask.delayed(LoadRoutines.load_array_one_minute)(test_spec), shape=INPUT_SHAPE, dtype=array_dtype)
            dask_array_lists.append(dask_arr)
        return xarray.Dataset(data_vars={'data_var': (['channel', 'time'], dask.array.rechunk(dask.array.concatenate(dask_array_lists, axis=1), chunks=(INPUT_SHAPE[0], rechunk_size)))}, coords={'channel': ('channel', np.arange(INPUT_SHAPE[0]))}, attrs={'hello': 'world'})

    @staticmethod
    def load_array_one_minute(test_spec: TestSpec) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Load an array representing 1 minute of data. Each load consumes\n        ~0.144GB of memory (3 * 200000 * 60 * 4 (bytes in a float)) = ~0.14GB\n\n        In real life, this is loaded from cloud storage or disk.\n        '
        if random.random() < test_spec.error_rate:
            raise Exception('Data error!')
        else:
            return np.random.random(INPUT_SHAPE)

class TransformRoutines:

    @staticmethod
    def fft_xarray(xr_input: xarray.Dataset, n_fft: int, hop_length: int):
        if False:
            while True:
                i = 10
        '\n        Perform FFT on an Xarray and return it as another Xarray.\n        '
        output_chunk_shape = TransformRoutines.infer_chunk_shape_after_fft(n_fft=n_fft, hop_length=hop_length, time_chunk_sizes=xr_input.chunks['time'])
        transformed_audio = dask.array.map_overlap(TransformRoutines.fft_algorithm, xr_input.data_var.data, depth={0: 0, 1: (0, n_fft - hop_length)}, boundary={0: 'none', 1: 'none'}, chunks=output_chunk_shape, dtype=np.float32, trim=True, algorithm_params={'hop_length': hop_length, 'n_fft': n_fft})
        return xarray.Dataset(data_vars={'data_var': (['channel', 'freq', 'time'], transformed_audio)}, coords={'freq': ('freq', np.arange(transformed_audio.shape[1])), 'channel': ('channel', np.arange(INPUT_SHAPE[0]))}, attrs={'hello': 'world2'})

    @staticmethod
    def decimate_xarray_after_load(xr_input: xarray.Dataset, decimate_factor: int):
        if False:
            return 10
        '\n        Downsample an Xarray.\n        '
        start_chunks = xr_input.data_var.data.chunks
        data_0 = xr_input.data_var.data[0] - xr_input.data_var.data[2]
        data_1 = xr_input.data_var.data[2]
        data_2 = xr_input.data_var.data[0]
        stacked_data = dask.array.stack([data_0, data_1, data_2], axis=0)
        stacked_chunks = stacked_data.chunks
        rechunking_to_chunks = (start_chunks[0], stacked_chunks[1])
        xr_input.data_var.data = stacked_data.rechunk(rechunking_to_chunks)
        in_chunks = xr_input.data_var.data.chunks
        out_chunks = (in_chunks[0], tuple([int(chunk / decimate_factor) for chunk in in_chunks[1]]))
        data_ds_data = xr_input.data_var.data.map_overlap(TransformRoutines.decimate_raw_data, decimate_time=decimate_factor, overlap_time=10, depth=(0, decimate_factor * 10), trim=False, dtype='float32', chunks=out_chunks)
        data_ds = copy(xr_input)
        data_ds = data_ds.isel(time=slice(0, data_ds_data.shape[1]))
        data_ds.data_var.data = data_ds_data
        return data_ds

    @staticmethod
    def decimate_raw_data(data: np.ndarray, decimate_time: int, overlap_time=0):
        if False:
            print('Hello World!')
        from scipy.signal import decimate
        data = np.nan_to_num(data)
        if decimate_time > 1:
            data = decimate(data, q=decimate_time, axis=1)
        if overlap_time > 0:
            data = data[:, overlap_time:-overlap_time]
        return data

    @staticmethod
    def fft_algorithm(data: np.ndarray, algorithm_params: dict) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Apply FFT algorithm to an input xarray.\n        '
        from scipy import signal
        hop_length = algorithm_params['hop_length']
        n_fft = algorithm_params['n_fft']
        noverlap = n_fft - hop_length
        (_, _, spectrogram) = signal.stft(data, nfft=n_fft, nperseg=n_fft, noverlap=noverlap, return_onesided=False, boundary=None)
        spectrogram = np.abs(spectrogram)
        spectrogram = 10 * np.log10(spectrogram ** 2)
        return spectrogram

    @staticmethod
    def infer_chunk_shape_after_fft(n_fft: int, hop_length: int, time_chunk_sizes: List) -> tuple:
        if False:
            return 10
        '\n        Infer the chunk shapes after applying FFT transformation.\n        Infer is necessary for lazy transformations in Dask when\n        transformations do not preserve chunk shape.\n        '
        output_time_chunk_sizes = list()
        for time_chunk_size in time_chunk_sizes:
            output_time_chunk_sizes.append(math.ceil(time_chunk_size / hop_length))
        num_freq = int(n_fft / 2 + 1)
        return ((INPUT_SHAPE[0],), (num_freq,), tuple(output_time_chunk_sizes))

    @staticmethod
    def fix_last_chunk_error(xr_input: xarray.Dataset, n_overlap):
        if False:
            for i in range(10):
                print('nop')
        time_chunks = list(xr_input.chunks['time'])
        if time_chunks[-1] < n_overlap:
            current_len = len(xr_input.time)
            xr_input = xr_input.isel(time=slice(0, current_len - time_chunks[-1]))
        if time_chunks[0] < n_overlap:
            current_len = len(xr_input.time)
            xr_input = xr_input.isel(time=slice(time_chunks[0], current_len))
        return xr_input

class SaveRoutines:

    @staticmethod
    def save_xarray(xarray_dataset, filename, dirpath):
        if False:
            while True:
                i = 10
        '\n        Save Xarray in zarr format.\n        '
        filepath = os.path.join(dirpath, filename)
        if os.path.exists(filepath):
            return 'already_exists'
        try:
            xarray_dataset.to_zarr(filepath)
        except Exception as e:
            return 'failure, exception = {}'.format(e)
        return 'success'

    @staticmethod
    def save_all_xarrays(xarray_filename_pairs: List[Tuple], ray_scheduler, dirpath: str, batch_size: int):
        if False:
            while True:
                i = 10

        def chunks(lst, n):
            if False:
                i = 10
                return i + 15
            'Yield successive n-sized chunks from lst.'
            for i in range(0, len(lst), n):
                yield lst[i:i + n]
        for (batch_idx, batch) in enumerate(chunks(xarray_filename_pairs, batch_size)):
            delayed_tasks = list()
            for xarray_filename_pair in batch:
                delayed_tasks.append(dask.delayed(SaveRoutines.save_xarray)(xarray_dataset=xarray_filename_pair[0], filename=xarray_filename_pair[1], dirpath=dirpath))
            logging.info('[Batch Index {}] Batch size {}: Sending work to Ray Cluster.'.format(batch_idx, batch_size))
            res = []
            try:
                res = dask.compute(delayed_tasks, scheduler=ray_scheduler)
            except Exception:
                logging.warning('[Batch Index {}] Exception while computing batch!'.format(batch_idx))
            finally:
                logging.info('[Batch Index {}], Result = {}'.format(batch_idx, res))

def lazy_create_xarray_filename_pairs(test_spec: TestSpec) -> List[Tuple[xarray.Dataset, str]]:
    if False:
        i = 10
        return i + 15
    n_fft = 4096
    hop_length = int(SAMPLING_RATE / 100)
    decimate_factor = 100
    logging.info('Creating 1 month lazy Xarray with decimation and FFT')
    xr1 = LoadRoutines.lazy_load_xarray_one_month(test_spec)
    xr2 = TransformRoutines.decimate_xarray_after_load(xr_input=xr1, decimate_factor=decimate_factor)
    xr3 = TransformRoutines.fix_last_chunk_error(xr2, n_overlap=n_fft - hop_length)
    xr4 = TransformRoutines.fft_xarray(xr_input=xr3, n_fft=n_fft, hop_length=hop_length)
    num_segments = int(MINUTES_IN_A_MONTH / NUM_MINS_PER_OUTPUT_FILE)
    start_time = 0
    xarray_filename_pairs: List[Tuple[xarray.Dataset, str]] = list()
    timestamp = int(time.time())
    for step in range(num_segments):
        segment_start = start_time + NUM_MINS_PER_OUTPUT_FILE * step
        segment_start_index = int(SECONDS_IN_A_MIN * NUM_MINS_PER_OUTPUT_FILE * step * (SAMPLING_RATE / decimate_factor) / hop_length)
        segment_end = segment_start + NUM_MINS_PER_OUTPUT_FILE
        segment_len_sec = (segment_end - segment_start) * SECONDS_IN_A_MIN
        segment_end_index = int(segment_start_index + segment_len_sec * SAMPLING_RATE / hop_length)
        xr_segment = deepcopy(xr4.isel(time=slice(segment_start_index, segment_end_index)))
        xarray_filename_pairs.append((xr_segment, 'xarray_step_{}_{}.zarr'.format(step, timestamp)))
    return xarray_filename_pairs

def parse_script_args():
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--worker_obj_store_size_in_gb', type=int)
    parser.add_argument('--error_rate', type=float, default=0)
    parser.add_argument('--data_save_path', type=str)
    parser.add_argument('--trigger_object_spill', dest='trigger_object_spill', action='store_true')
    parser.set_defaults(trigger_object_spill=False)
    return parser.parse_known_args()

def main():
    if False:
        i = 10
        return i + 15
    (args, unknown) = parse_script_args()
    logging.info('Received arguments: {}'.format(args))
    test_spec = TestSpec(num_workers=args.num_workers, worker_obj_store_size_in_gb=args.worker_obj_store_size_in_gb, error_rate=args.error_rate, trigger_object_spill=args.trigger_object_spill)
    logging.info('Created test spec: {}'.format(test_spec))
    data_save_path = args.data_save_path
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path, mode=511, exist_ok=True)
    xarray_filename_pairs = lazy_create_xarray_filename_pairs(test_spec)
    ray.init(address='auto')
    monitor_actor = monitor_memory_usage()
    logging.info('Saving {} xarrays..'.format(len(xarray_filename_pairs)))
    SaveRoutines.save_all_xarrays(xarray_filename_pairs=xarray_filename_pairs, dirpath=data_save_path, batch_size=test_spec.batch_size, ray_scheduler=ray_dask_get)
    ray.get(monitor_actor.stop_run.remote())
    (used_gb, usage) = ray.get(monitor_actor.get_peak_memory_info.remote())
    print(f'Peak memory usage: {round(used_gb, 2)}GB')
    print(f'Peak memory usage per processes:\n {usage}')
    try:
        print(ray._private.internal_api.memory_summary(stats_only=True))
    except Exception as e:
        print(f'Warning: query memory summary failed: {e}')
    with open(os.environ['TEST_OUTPUT_JSON'], 'w') as f:
        f.write(json.dumps({'success': 1, '_peak_memory': round(used_gb, 2), '_peak_process_memory': usage}))
if __name__ == '__main__':
    main()