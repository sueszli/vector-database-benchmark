import itertools
import logging
import os
import queue
import sys
import threading
import time
import warnings
import numpy as np
import paddle
from paddle import profiler
from paddle.base.framework import _current_expected_place, _set_expected_place
from paddle.profiler.timer import benchmark
from paddle.profiler.utils import in_profiler_mode
from ...framework import core, in_dynamic_mode
from ..multiprocess_utils import MP_STATUS_CHECK_INTERVAL, CleanupFuncRegistrar, _set_SIGCHLD_handler
from .batch_sampler import _InfiniteIterableSampler
from .collate import default_collate_fn, default_convert_fn
from .flat import _flatten_batch, _restore_batch
from .worker import _DatasetKind, _IterableDatasetStopIteration, _ResumeIteration, _worker_loop, _WorkerException
_loader = None

def _clear_loader():
    if False:
        print('Hello World!')
    global _loader
    if _loader is not None:
        try:
            _loader.__del__()
            del _loader
        except:
            pass
CleanupFuncRegistrar.register(_clear_loader)

class _DataLoaderIterBase:
    """
    Iterator implement of DataLoader, will load and feed mini-batch
    data by setting in given dataloader.

    Args:
        loader(instance of DataLoader): instance of `paddle.io.DataLoader`
    """

    def __init__(self, loader):
        if False:
            return 10
        self._dataset = loader.dataset
        self._feed_list = loader.feed_list or []
        self._places = loader.places
        self._return_list = loader.return_list
        self._batch_sampler = loader.batch_sampler
        self._drop_last = loader.drop_last
        self._auto_collate_batch = loader.auto_collate_batch
        self._num_workers = loader.num_workers
        self._use_buffer_reader = loader.use_buffer_reader
        self._prefetch_factor = loader.prefetch_factor
        self._use_shared_memory = loader.use_shared_memory
        self._timeout = loader.timeout if loader.timeout > 0 else MP_STATUS_CHECK_INTERVAL
        self._worker_init_fn = loader.worker_init_fn
        self._dataset_kind = loader.dataset_kind
        self._pin_memory = loader.pin_memory
        self._sampler_iter = iter(self._index_sampler)
        if self._auto_collate_batch:
            self._collate_fn = loader.collate_fn or default_collate_fn
        else:
            self._collate_fn = loader.collate_fn or default_convert_fn
        self._blocking_queue = None
        self._thread = None
        self._thread_done_event = threading.Event()

    @property
    def _index_sampler(self):
        if False:
            i = 10
            return i + 15
        if self._auto_collate_batch:
            return self._batch_sampler
        elif self._dataset_kind == _DatasetKind.MAP:
            return list(range(len(self._dataset)))
        else:
            return _InfiniteIterableSampler(self._dataset, 1)

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self._batch_sampler)

    def _exit_thread_expectedly(self):
        if False:
            for i in range(10):
                print('nop')
        self._thread_done_event.set()
        if self._blocking_queue:
            self._blocking_queue.close()

    def _exit_thread_unexpectedly(self):
        if False:
            print('Hello World!')
        self._thread_done_event.set()
        if self._blocking_queue:
            self._blocking_queue.kill()

class _DataLoaderIterSingleProcess(_DataLoaderIterBase):
    """
    Single process implement of DataLoaderIter, loading data from
    loader.data in main process
    """

    def __init__(self, loader):
        if False:
            print('Hello World!')
        super().__init__(loader)
        self._dataset_fetcher = _DatasetKind.create_fetcher(self._dataset_kind, self._dataset, self._auto_collate_batch, self._collate_fn, self._drop_last)
        self._structure_infos = []
        self._blocking_queue_capacity = self._prefetch_factor * len(self._places)
        self._init_thread()
        self._shutdown = False
        global _loader
        _loader = self

    def _init_thread(self):
        if False:
            for i in range(10):
                print('nop')
        self._var_names = [v.name for v in self._feed_list]
        self._shapes = [v.shape for v in self._feed_list]
        self._dtypes = [v.dtype for v in self._feed_list]
        self._need_check_feed = [v.desc.need_check_feed() for v in self._feed_list]
        self._blocking_queue = core.init_lod_tensor_blocking_queue(core.Variable(), self._blocking_queue_capacity, len(self._places) > 1)
        self._reader = core.create_py_reader(self._blocking_queue, self._var_names, self._shapes, self._dtypes, self._need_check_feed, self._places, self._use_buffer_reader, True, self._pin_memory)
        self._thread = threading.Thread(target=self._thread_loop, args=(_current_expected_place(),))
        self._thread.daemon = True
        self._thread.start()

    def _thread_loop(self, legacy_expected_place):
        if False:
            return 10
        core.set_current_thread_name('Dataloader_' + str(id(self)))
        _set_expected_place(legacy_expected_place)
        while not self._thread_done_event.is_set():
            try:
                indices = next(self._sampler_iter)
                batch = self._dataset_fetcher.fetch(indices, self._thread_done_event)
            except StopIteration:
                self._exit_thread_expectedly()
                return
            if batch is None or self._thread_done_event.is_set():
                break
            (batch, structure) = _flatten_batch(batch)
            self._structure_infos.append(structure)
            if self._thread_done_event.is_set():
                break
            try:
                array = core.LoDTensorArray()
                for slot in batch:
                    if isinstance(slot, (paddle.Tensor, core.eager.Tensor)):
                        slot = slot.value().get_tensor()
                    elif not isinstance(slot, core.LoDTensor):
                        tmp = core.LoDTensor()
                        tmp.set(slot, core.CPUPlace())
                        slot = tmp
                    array.append(slot)
                if self._thread_done_event.is_set():
                    break
                try:
                    self._blocking_queue.push(array)
                except:
                    self._exit_thread_expectedly()
            except Exception as e:
                self._exit_thread_unexpectedly()
                raise e
        self._exit_thread_expectedly()

    def __next__(self):
        if False:
            while True:
                i = 10
        if in_profiler_mode():
            trace_event = profiler.RecordEvent(name='_DataLoaderIterSingleProcess', event_type=profiler.TracerEventType.Dataloader)
            trace_event.begin()
        try:
            benchmark().check_if_need_record(self)
            benchmark().before_reader()
            if in_dynamic_mode():
                data = core.eager.read_next_tensor_list(self._reader.read_next_list()[0])
                data = _restore_batch(data, self._structure_infos.pop(0))
            elif self._return_list:
                data = self._reader.read_next_list()
                for i in range(len(data)):
                    data[i] = data[i]._move_to_list()
                structs = [self._structure_infos.pop(0) for _ in range(len(self._places))]
                data = [_restore_batch(d, s) for (d, s) in zip(data, structs)]
                if len(self._places) == 1:
                    data = data[0]
            else:
                data = self._reader.read_next()
            benchmark().after_reader()
            return data
        except StopIteration:
            self._reader.shutdown()
            self._try_shutdown_all()
            raise
        finally:
            if in_profiler_mode():
                trace_event.end()

    def _shutdown_thread(self):
        if False:
            i = 10
            return i + 15
        if self._thread:
            self._thread_done_event.set()
            for _ in range(3):
                if self._thread.is_alive():
                    time.sleep(1)
                else:
                    break
            else:
                if self._thread is not threading.current_thread():
                    self._thread.join()
            self._thread = None

    def _try_shutdown_all(self):
        if False:
            print('Hello World!')
        if not self._shutdown:
            try:
                if self._blocking_queue:
                    self._blocking_queue.close()
                    self._blocking_queue = None
                self._shutdown_thread()
            finally:
                self._shutdown = True

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        self._try_shutdown_all()

class _DataLoaderIterMultiProcess(_DataLoaderIterBase):

    def __init__(self, loader):
        if False:
            return 10
        super().__init__(loader)
        self._persistent_workers = loader._persistent_workers
        self._resume_worker_cnt = 0
        assert self._num_workers > 0, f'Multi-process DataLoader invalid num_workers({self._num_workers})'
        self._data_queue = None
        self._send_idx = 0
        self._rcvd_idx = 0
        self._batches_outstanding = 0
        self._task_infos = {}
        self._structure_infos = []
        self._outstanding_capacity = self._prefetch_factor * max(self._num_workers, len(self._places))
        self._thread_lock = threading.Lock()
        self._base_seed = np.random.randint(low=0, high=sys.maxsize)
        if os.environ.get('FLAGS_use_shm_cache', False) in [1, '1', True, 'True', 'true']:
            try:
                self._worker_shm_buffer_size = (2 + 1) * len(self._dataset[0])
            except:
                self._worker_shm_buffer_size = 0
                warnings.warn('Setting the shm cache buffer size to 0, equivalent to not using the shm cache policy.')
        else:
            self._worker_shm_buffer_size = 0
        self._main_thread_shm_buffer_size = self._worker_shm_buffer_size * 2 * self._num_workers
        self._init_workers()
        for _ in range(self._outstanding_capacity):
            self._try_put_indices()
        self._init_thread()
        self._shutdown = False

    def _init_workers(self):
        if False:
            for i in range(10):
                print('nop')
        from paddle.incubate import multiprocessing
        self._workers = []
        self._worker_status = []
        self._indices_queues = []
        self._workers_idx_cycle = itertools.cycle(range(self._num_workers))
        self._data_queue = multiprocessing.Queue()
        self._workers_done_event = multiprocessing.Event()
        self._thread_done_event = threading.Event()
        for i in range(self._num_workers):
            indices_queue = multiprocessing.Queue()
            indices_queue.cancel_join_thread()
            self._indices_queues.append(indices_queue)
            worker = multiprocessing.Process(target=_worker_loop, args=(self._dataset, self._dataset_kind, indices_queue, self._data_queue, self._workers_done_event, self._auto_collate_batch, self._collate_fn, self._drop_last, self._worker_init_fn, i, self._num_workers, self._use_shared_memory, self._base_seed, self._worker_shm_buffer_size))
            worker.daemon = True
            worker.start()
            self._workers.append(worker)
            self._worker_status.append(True)
        core._set_process_pids(id(self), tuple((w.pid for w in self._workers)))
        _set_SIGCHLD_handler()

    def _clear_and_remove_data_queue(self):
        if False:
            i = 10
            return i + 15
        if self._data_queue is not None:
            while True:
                try:
                    self._data_queue.get_nowait()
                except:
                    self._data_queue.cancel_join_thread()
                    self._data_queue.close()
                    break

    def _init_thread(self):
        if False:
            i = 10
            return i + 15
        self._var_names = [v.name for v in self._feed_list]
        self._shapes = [v.shape for v in self._feed_list]
        self._dtypes = [v.dtype for v in self._feed_list]
        self._need_check_feed = [v.desc.need_check_feed() for v in self._feed_list]
        self._blocking_queue = core.init_lod_tensor_blocking_queue(core.Variable(), self._outstanding_capacity, len(self._places) > 1)
        core._set_max_memory_map_allocation_pool_size(self._main_thread_shm_buffer_size)
        self._reader = core.create_py_reader(self._blocking_queue, self._var_names, self._shapes, self._dtypes, self._need_check_feed, self._places, self._use_buffer_reader, True, self._pin_memory)
        self._thread_done_event = threading.Event()
        self._thread = threading.Thread(target=self._thread_loop, args=(_current_expected_place(),))
        self._thread.daemon = True
        self._thread.start()

    def _reset(self):
        if False:
            while True:
                i = 10
        with self._thread_lock:
            self._resume_worker_cnt = self._num_workers
            for worker_id in range(self._num_workers):
                self._indices_queues[worker_id].put(_ResumeIteration())
                self._batches_outstanding += 1
        while self._resume_worker_cnt > 0:
            time.sleep(0.5)
        while self._blocking_queue.size() >= len(self._places):
            if in_dynamic_mode():
                data = core.eager.read_next_tensor_list(self._reader.read_next_list()[0])
            elif self._return_list:
                self._reader.read_next_list()
            else:
                data = self._reader.read_next()
        self._send_idx = 0
        self._rcvd_idx = 0
        self._batches_outstanding = 0
        self._task_infos = {}
        self._structure_infos = []
        self._worker_status = [True] * self._num_workers
        self._sampler_iter = iter(self._index_sampler)
        for _ in range(self._outstanding_capacity):
            self._try_put_indices()

    def _shutdown_worker(self, worker_id, shutdown=False):
        if False:
            for i in range(10):
                print('nop')
        if self._worker_status[worker_id] or (self._persistent_workers and shutdown):
            self._indices_queues[worker_id].put(None)
            self._worker_status[worker_id] = False

    def _try_shutdown_all(self, timeout=None):
        if False:
            print('Hello World!')
        if not self._shutdown:
            try:
                self._exit_thread_expectedly()
                self._clear_and_remove_data_queue()
                self._workers_done_event.set()
                for i in range(self._num_workers):
                    self._shutdown_worker(i, shutdown=True)
                if not self._shutdown:
                    for w in self._workers:
                        w.join(timeout)
                    for q in self._indices_queues:
                        q.cancel_join_thread()
                        q.close()
            finally:
                core._erase_process_pids(id(self))
                self._shutdown = True

    def _thread_loop(self, legacy_expected_place):
        if False:
            print('Hello World!')
        core.set_current_thread_name('Dataloader_' + str(id(self)))
        _set_expected_place(legacy_expected_place)
        while not self._thread_done_event.is_set():
            batch = self._get_data()
            if not self._thread_done_event.is_set():
                if batch is None:
                    self._exit_thread_expectedly()
                else:
                    if isinstance(batch, _ResumeIteration):
                        assert self._resume_worker_cnt > 0
                        self._resume_worker_cnt -= 1
                        continue
                    try:
                        array = core.LoDTensorArray()
                        if self._use_shared_memory:
                            for tensor in batch:
                                array.append(tensor)
                        else:
                            for slot in batch:
                                if isinstance(slot, (paddle.Tensor, core.eager.Tensor)):
                                    slot = slot.value().get_tensor()
                                elif not isinstance(slot, core.LoDTensor):
                                    tmp = core.LoDTensor()
                                    tmp.set(slot, core.CPUPlace())
                                    slot = tmp
                                array.append(slot)
                        if not self._blocking_queue.push(array):
                            self._blocking_queue.close()
                    except Exception as e:
                        self._exit_thread_unexpectedly()
                        raise e
                    finally:
                        self._rcvd_idx += 1

    def _get_data(self):
        if False:
            i = 10
            return i + 15
        while not self._thread_done_event.is_set():
            if self._dataset_kind == _DatasetKind.ITER:
                while self._rcvd_idx < self._send_idx:
                    info = self._task_infos[self._rcvd_idx]
                    if len(info) == 3 or self._worker_status[info[0]]:
                        break
                    del self._task_infos[self._rcvd_idx]
                    self._rcvd_idx += 1
                    self._batches_outstanding -= 1
                else:
                    if not self._persistent_workers:
                        if self._batches_outstanding < len(self._places):
                            return None
            if self._rcvd_idx in self._task_infos and len(self._task_infos[self._rcvd_idx]) == 3:
                info = self._task_infos.pop(self._rcvd_idx)
                self._structure_infos.append(info[2])
                return info[1]
            try:
                data = self._data_queue.get(timeout=self._timeout)
            except Exception as e:
                if self._thread_done_event.is_set():
                    continue
                failed_workers = []
                for (i, w) in enumerate(self._workers):
                    if self._worker_status[i] and (not w.is_alive()):
                        failed_workers.append(w)
                        self._shutdown_worker(i)
                if len(failed_workers) > 0:
                    self._exit_thread_unexpectedly()
                    pids = ', '.join((str(w.pid) for w in failed_workers))
                    raise RuntimeError(f'DataLoader {len(failed_workers)} workers exit unexpectedly, pids: {pids}')
                if isinstance(e, (IOError, queue.Empty)):
                    continue
                self._exit_thread_unexpectedly()
                logging.error(f"DataLoader reader thread failed({e}) to read data from workers' result queue.")
                raise e
            else:
                if self._dataset_kind == _DatasetKind.ITER and isinstance(data, _IterableDatasetStopIteration):
                    if self._persistent_workers:
                        self._worker_status[data.worker_id] = False
                    else:
                        self._shutdown_worker(data.worker_id)
                        self._batches_outstanding -= 1
                    self._try_put_indices()
                    continue
                (idx, batch, structure) = data
                if isinstance(idx, _ResumeIteration) and batch is None and (structure is None):
                    return idx
                if isinstance(batch, _WorkerException):
                    self._exit_thread_unexpectedly()
                    batch.reraise()
                if idx == self._rcvd_idx:
                    if idx in self._task_infos:
                        del self._task_infos[idx]
                    self._structure_infos.append(structure)
                    return batch
                else:
                    self._task_infos[idx] += (batch, structure)
                    continue

    def _try_put_indices(self):
        if False:
            print('Hello World!')
        assert self._batches_outstanding <= self._outstanding_capacity, 'too many indices have been put to queue'
        with self._thread_lock:
            try:
                indices = next(self._sampler_iter)
            except StopIteration:
                return
            for i in range(self._num_workers):
                worker_idx = next(self._workers_idx_cycle)
                if self._worker_status[worker_idx]:
                    break
            else:
                return
            self._indices_queues[worker_idx].put((self._send_idx, indices))
            self._task_infos[self._send_idx] = (worker_idx,)
            self._batches_outstanding += 1
            self._send_idx += 1

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        self._try_shutdown_all()

    def _shutdown_on_exit(self):
        if False:
            print('Hello World!')
        self._try_shutdown_all(1)

    def __next__(self):
        if False:
            while True:
                i = 10
        if in_profiler_mode():
            trace_event = profiler.RecordEvent(name='_DataLoaderIterMultiProcess', event_type=profiler.TracerEventType.Dataloader)
            trace_event.begin()
        try:
            benchmark().check_if_need_record(self)
            benchmark().before_reader()
            if self._batches_outstanding < len(self._places):
                if self._persistent_workers:
                    raise StopIteration
                else:
                    self._thread_done_event.set()
                    self._blocking_queue.close()
            if in_dynamic_mode():
                data = core.eager.read_next_tensor_list(self._reader.read_next_list()[0])
                data = _restore_batch(data, self._structure_infos.pop(0))
            elif self._return_list:
                data = self._reader.read_next_list()
                for i in range(len(data)):
                    data[i] = data[i]._move_to_list()
                structs = [self._structure_infos.pop(0) for _ in range(len(self._places))]
                data = [_restore_batch(d, s) for (d, s) in zip(data, structs)]
                if len(self._places) == 1:
                    data = data[0]
            else:
                data = self._reader.read_next()
            self._on_output_batch()
            benchmark().after_reader()
            return data
        except StopIteration:
            if not self._persistent_workers:
                self._reader.shutdown()
                self._try_shutdown_all()
            raise
        finally:
            if in_profiler_mode():
                trace_event.end()

    def _on_output_batch(self):
        if False:
            for i in range(10):
                print('nop')
        for _ in range(len(self._places)):
            self._batches_outstanding -= 1
            self._try_put_indices()