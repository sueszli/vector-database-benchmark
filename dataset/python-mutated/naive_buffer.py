import os
import copy
from typing import Union, Any, Optional, List
import numpy as np
import math
import hickle
from easydict import EasyDict
from ding.worker.replay_buffer import IBuffer
from ding.utils import LockContext, LockContextType, BUFFER_REGISTRY, build_logger
from .utils import UsedDataRemover, PeriodicThruputMonitor

@BUFFER_REGISTRY.register('naive')
class NaiveReplayBuffer(IBuffer):
    """
    Overview:
        Naive replay buffer, can store and sample data.
        An naive implementation of replay buffer with no priority or any other advanced features.
        This buffer refers to multi-thread/multi-process and guarantees thread-safe, which means that methods like
        ``sample``, ``push``, ``clear`` are all mutual to each other.
    Interface:
        start, close, push, update, sample, clear, count, state_dict, load_state_dict, default_config
    Property:
        replay_buffer_size, push_count
    """
    config = dict(type='naive', replay_buffer_size=10000, deepcopy=False, enable_track_used_data=False, periodic_thruput_seconds=60)

    def __init__(self, cfg: 'EasyDict', tb_logger: Optional['SummaryWriter']=None, exp_name: Optional[str]='default_experiment', instance_name: Optional[str]='buffer') -> None:
        if False:
            print('Hello World!')
        "\n        Overview:\n            Initialize the buffer\n        Arguments:\n            - cfg (:obj:`dict`): Config dict.\n            - tb_logger (:obj:`Optional['SummaryWriter']`): Outer tb logger. Usually get this argument in serial mode.\n            - exp_name (:obj:`Optional[str]`): Name of this experiment.\n            - instance_name (:obj:`Optional[str]`): Name of this instance.\n        "
        self._exp_name = exp_name
        self._instance_name = instance_name
        self._cfg = cfg
        self._replay_buffer_size = self._cfg.replay_buffer_size
        self._deepcopy = self._cfg.deepcopy
        self._data = [None for _ in range(self._replay_buffer_size)]
        self._valid_count = 0
        self._push_count = 0
        self._tail = 0
        self._lock = LockContext(type_=LockContextType.THREAD_LOCK)
        self._end_flag = False
        self._enable_track_used_data = self._cfg.enable_track_used_data
        if self._enable_track_used_data:
            self._used_data_remover = UsedDataRemover()
        if tb_logger is not None:
            (self._logger, _) = build_logger('./{}/log/{}'.format(self._exp_name, self._instance_name), self._instance_name, need_tb=False)
            self._tb_logger = tb_logger
        else:
            (self._logger, self._tb_logger) = build_logger('./{}/log/{}'.format(self._exp_name, self._instance_name), self._instance_name)
        self._periodic_thruput_monitor = PeriodicThruputMonitor(self._instance_name, EasyDict(seconds=self._cfg.periodic_thruput_seconds), self._logger, self._tb_logger)

    def start(self) -> None:
        if False:
            while True:
                i = 10
        "\n        Overview:\n            Start the buffer's used_data_remover thread if enables track_used_data.\n        "
        if self._enable_track_used_data:
            self._used_data_remover.start()

    def close(self) -> None:
        if False:
            while True:
                i = 10
        "\n        Overview:\n            Clear the buffer; Join the buffer's used_data_remover thread if enables track_used_data.\n        "
        self.clear()
        if self._enable_track_used_data:
            self._used_data_remover.close()
        self._tb_logger.flush()
        self._tb_logger.close()

    def push(self, data: Union[List[Any], Any], cur_collector_envstep: int) -> None:
        if False:
            return 10
        "\n        Overview:\n            Push a data into buffer.\n        Arguments:\n            - data (:obj:`Union[List[Any], Any]`): The data which will be pushed into buffer. Can be one \\\n                (in `Any` type), or many(int `List[Any]` type).\n            - cur_collector_envstep (:obj:`int`): Collector's current env step. \\\n                Not used in naive buffer, but preserved for compatibility.\n        "
        if isinstance(data, list):
            self._extend(data, cur_collector_envstep)
            self._periodic_thruput_monitor.push_data_count += len(data)
        else:
            self._append(data, cur_collector_envstep)
            self._periodic_thruput_monitor.push_data_count += 1

    def sample(self, size: int, cur_learner_iter: int, sample_range: slice=None, replace: bool=False) -> Optional[list]:
        if False:
            return 10
        "\n        Overview:\n            Sample data with length ``size``.\n        Arguments:\n            - size (:obj:`int`): The number of the data that will be sampled.\n            - cur_learner_iter (:obj:`int`): Learner's current iteration.                 Not used in naive buffer, but preserved for compatibility.\n            - sample_range (:obj:`slice`): Buffer slice for sampling, such as `slice(-10, None)`, which                 means only sample among the last 10 data\n            - replace (:obj:`bool`): Whether sample with replacement\n        Returns:\n            - sample_data (:obj:`list`): A list of data with length ``size``.\n        "
        if size == 0:
            return []
        can_sample = self._sample_check(size, replace)
        if not can_sample:
            return None
        with self._lock:
            indices = self._get_indices(size, sample_range, replace)
            sample_data = self._sample_with_indices(indices, cur_learner_iter)
        self._periodic_thruput_monitor.sample_data_count += len(sample_data)
        return sample_data

    def save_data(self, file_name: str):
        if False:
            print('Hello World!')
        if not os.path.exists(os.path.dirname(file_name)):
            if os.path.dirname(file_name) != '':
                os.makedirs(os.path.dirname(file_name))
        hickle.dump(py_obj=self._data, file_obj=file_name)

    def load_data(self, file_name: str):
        if False:
            return 10
        self.push(hickle.load(file_name), 0)

    def _append(self, ori_data: Any, cur_collector_envstep: int=-1) -> None:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Append a data item into ``self._data``.\n        Arguments:\n            - ori_data (:obj:`Any`): The data which will be inserted.\n            - cur_collector_envstep (:obj:`int`): Not used in this method, but preserved for compatibility.\n        '
        with self._lock:
            if self._deepcopy:
                data = copy.deepcopy(ori_data)
            else:
                data = ori_data
            self._push_count += 1
            if self._data[self._tail] is None:
                self._valid_count += 1
                self._periodic_thruput_monitor.valid_count = self._valid_count
            elif self._enable_track_used_data:
                self._used_data_remover.add_used_data(self._data[self._tail])
            self._data[self._tail] = data
            self._tail = (self._tail + 1) % self._replay_buffer_size

    def _extend(self, ori_data: List[Any], cur_collector_envstep: int=-1) -> None:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Extend a data list into queue.\n            Add two keys in each data item, you can refer to ``_append`` for details.\n        Arguments:\n            - ori_data (:obj:`List[Any]`): The data list.\n            - cur_collector_envstep (:obj:`int`): Not used in this method, but preserved for compatibility.\n        '
        with self._lock:
            if self._deepcopy:
                data = copy.deepcopy(ori_data)
            else:
                data = ori_data
            length = len(data)
            if self._tail + length <= self._replay_buffer_size:
                if self._valid_count != self._replay_buffer_size:
                    self._valid_count += length
                    self._periodic_thruput_monitor.valid_count = self._valid_count
                elif self._enable_track_used_data:
                    for i in range(length):
                        self._used_data_remover.add_used_data(self._data[self._tail + i])
                self._push_count += length
                self._data[self._tail:self._tail + length] = data
            else:
                new_tail = self._tail
                data_start = 0
                residual_num = len(data)
                while True:
                    space = self._replay_buffer_size - new_tail
                    L = min(space, residual_num)
                    if self._valid_count != self._replay_buffer_size:
                        self._valid_count += L
                        self._periodic_thruput_monitor.valid_count = self._valid_count
                    elif self._enable_track_used_data:
                        for i in range(L):
                            self._used_data_remover.add_used_data(self._data[new_tail + i])
                    self._push_count += L
                    self._data[new_tail:new_tail + L] = data[data_start:data_start + L]
                    residual_num -= L
                    assert residual_num >= 0
                    if residual_num == 0:
                        break
                    else:
                        new_tail = 0
                        data_start += L
            self._tail = (self._tail + length) % self._replay_buffer_size

    def _sample_check(self, size: int, replace: bool=False) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Check whether this buffer has more than `size` datas to sample.\n        Arguments:\n            - size (:obj:`int`): Number of data that will be sampled.\n            - replace (:obj:`bool`): Whether sample with replacement.\n        Returns:\n            - can_sample (:obj:`bool`): Whether this buffer can sample enough data.\n        '
        if self._valid_count == 0:
            print('The buffer is empty')
            return False
        if self._valid_count < size and (not replace):
            print('No enough elements for sampling without replacement (expect: {} / current: {})'.format(size, self._valid_count))
            return False
        else:
            return True

    def update(self, info: dict) -> None:
        if False:
            return 10
        '\n        Overview:\n            Naive Buffer does not need to update any info, but this method is preserved for compatibility.\n        '
        print('[BUFFER WARNING] Naive Buffer does not need to update any info,                 but `update` method is preserved for compatibility.')

    def clear(self) -> None:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Clear all the data and reset the related variables.\n        '
        with self._lock:
            for i in range(len(self._data)):
                if self._data[i] is not None:
                    if self._enable_track_used_data:
                        self._used_data_remover.add_used_data(self._data[i])
                    self._data[i] = None
            self._valid_count = 0
            self._periodic_thruput_monitor.valid_count = self._valid_count
            self._push_count = 0
            self._tail = 0

    def __del__(self) -> None:
        if False:
            return 10
        '\n        Overview:\n            Call ``close`` to delete the object.\n        '
        self.close()

    def _get_indices(self, size: int, sample_range: slice=None, replace: bool=False) -> list:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Get the sample index list.\n        Arguments:\n            - size (:obj:`int`): The number of the data that will be sampled\n            - sample_range (:obj:`slice`): Buffer slice for sampling, such as `slice(-10, None)`, which \\\n                means only sample among the last 10 data\n        Returns:\n            - index_list (:obj:`list`): A list including all the sample indices, whose length should equal to ``size``.\n        '
        assert self._valid_count <= self._replay_buffer_size
        if self._valid_count == self._replay_buffer_size:
            tail = self._replay_buffer_size
        else:
            tail = self._tail
        if sample_range is None:
            indices = list(np.random.choice(a=tail, size=size, replace=replace))
        else:
            indices = list(range(tail))[sample_range]
            indices = list(np.random.choice(indices, size=size, replace=replace))
        return indices

    def _sample_with_indices(self, indices: List[int], cur_learner_iter: int) -> list:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Sample data with ``indices``.\n        Arguments:\n            - indices (:obj:`List[int]`): A list including all the sample indices.\n            - cur_learner_iter (:obj:`int`): Not used in this method, but preserved for compatibility.\n        Returns:\n            - data (:obj:`list`) Sampled data.\n        '
        data = []
        for idx in indices:
            assert self._data[idx] is not None, idx
            if self._deepcopy:
                copy_data = copy.deepcopy(self._data[idx])
            else:
                copy_data = self._data[idx]
            data.append(copy_data)
        return data

    def count(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Count how many valid datas there are in the buffer.\n        Returns:\n            - count (:obj:`int`): Number of valid data.\n        '
        return self._valid_count

    def state_dict(self) -> dict:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Provide a state dict to keep a record of current buffer.\n        Returns:\n            - state_dict (:obj:`Dict[str, Any]`): A dict containing all important values in the buffer.                 With the dict, one can easily reproduce the buffer.\n        '
        return {'data': self._data, 'tail': self._tail, 'valid_count': self._valid_count, 'push_count': self._push_count}

    def load_state_dict(self, _state_dict: dict) -> None:
        if False:
            return 10
        '\n        Overview:\n            Load state dict to reproduce the buffer.\n        Returns:\n            - state_dict (:obj:`Dict[str, Any]`): A dict containing all important values in the buffer.\n        '
        assert 'data' in _state_dict
        if set(_state_dict.keys()) == set(['data']):
            self._extend(_state_dict['data'])
        else:
            for (k, v) in _state_dict.items():
                setattr(self, '_{}'.format(k), v)

    @property
    def replay_buffer_size(self) -> int:
        if False:
            return 10
        return self._replay_buffer_size

    @property
    def push_count(self) -> int:
        if False:
            return 10
        return self._push_count

@BUFFER_REGISTRY.register('elastic')
class ElasticReplayBuffer(NaiveReplayBuffer):
    """
    Overview:
        Elastic replay buffer, it stores data and support dynamically change the buffer size.
        An naive implementation of replay buffer with no priority or any other advanced features.
        This buffer refers to multi-thread/multi-process and guarantees thread-safe, which means that methods like
        ``sample``, ``push``, ``clear`` are all mutual to each other.
    Interface:
        start, close, push, update, sample, clear, count, state_dict, load_state_dict, default_config
    Property:
        replay_buffer_size, push_count
    """
    config = dict(type='elastic', replay_buffer_size=10000, deepcopy=False, enable_track_used_data=False, periodic_thruput_seconds=60)

    def __init__(self, cfg: 'EasyDict', tb_logger: Optional['SummaryWriter']=None, exp_name: Optional[str]='default_experiment', instance_name: Optional[str]='buffer') -> None:
        if False:
            print('Hello World!')
        "\n        Overview:\n            Initialize the buffer\n        Arguments:\n            - cfg (:obj:`dict`): Config dict.\n            - tb_logger (:obj:`Optional['SummaryWriter']`): Outer tb logger. Usually get this argument in serial mode.\n            - exp_name (:obj:`Optional[str]`): Name of this experiment.\n            - instance_name (:obj:`Optional[str]`): Name of this instance.\n        "
        super().__init__(cfg, tb_logger, exp_name, instance_name)
        self._set_buffer_size = self._cfg.set_buffer_size
        self._current_buffer_size = self._set_buffer_size(0)

    def _sample_check(self, size: int, replace: bool=False) -> bool:
        if False:
            return 10
        '\n        Overview:\n            Check whether this buffer has more than `size` datas to sample.\n        Arguments:\n            - size (:obj:`int`): Number of data that will be sampled.\n            - replace (:obj:`bool`): Whether sample with replacement.\n        Returns:\n            - can_sample (:obj:`bool`): Whether this buffer can sample enough data.\n        '
        valid_count = min(self._valid_count, self._current_buffer_size)
        if valid_count == 0:
            print('The buffer is empty')
            return False
        if valid_count < size and (not replace):
            print('No enough elements for sampling without replacement (expect: {} / current: {})'.format(size, self._valid_count))
            return False
        else:
            return True

    def _get_indices(self, size: int, sample_range: slice=None, replace: bool=False) -> list:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Get the sample index list.\n        Arguments:\n            - size (:obj:`int`): The number of the data that will be sampled.\n            - replace (:obj:`bool`): Whether sample with replacement.\n        Returns:\n            - index_list (:obj:`list`): A list including all the sample indices, whose length should equal to ``size``.\n        '
        assert self._valid_count <= self._replay_buffer_size
        assert sample_range is None
        range = min(self._valid_count, self._current_buffer_size)
        indices = list((self._tail - 1 - np.random.choice(a=range, size=size, replace=replace)) % self._replay_buffer_size)
        return indices

    def update(self, envstep):
        if False:
            for i in range(10):
                print('nop')
        self._current_buffer_size = self._set_buffer_size(envstep)

@BUFFER_REGISTRY.register('sequence')
class SequenceReplayBuffer(NaiveReplayBuffer):
    """
    Overview:
    Interface:
        start, close, push, update, sample, clear, count, state_dict, load_state_dict, default_config
    Property:
        replay_buffer_size, push_count
    """

    def sample(self, batch: int, sequence: int, cur_learner_iter: int, sample_range: slice=None, replace: bool=False) -> Optional[list]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Overview:\n            Sample data with length ``size``.\n        Arguments:\n            - size (:obj:`int`): The number of the data that will be sampled.\n            - sequence (:obj:`int`): The length of the sequence of a data that will be sampled.\n            - cur_learner_iter (:obj:`int`): Learner's current iteration.                 Not used in naive buffer, but preserved for compatibility.\n            - sample_range (:obj:`slice`): Buffer slice for sampling, such as `slice(-10, None)`, which                 means only sample among the last 10 data\n            - replace (:obj:`bool`): Whether sample with replacement\n        Returns:\n            - sample_data (:obj:`list`): A list of data with length ``size``.\n        "
        if batch == 0:
            return []
        can_sample = self._sample_check(batch * sequence, replace)
        if not can_sample:
            return None
        with self._lock:
            indices = self._get_indices(batch, sequence, sample_range, replace)
            sample_data = self._sample_with_indices(indices, sequence, cur_learner_iter)
        self._periodic_thruput_monitor.sample_data_count += len(sample_data)
        return sample_data

    def _get_indices(self, size: int, sequence: int, sample_range: slice=None, replace: bool=False) -> list:
        if False:
            return 10
        '\n        Overview:\n            Get the sample index list.\n        Arguments:\n            - size (:obj:`int`): The number of the data that will be sampled\n            - sample_range (:obj:`slice`): Buffer slice for sampling, such as `slice(-10, None)`, which \\\n                means only sample among the last 10 data\n        Returns:\n            - index_list (:obj:`list`): A list including all the sample indices, whose length should equal to ``size``.\n        '
        assert self._valid_count <= self._replay_buffer_size
        if self._valid_count == self._replay_buffer_size:
            tail = self._replay_buffer_size
        else:
            tail = self._tail
        episodes = math.ceil(self._valid_count / 500)
        batch = 0
        indices = []
        if sample_range is None:
            while batch < size:
                episode = np.random.choice(episodes)
                length = tail - episode * 500 if tail - episode * 500 < 500 else 500
                available = length - sequence
                if available < 1:
                    continue
                list(range(episode * 500, episode * 500 + available))
                indices.append(np.random.randint(episode * 500, episode * 500 + available + 1))
                batch += 1
        else:
            raise NotImplementedError('sample_range is not implemented in this version')
        return indices

    def _sample_with_indices(self, indices: List[int], sequence: int, cur_learner_iter: int) -> list:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Sample data with ``indices``.\n        Arguments:\n            - indices (:obj:`List[int]`): A list including all the sample indices.\n            - cur_learner_iter (:obj:`int`): Not used in this method, but preserved for compatibility.\n        Returns:\n            - data (:obj:`list`) Sampled data.\n        '
        data = []
        for idx in indices:
            assert self._data[idx] is not None, idx
            if self._deepcopy:
                copy_data = copy.deepcopy(self._data[idx:idx + sequence])
            else:
                copy_data = self._data[idx:idx + sequence]
            data.append(copy_data)
        return data