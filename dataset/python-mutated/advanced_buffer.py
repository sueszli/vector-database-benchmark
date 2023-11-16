import os
import copy
import time
from typing import Union, Any, Optional, List, Dict, Tuple
import numpy as np
import hickle
from ding.worker.replay_buffer import IBuffer
from ding.utils import SumSegmentTree, MinSegmentTree, BUFFER_REGISTRY
from ding.utils import LockContext, LockContextType, build_logger, get_rank
from ding.utils.autolog import TickTime
from .utils import UsedDataRemover, generate_id, SampledDataAttrMonitor, PeriodicThruputMonitor, ThruputController

def to_positive_index(idx: Union[int, None], size: int) -> int:
    if False:
        i = 10
        return i + 15
    if idx is None or idx >= 0:
        return idx
    else:
        return size + idx

@BUFFER_REGISTRY.register('advanced')
class AdvancedReplayBuffer(IBuffer):
    """
    Overview:
        Prioritized replay buffer derived from ``NaiveReplayBuffer``.
        This replay buffer adds:

            1) Prioritized experience replay implemented by segment tree.
            2) Data quality monitor. Monitor use count and staleness of each data.
            3) Throughput monitor and control.
            4) Logger. Log 2) and 3) in tensorboard or text.
    Interface:
        start, close, push, update, sample, clear, count, state_dict, load_state_dict, default_config
    Property:
        beta, replay_buffer_size, push_count
    """
    config = dict(type='advanced', replay_buffer_size=4096, max_use=float('inf'), max_staleness=float('inf'), alpha=0.6, beta=0.4, anneal_step=int(100000.0), enable_track_used_data=False, deepcopy=False, thruput_controller=dict(push_sample_rate_limit=dict(max=float('inf'), min=0), window_seconds=30, sample_min_limit_ratio=1), monitor=dict(sampled_data_attr=dict(average_range=5, print_freq=200), periodic_thruput=dict(seconds=60)))

    def __init__(self, cfg: dict, tb_logger: Optional['SummaryWriter']=None, exp_name: Optional[str]='default_experiment', instance_name: Optional[str]='buffer') -> int:
        if False:
            return 10
        "\n        Overview:\n            Initialize the buffer\n        Arguments:\n            - cfg (:obj:`dict`): Config dict.\n            - tb_logger (:obj:`Optional['SummaryWriter']`): Outer tb logger. Usually get this argument in serial mode.\n            - exp_name (:obj:`Optional[str]`): Name of this experiment.\n            - instance_name (:obj:`Optional[str]`): Name of this instance.\n        "
        self._exp_name = exp_name
        self._instance_name = instance_name
        self._end_flag = False
        self._cfg = cfg
        self._rank = get_rank()
        self._replay_buffer_size = self._cfg.replay_buffer_size
        self._deepcopy = self._cfg.deepcopy
        self._data = [None for _ in range(self._replay_buffer_size)]
        self._valid_count = 0
        self._push_count = 0
        self._tail = 0
        self._next_unique_id = 0
        self._lock = LockContext(type_=LockContextType.THREAD_LOCK)
        self._head = 0
        self._use_count = {idx: 0 for idx in range(self._cfg.replay_buffer_size)}
        self._max_priority = 1.0
        self._eps = 1e-05
        self.check_list = [lambda x: isinstance(x, dict)]
        self._max_use = self._cfg.max_use
        self._max_staleness = self._cfg.max_staleness
        self.alpha = self._cfg.alpha
        assert 0 <= self.alpha <= 1, self.alpha
        self._beta = self._cfg.beta
        assert 0 <= self._beta <= 1, self._beta
        self._anneal_step = self._cfg.anneal_step
        if self._anneal_step != 0:
            self._beta_anneal_step = (1 - self._beta) / self._anneal_step
        capacity = int(np.power(2, np.ceil(np.log2(self.replay_buffer_size))))
        self._sum_tree = SumSegmentTree(capacity)
        self._min_tree = MinSegmentTree(capacity)
        push_sample_rate_limit = self._cfg.thruput_controller.push_sample_rate_limit
        self._always_can_push = True if push_sample_rate_limit['max'] == float('inf') else False
        self._always_can_sample = True if push_sample_rate_limit['min'] == 0 else False
        self._use_thruput_controller = not self._always_can_push or not self._always_can_sample
        if self._use_thruput_controller:
            self._thruput_controller = ThruputController(self._cfg.thruput_controller)
        self._sample_min_limit_ratio = self._cfg.thruput_controller.sample_min_limit_ratio
        assert self._sample_min_limit_ratio >= 1
        monitor_cfg = self._cfg.monitor
        if self._rank == 0:
            if tb_logger is not None:
                (self._logger, _) = build_logger('./{}/log/{}'.format(self._exp_name, self._instance_name), self._instance_name, need_tb=False)
                self._tb_logger = tb_logger
            else:
                (self._logger, self._tb_logger) = build_logger('./{}/log/{}'.format(self._exp_name, self._instance_name), self._instance_name)
        else:
            (self._logger, _) = build_logger('./{}/log/{}'.format(self._exp_name, self._instance_name), self._instance_name, need_tb=False)
            self._tb_logger = None
        self._start_time = time.time()
        self._cur_learner_iter = -1
        self._cur_collector_envstep = -1
        self._sampled_data_attr_print_count = 0
        self._sampled_data_attr_monitor = SampledDataAttrMonitor(TickTime(), expire=monitor_cfg.sampled_data_attr.average_range)
        self._sampled_data_attr_print_freq = monitor_cfg.sampled_data_attr.print_freq
        if self._rank == 0:
            self._periodic_thruput_monitor = PeriodicThruputMonitor(self._instance_name, monitor_cfg.periodic_thruput, self._logger, self._tb_logger)
        self._enable_track_used_data = self._cfg.enable_track_used_data
        if self._enable_track_used_data:
            self._used_data_remover = UsedDataRemover()

    def start(self) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Overview:\n            Start the buffer's used_data_remover thread if enables track_used_data.\n        "
        if self._enable_track_used_data:
            self._used_data_remover.start()

    def close(self) -> None:
        if False:
            print('Hello World!')
        "\n        Overview:\n            Clear the buffer; Join the buffer's used_data_remover thread if enables track_used_data.\n            Join periodic throughtput monitor, flush tensorboard logger.\n        "
        if self._end_flag:
            return
        self._end_flag = True
        self.clear()
        if self._rank == 0:
            self._periodic_thruput_monitor.close()
            self._tb_logger.flush()
            self._tb_logger.close()
        if self._enable_track_used_data:
            self._used_data_remover.close()

    def sample(self, size: int, cur_learner_iter: int, sample_range: slice=None) -> Optional[list]:
        if False:
            return 10
        "\n        Overview:\n            Sample data with length ``size``.\n        Arguments:\n            - size (:obj:`int`): The number of the data that will be sampled.\n            - cur_learner_iter (:obj:`int`): Learner's current iteration, used to calculate staleness.\n            - sample_range (:obj:`slice`): Buffer slice for sampling, such as `slice(-10, None)`, which                 means only sample among the last 10 data\n        Returns:\n            - sample_data (:obj:`list`): A list of data with length ``size``\n        ReturnsKeys:\n            - necessary: original keys(e.g. `obs`, `action`, `next_obs`, `reward`, `info`),                 `replay_unique_id`, `replay_buffer_idx`\n            - optional(if use priority): `IS`, `priority`\n        "
        if size == 0:
            return []
        (can_sample_stalenss, staleness_info) = self._sample_check(size, cur_learner_iter)
        if self._always_can_sample:
            (can_sample_thruput, thruput_info) = (True, "Always can sample because push_sample_rate_limit['min'] == 0")
        else:
            (can_sample_thruput, thruput_info) = self._thruput_controller.can_sample(size)
        if not can_sample_stalenss or not can_sample_thruput:
            self._logger.info('Refuse to sample due to -- \nstaleness: {}, {} \nthruput: {}, {}'.format(not can_sample_stalenss, staleness_info, not can_sample_thruput, thruput_info))
            return None
        with self._lock:
            indices = self._get_indices(size, sample_range)
            result = self._sample_with_indices(indices, cur_learner_iter)
            if not self._deepcopy and len(indices) != len(set(indices)):
                for (i, index) in enumerate(indices):
                    tmp = []
                    for j in range(i + 1, size):
                        if index == indices[j]:
                            tmp.append(j)
                    for j in tmp:
                        result[j] = copy.deepcopy(result[j])
            self._monitor_update_of_sample(result, cur_learner_iter)
            return result

    def push(self, data: Union[List[Any], Any], cur_collector_envstep: int) -> None:
        if False:
            print('Hello World!')
        "\n        Overview:\n            Push a data into buffer.\n        Arguments:\n            - data (:obj:`Union[List[Any], Any]`): The data which will be pushed into buffer. Can be one \\\n                (in `Any` type), or many(int `List[Any]` type).\n            - cur_collector_envstep (:obj:`int`): Collector's current env step.\n        "
        push_size = len(data) if isinstance(data, list) else 1
        if self._always_can_push:
            (can_push, push_info) = (True, "Always can push because push_sample_rate_limit['max'] == float('inf')")
        else:
            (can_push, push_info) = self._thruput_controller.can_push(push_size)
        if not can_push:
            self._logger.info('Refuse to push because {}'.format(push_info))
            return
        if isinstance(data, list):
            self._extend(data, cur_collector_envstep)
        else:
            self._append(data, cur_collector_envstep)

    def save_data(self, file_name: str):
        if False:
            for i in range(10):
                print('nop')
        if not os.path.exists(os.path.dirname(file_name)):
            if os.path.dirname(file_name) != '':
                os.makedirs(os.path.dirname(file_name))
        hickle.dump(py_obj=self._data, file_obj=file_name)

    def load_data(self, file_name: str):
        if False:
            return 10
        self.push(hickle.load(file_name), 0)

    def _sample_check(self, size: int, cur_learner_iter: int) -> Tuple[bool, str]:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Do preparations for sampling and check whether data is enough for sampling\n            Preparation includes removing stale datas in ``self._data``.\n            Check includes judging whether this buffer has more than ``size`` datas to sample.\n        Arguments:\n            - size (:obj:`int`): The number of the data that will be sampled.\n            - cur_learner_iter (:obj:`int`): Learner\'s current iteration, used to calculate staleness.\n        Returns:\n            - can_sample (:obj:`bool`): Whether this buffer can sample enough data.\n            - str_info (:obj:`str`): Str type info, explaining why cannot sample. (If can sample, return "Can sample")\n\n        .. note::\n            This function must be called before data sample.\n        '
        staleness_remove_count = 0
        with self._lock:
            if self._max_staleness != float('inf'):
                p = self._head
                while True:
                    if self._data[p] is not None:
                        staleness = self._calculate_staleness(p, cur_learner_iter)
                        if staleness >= self._max_staleness:
                            self._remove(p)
                            staleness_remove_count += 1
                        else:
                            break
                    p = (p + 1) % self._replay_buffer_size
                    if p == self._tail:
                        break
            str_info = 'Remove {} elements due to staleness. '.format(staleness_remove_count)
            if self._valid_count / size < self._sample_min_limit_ratio:
                str_info += 'Not enough for sampling. valid({}) / sample({}) < sample_min_limit_ratio({})'.format(self._valid_count, size, self._sample_min_limit_ratio)
                return (False, str_info)
            else:
                str_info += 'Can sample.'
                return (True, str_info)

    def _append(self, ori_data: Any, cur_collector_envstep: int=-1) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Overview:\n            Append a data item into queue.\n            Add two keys in data:\n\n                - replay_unique_id: The data item's unique id, using ``generate_id`` to generate it.\n                - replay_buffer_idx: The data item's position index in the queue, this position may already have an \\\n                    old element, then it would be replaced by this new input one. using ``self._tail`` to locate.\n        Arguments:\n            - ori_data (:obj:`Any`): The data which will be inserted.\n            - cur_collector_envstep (:obj:`int`): Collector's current env step, used to draw tensorboard.\n        "
        with self._lock:
            if self._deepcopy:
                data = copy.deepcopy(ori_data)
            else:
                data = ori_data
            try:
                assert self._data_check(data)
            except AssertionError:
                self._logger.info('Illegal data type [{}], reject it...'.format(type(data)))
                return
            self._push_count += 1
            if self._data[self._tail] is not None:
                self._head = (self._tail + 1) % self._replay_buffer_size
            self._remove(self._tail)
            data['replay_unique_id'] = generate_id(self._instance_name, self._next_unique_id)
            data['replay_buffer_idx'] = self._tail
            self._set_weight(data)
            self._data[self._tail] = data
            self._valid_count += 1
            if self._rank == 0:
                self._periodic_thruput_monitor.valid_count = self._valid_count
            self._tail = (self._tail + 1) % self._replay_buffer_size
            self._next_unique_id += 1
            self._monitor_update_of_push(1, cur_collector_envstep)

    def _extend(self, ori_data: List[Any], cur_collector_envstep: int=-1) -> None:
        if False:
            print('Hello World!')
        "\n        Overview:\n            Extend a data list into queue.\n            Add two keys in each data item, you can refer to ``_append`` for more details.\n        Arguments:\n            - ori_data (:obj:`List[Any]`): The data list.\n            - cur_collector_envstep (:obj:`int`): Collector's current env step, used to draw tensorboard.\n        "
        with self._lock:
            if self._deepcopy:
                data = copy.deepcopy(ori_data)
            else:
                data = ori_data
            check_result = [self._data_check(d) for d in data]
            valid_data = [d for (d, flag) in zip(data, check_result) if flag]
            length = len(valid_data)
            if self._tail + length <= self._replay_buffer_size:
                for j in range(self._tail, self._tail + length):
                    if self._data[j] is not None:
                        self._head = (j + 1) % self._replay_buffer_size
                    self._remove(j)
                for i in range(length):
                    valid_data[i]['replay_unique_id'] = generate_id(self._instance_name, self._next_unique_id + i)
                    valid_data[i]['replay_buffer_idx'] = (self._tail + i) % self._replay_buffer_size
                    self._set_weight(valid_data[i])
                    self._push_count += 1
                self._data[self._tail:self._tail + length] = valid_data
            else:
                data_start = self._tail
                valid_data_start = 0
                residual_num = len(valid_data)
                while True:
                    space = self._replay_buffer_size - data_start
                    L = min(space, residual_num)
                    for j in range(data_start, data_start + L):
                        if self._data[j] is not None:
                            self._head = (j + 1) % self._replay_buffer_size
                        self._remove(j)
                    for i in range(valid_data_start, valid_data_start + L):
                        valid_data[i]['replay_unique_id'] = generate_id(self._instance_name, self._next_unique_id + i)
                        valid_data[i]['replay_buffer_idx'] = (self._tail + i) % self._replay_buffer_size
                        self._set_weight(valid_data[i])
                        self._push_count += 1
                    self._data[data_start:data_start + L] = valid_data[valid_data_start:valid_data_start + L]
                    residual_num -= L
                    if residual_num <= 0:
                        break
                    else:
                        data_start = 0
                        valid_data_start += L
            self._valid_count += len(valid_data)
            if self._rank == 0:
                self._periodic_thruput_monitor.valid_count = self._valid_count
            self._tail = (self._tail + length) % self._replay_buffer_size
            self._next_unique_id += length
            self._monitor_update_of_push(length, cur_collector_envstep)

    def update(self, info: dict) -> None:
        if False:
            print('Hello World!')
        "\n        Overview:\n            Update a data's priority. Use `repaly_buffer_idx` to locate, and use `replay_unique_id` to verify.\n        Arguments:\n            - info (:obj:`dict`): Info dict containing all necessary keys for priority update.\n        ArgumentsKeys:\n            - necessary: `replay_unique_id`, `replay_buffer_idx`, `priority`. All values are lists with the same length.\n        "
        with self._lock:
            if 'priority' not in info:
                return
            data = [info['replay_unique_id'], info['replay_buffer_idx'], info['priority']]
            for (id_, idx, priority) in zip(*data):
                if self._data[idx] is not None and self._data[idx]['replay_unique_id'] == id_:
                    assert priority >= 0, priority
                    assert self._data[idx]['replay_buffer_idx'] == idx
                    self._data[idx]['priority'] = priority + self._eps
                    self._set_weight(self._data[idx])
                    self._max_priority = max(self._max_priority, priority)
                else:
                    self._logger.debug('[Skip Update]: buffer_idx: {}; id_in_buffer: {}; id_in_update_info: {}'.format(idx, id_, priority))

    def clear(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Clear all the data and reset the related variables.\n        '
        with self._lock:
            for i in range(len(self._data)):
                self._remove(i)
            assert self._valid_count == 0, self._valid_count
            self._head = 0
            self._tail = 0
            self._max_priority = 1.0

    def __del__(self) -> None:
        if False:
            return 10
        '\n        Overview:\n            Call ``close`` to delete the object.\n        '
        if not self._end_flag:
            self.close()

    def _set_weight(self, data: Dict) -> None:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Set sumtree and mintree\'s weight of the input data according to its priority.\n            If input data does not have key "priority", it would set to ``self._max_priority`` instead.\n        Arguments:\n            - data (:obj:`Dict`): The data whose priority(weight) in segement tree should be set/updated.\n        '
        if 'priority' not in data.keys() or data['priority'] is None:
            data['priority'] = self._max_priority
        weight = data['priority'] ** self.alpha
        idx = data['replay_buffer_idx']
        self._sum_tree[idx] = weight
        self._min_tree[idx] = weight

    def _data_check(self, d: Any) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Data legality check, using rules(functions) in ``self.check_list``.\n        Arguments:\n            - d (:obj:`Any`): The data which needs to be checked.\n        Returns:\n            - result (:obj:`bool`): Whether the data passes the check.\n        '
        return all([fn(d) for fn in self.check_list])

    def _get_indices(self, size: int, sample_range: slice=None) -> list:
        if False:
            return 10
        '\n        Overview:\n            Get the sample index list according to the priority probability.\n        Arguments:\n            - size (:obj:`int`): The number of the data that will be sampled\n        Returns:\n            - index_list (:obj:`list`): A list including all the sample indices, whose length should equal to ``size``.\n        '
        intervals = np.array([i * 1.0 / size for i in range(size)])
        mass = intervals + np.random.uniform(size=(size,)) * 1.0 / size
        if sample_range is None:
            mass *= self._sum_tree.reduce()
        else:
            start = to_positive_index(sample_range.start, self._replay_buffer_size)
            end = to_positive_index(sample_range.stop, self._replay_buffer_size)
            a = self._sum_tree.reduce(0, start)
            b = self._sum_tree.reduce(0, end)
            mass = mass * (b - a) + a
        return [self._sum_tree.find_prefixsum_idx(m) for m in mass]

    def _remove(self, idx: int, use_too_many_times: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Remove a data(set the element in the list to ``None``) and update corresponding variables,\n            e.g. sum_tree, min_tree, valid_count.\n        Arguments:\n            - idx (:obj:`int`): Data at this position will be removed.\n        '
        if use_too_many_times:
            if self._enable_track_used_data:
                self._data[idx]['priority'] = 0
                self._sum_tree[idx] = self._sum_tree.neutral_element
                self._min_tree[idx] = self._min_tree.neutral_element
                return
            elif idx == self._head:
                self._head = (self._head + 1) % self._replay_buffer_size
        if self._data[idx] is not None:
            if self._enable_track_used_data:
                self._used_data_remover.add_used_data(self._data[idx])
            self._valid_count -= 1
            if self._rank == 0:
                self._periodic_thruput_monitor.valid_count = self._valid_count
                self._periodic_thruput_monitor.remove_data_count += 1
            self._data[idx] = None
            self._sum_tree[idx] = self._sum_tree.neutral_element
            self._min_tree[idx] = self._min_tree.neutral_element
            self._use_count[idx] = 0

    def _sample_with_indices(self, indices: List[int], cur_learner_iter: int) -> list:
        if False:
            while True:
                i = 10
        "\n        Overview:\n            Sample data with ``indices``; Remove a data item if it is used for too many times.\n        Arguments:\n            - indices (:obj:`List[int]`): A list including all the sample indices.\n            - cur_learner_iter (:obj:`int`): Learner's current iteration, used to calculate staleness.\n        Returns:\n            - data (:obj:`list`) Sampled data.\n        "
        sum_tree_root = self._sum_tree.reduce()
        p_min = self._min_tree.reduce() / sum_tree_root
        max_weight = (self._valid_count * p_min) ** (-self._beta)
        data = []
        for idx in indices:
            assert self._data[idx] is not None
            assert self._data[idx]['replay_buffer_idx'] == idx, (self._data[idx]['replay_buffer_idx'], idx)
            if self._deepcopy:
                copy_data = copy.deepcopy(self._data[idx])
            else:
                copy_data = self._data[idx]
            self._use_count[idx] += 1
            copy_data['staleness'] = self._calculate_staleness(idx, cur_learner_iter)
            copy_data['use'] = self._use_count[idx]
            p_sample = self._sum_tree[idx] / sum_tree_root
            weight = (self._valid_count * p_sample) ** (-self._beta)
            copy_data['IS'] = weight / max_weight
            data.append(copy_data)
        if self._max_use != float('inf'):
            for idx in indices:
                if self._use_count[idx] >= self._max_use:
                    self._remove(idx, use_too_many_times=True)
        if self._anneal_step != 0:
            self._beta = min(1.0, self._beta + self._beta_anneal_step)
        return data

    def _monitor_update_of_push(self, add_count: int, cur_collector_envstep: int=-1) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Update values in monitor, then update text logger and tensorboard logger.\n            Called in ``_append`` and ``_extend``.\n        Arguments:\n            - add_count (:obj:`int`): How many datas are added into buffer.\n            - cur_collector_envstep (:obj:`int`): Collector envstep, passed in by collector.\n        '
        if self._rank == 0:
            self._periodic_thruput_monitor.push_data_count += add_count
        if self._use_thruput_controller:
            self._thruput_controller.history_push_count += add_count
        self._cur_collector_envstep = cur_collector_envstep

    def _monitor_update_of_sample(self, sample_data: list, cur_learner_iter: int) -> None:
        if False:
            print('Hello World!')
        "\n        Overview:\n            Update values in monitor, then update text logger and tensorboard logger.\n            Called in ``sample``.\n        Arguments:\n            - sample_data (:obj:`list`): Sampled data. Used to get sample length and data's attributes, \\\n                e.g. use, priority, staleness, etc.\n            - cur_learner_iter (:obj:`int`): Learner iteration, passed in by learner.\n        "
        if self._rank == 0:
            self._periodic_thruput_monitor.sample_data_count += len(sample_data)
        if self._use_thruput_controller:
            self._thruput_controller.history_sample_count += len(sample_data)
        self._cur_learner_iter = cur_learner_iter
        use_avg = sum([d['use'] for d in sample_data]) / len(sample_data)
        use_max = max([d['use'] for d in sample_data])
        priority_avg = sum([d['priority'] for d in sample_data]) / len(sample_data)
        priority_max = max([d['priority'] for d in sample_data])
        priority_min = min([d['priority'] for d in sample_data])
        staleness_avg = sum([d['staleness'] for d in sample_data]) / len(sample_data)
        staleness_max = max([d['staleness'] for d in sample_data])
        self._sampled_data_attr_monitor.use_avg = use_avg
        self._sampled_data_attr_monitor.use_max = use_max
        self._sampled_data_attr_monitor.priority_avg = priority_avg
        self._sampled_data_attr_monitor.priority_max = priority_max
        self._sampled_data_attr_monitor.priority_min = priority_min
        self._sampled_data_attr_monitor.staleness_avg = staleness_avg
        self._sampled_data_attr_monitor.staleness_max = staleness_max
        self._sampled_data_attr_monitor.time.step()
        out_dict = {'use_avg': self._sampled_data_attr_monitor.avg['use'](), 'use_max': self._sampled_data_attr_monitor.max['use'](), 'priority_avg': self._sampled_data_attr_monitor.avg['priority'](), 'priority_max': self._sampled_data_attr_monitor.max['priority'](), 'priority_min': self._sampled_data_attr_monitor.min['priority'](), 'staleness_avg': self._sampled_data_attr_monitor.avg['staleness'](), 'staleness_max': self._sampled_data_attr_monitor.max['staleness'](), 'beta': self._beta}
        if self._sampled_data_attr_print_count % self._sampled_data_attr_print_freq == 0 and self._rank == 0:
            self._logger.info('=== Sample data {} Times ==='.format(self._sampled_data_attr_print_count))
            self._logger.info(self._logger.get_tabulate_vars_hor(out_dict))
            for (k, v) in out_dict.items():
                iter_metric = self._cur_learner_iter if self._cur_learner_iter != -1 else None
                step_metric = self._cur_collector_envstep if self._cur_collector_envstep != -1 else None
                if iter_metric is not None:
                    self._tb_logger.add_scalar('{}_iter/'.format(self._instance_name) + k, v, iter_metric)
                if step_metric is not None:
                    self._tb_logger.add_scalar('{}_step/'.format(self._instance_name) + k, v, step_metric)
        self._sampled_data_attr_print_count += 1

    def _calculate_staleness(self, pos_index: int, cur_learner_iter: int) -> Optional[int]:
        if False:
            return 10
        "\n        Overview:\n            Calculate a data's staleness according to its own attribute ``collect_iter``\n            and input parameter ``cur_learner_iter``.\n        Arguments:\n            - pos_index (:obj:`int`): The position index. Staleness of the data at this index will be calculated.\n            - cur_learner_iter (:obj:`int`): Learner's current iteration, used to calculate staleness.\n        Returns:\n            - staleness (:obj:`int`): Staleness of data at position ``pos_index``.\n\n        .. note::\n            Caller should guarantee that data at ``pos_index`` is not None; Otherwise this function may raise an error.\n        "
        if self._data[pos_index] is None:
            raise ValueError("Prioritized's data at index {} is None".format(pos_index))
        else:
            collect_iter = self._data[pos_index].get('collect_iter', cur_learner_iter + 1)
            if isinstance(collect_iter, list):
                collect_iter = min(collect_iter)
            staleness = cur_learner_iter - collect_iter
            return staleness

    def count(self) -> int:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Count how many valid datas there are in the buffer.\n        Returns:\n            - count (:obj:`int`): Number of valid data.\n        '
        return self._valid_count

    @property
    def beta(self) -> float:
        if False:
            return 10
        return self._beta

    @beta.setter
    def beta(self, beta: float) -> None:
        if False:
            return 10
        self._beta = beta

    def state_dict(self) -> dict:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Provide a state dict to keep a record of current buffer.\n        Returns:\n            - state_dict (:obj:`Dict[str, Any]`): A dict containing all important values in the buffer.                 With the dict, one can easily reproduce the buffer.\n        '
        return {'data': self._data, 'use_count': self._use_count, 'tail': self._tail, 'max_priority': self._max_priority, 'anneal_step': self._anneal_step, 'beta': self._beta, 'head': self._head, 'next_unique_id': self._next_unique_id, 'valid_count': self._valid_count, 'push_count': self._push_count, 'sum_tree': self._sum_tree, 'min_tree': self._min_tree}

    def load_state_dict(self, _state_dict: dict, deepcopy: bool=False) -> None:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Load state dict to reproduce the buffer.\n        Returns:\n            - state_dict (:obj:`Dict[str, Any]`): A dict containing all important values in the buffer.\n        '
        assert 'data' in _state_dict
        if set(_state_dict.keys()) == set(['data']):
            self._extend(_state_dict['data'])
        else:
            for (k, v) in _state_dict.items():
                if deepcopy:
                    setattr(self, '_{}'.format(k), copy.deepcopy(v))
                else:
                    setattr(self, '_{}'.format(k), v)

    @property
    def replay_buffer_size(self) -> int:
        if False:
            while True:
                i = 10
        return self._replay_buffer_size

    @property
    def push_count(self) -> int:
        if False:
            print('Hello World!')
        return self._push_count