import os
import time
from typing import List, Union, Dict, Callable, Any
from functools import partial
from queue import Queue
from threading import Thread
from ding.utils import read_file, save_file, get_data_decompressor, COMM_LEARNER_REGISTRY
from ding.utils.file_helper import read_from_di_store
from ding.interaction import Slave, TaskFail
from .base_comm_learner import BaseCommLearner
from ..learner_hook import LearnerHook

class LearnerSlave(Slave):
    """
    Overview:
        A slave, whose master is coordinator.
        Used to pass message between comm learner and coordinator.
    """

    def __init__(self, *args, callback_fn: Dict[str, Callable], **kwargs) -> None:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Init callback functions additionally. Callback functions are methods in comm learner.\n        '
        super().__init__(*args, **kwargs)
        self._callback_fn = callback_fn

    def _process_task(self, task: dict) -> Union[dict, TaskFail]:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Process a task according to input task info dict, which is passed in by master coordinator.\n            For each type of task, you can refer to corresponding callback function in comm learner for details.\n        Arguments:\n            - cfg (:obj:`EasyDict`): Task dict. Must contain key "name".\n        Returns:\n            - result (:obj:`Union[dict, TaskFail]`): Task result dict, or task fail exception.\n        '
        task_name = task['name']
        if task_name == 'resource':
            return self._callback_fn['deal_with_resource']()
        elif task_name == 'learner_start_task':
            self._current_task_info = task['task_info']
            self._callback_fn['deal_with_learner_start'](self._current_task_info)
            return {'message': 'learner task has started'}
        elif task_name == 'learner_get_data_task':
            data_demand = self._callback_fn['deal_with_get_data']()
            ret = {'task_id': self._current_task_info['task_id'], 'buffer_id': self._current_task_info['buffer_id']}
            ret.update(data_demand)
            return ret
        elif task_name == 'learner_learn_task':
            info = self._callback_fn['deal_with_learner_learn'](task['data'])
            data = {'info': info}
            data['buffer_id'] = self._current_task_info['buffer_id']
            data['task_id'] = self._current_task_info['task_id']
            return data
        elif task_name == 'learner_close_task':
            self._callback_fn['deal_with_learner_close']()
            return {'task_id': self._current_task_info['task_id'], 'buffer_id': self._current_task_info['buffer_id']}
        else:
            raise TaskFail(result={'message': 'task name error'}, message='illegal learner task <{}>'.format(task_name))

@COMM_LEARNER_REGISTRY.register('flask_fs')
class FlaskFileSystemLearner(BaseCommLearner):
    """
    Overview:
        An implementation of CommLearner, using flask and the file system.
    Interfaces:
        __init__, send_policy, get_data, send_learn_info, start, close
    Property:
        hooks4call
    """

    def __init__(self, cfg: 'EasyDict') -> None:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Init method.\n        Arguments:\n            - cfg (:obj:`EasyDict`): Config dict.\n        '
        BaseCommLearner.__init__(self, cfg)
        self._callback_fn = {'deal_with_resource': self.deal_with_resource, 'deal_with_learner_start': self.deal_with_learner_start, 'deal_with_get_data': self.deal_with_get_data, 'deal_with_learner_learn': self.deal_with_learner_learn, 'deal_with_learner_close': self.deal_with_learner_close}
        (host, port) = (cfg.host, cfg.port)
        if isinstance(port, list):
            port = port[self._rank]
        elif isinstance(port, int) and self._world_size > 1:
            port = port + self._rank
        self._slave = LearnerSlave(host, port, callback_fn=self._callback_fn)
        self._path_data = cfg.path_data
        self._path_policy = cfg.path_policy
        self._data_demand_queue = Queue(maxsize=1)
        self._data_result_queue = Queue(maxsize=1)
        self._learn_info_queue = Queue(maxsize=1)
        self._learner = None
        self._policy_id = None

    def start(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Start comm learner itself and the learner slave.\n        '
        BaseCommLearner.start(self)
        self._slave.start()

    def close(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Join learner thread and close learner if still running.\n            Then close learner slave and comm learner itself.\n        '
        if self._end_flag:
            return
        if self._learner is not None:
            self.deal_with_learner_close()
        self._slave.close()
        BaseCommLearner.close(self)

    def __del__(self) -> None:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Call ``close`` for deletion.\n        '
        self.close()

    def deal_with_resource(self) -> dict:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Callback function. Return how many resources are needed to start current learner.\n        Returns:\n            - resource (:obj:`dict`): Resource info dict, including ["gpu"].\n        '
        return {'gpu': self._world_size}

    def deal_with_learner_start(self, task_info: dict) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Overview:\n            Callback function. Create a learner and help register its hooks. Start a learner thread of the created one.\n        Arguments:\n            - task_info (:obj:`dict`): Task info dict.\n\n        .. note::\n            In ``_create_learner`` method in base class ``BaseCommLearner``, 3 methods\n            ('get_data', 'send_policy', 'send_learn_info'), dataloader and policy are set.\n            You can refer to it for details.\n        "
        self._policy_id = task_info['policy_id']
        self._league_save_checkpoint_path = task_info.get('league_save_checkpoint_path', None)
        self._learner = self._create_learner(task_info)
        for h in self.hooks4call:
            self._learner.register_hook(h)
        self._learner_thread = Thread(target=self._learner.start, args=(), daemon=True, name='learner_start')
        self._learner_thread.start()

    def deal_with_get_data(self) -> Any:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Callback function. Get data demand info dict from ``_data_demand_queue``,\n            which will be sent to coordinator afterwards.\n        Returns:\n            - data_demand (:obj:`Any`): Data demand info dict.\n        '
        data_demand = self._data_demand_queue.get()
        return data_demand

    def deal_with_learner_learn(self, data: dict) -> dict:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Callback function. Put training data info dict (i.e. meta data), which is received from coordinator, into\n            ``_data_result_queue``, and wait for ``get_data`` to retrieve. Wait for learner training and\n            get learn info dict from ``_learn_info_queue``. If task is finished, join the learner thread and\n            close the learner.\n        Returns:\n            - learn_info (:obj:`Any`): Learn info dict.\n        '
        self._data_result_queue.put(data)
        learn_info = self._learn_info_queue.get()
        return learn_info

    def deal_with_learner_close(self) -> None:
        if False:
            while True:
                i = 10
        self._learner.close()
        self._learner_thread.join()
        del self._learner_thread
        self._learner = None
        self._policy_id = None

    def send_policy(self, state_dict: dict) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Overview:\n            Save learner's policy in corresponding path, called by ``SendPolicyHook``.\n        Arguments:\n            - state_dict (:obj:`dict`): State dict of the policy.\n        "
        if not os.path.exists(self._path_policy):
            os.mkdir(self._path_policy)
        path = self._policy_id
        if self._path_policy not in path:
            path = os.path.join(self._path_policy, path)
        setattr(self, '_latest_policy_path', path)
        save_file(path, state_dict, use_lock=True)
        if self._league_save_checkpoint_path is not None:
            save_file(self._league_save_checkpoint_path, state_dict, use_lock=True)

    @staticmethod
    def load_data_fn(path, meta: Dict[str, Any], decompressor: Callable) -> Any:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            The function that is used to load data file.\n        Arguments:\n            - meta (:obj:`Dict[str, Any]`): Meta data info dict.\n            - decompressor (:obj:`Callable`): Decompress function.\n        Returns:\n            - s (:obj:`Any`): Data which is read from file.\n        '
        while True:
            try:
                s = read_from_di_store(path) if read_from_di_store else read_file(path, use_lock=False)
                s = decompressor(s)
                break
            except Exception:
                time.sleep(0.01)
        unroll_len = meta.get('unroll_len', 1)
        if 'unroll_split_begin' in meta:
            begin = meta['unroll_split_begin']
            if unroll_len == 1:
                s = s[begin]
                s.update(meta)
            else:
                end = begin + unroll_len
                s = s[begin:end]
                for i in range(len(s)):
                    s[i].update(meta)
        else:
            s.update(meta)
        return s

    def get_data(self, batch_size: int) -> List[Callable]:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Get a list of data loading function, which can be implemented by dataloader to read data from files.\n        Arguments:\n            - batch_size (:obj:`int`): Batch size.\n        Returns:\n            - data (:obj:`List[Callable]`): A list of callable data loading function.\n        '
        while self._learner is None:
            time.sleep(1)
        assert self._data_demand_queue.qsize() == 0
        self._data_demand_queue.put({'batch_size': batch_size, 'cur_learner_iter': self._learner.last_iter.val})
        data = self._data_result_queue.get()
        assert isinstance(data, list)
        assert len(data) == batch_size, '{}/{}'.format(len(data), batch_size)
        decompressor = get_data_decompressor(data[0].get('compressor', 'none'))
        data = [partial(FlaskFileSystemLearner.load_data_fn, path=m['object_ref'] if read_from_di_store else os.path.join(self._path_data, m['data_id']), meta=m, decompressor=decompressor) for m in data]
        return data

    def send_learn_info(self, learn_info: dict) -> None:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Store learn info dict in queue, which will be retrieved by callback function "deal_with_learner_learn"\n            in learner slave, then will be sent to coordinator.\n        Arguments:\n            - learn_info (:obj:`dict`): Learn info in `dict` type. Keys are like \'learner_step\', \'priority_info\'                 \'finished_task\', etc. You can refer to ``learn_info``(``worker/learner/base_learner.py``) for details.\n        '
        assert self._learn_info_queue.qsize() == 0
        self._learn_info_queue.put(learn_info)

    @property
    def hooks4call(self) -> List[LearnerHook]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Return the hooks that are related to message passing with coordinator.\n        Returns:\n            - hooks (:obj:`list`): The hooks which comm learner has. Will be registered in learner as well.\n        '
        return [SendPolicyHook('send_policy', 100, position='before_run', ext_args={}), SendPolicyHook('send_policy', 100, position='after_iter', ext_args={'send_policy_freq': 1}), SendLearnInfoHook('send_learn_info', 100, position='after_iter', ext_args={'freq': 10}), SendLearnInfoHook('send_learn_info', 100, position='after_run', ext_args={'freq': 1})]

class SendPolicyHook(LearnerHook):
    """
    Overview:
        Hook to send policy
    Interfaces:
        __init__, __call__
    Property:
        name, priority, position
    """

    def __init__(self, *args, ext_args: dict={}, **kwargs) -> None:
        if False:
            print('Hello World!')
        '\n        Overview:\n            init SendpolicyHook\n        Arguments:\n            - ext_args (:obj:`dict`): Extended arguments. Use ``ext_args.freq`` to set send_policy_freq\n        '
        super().__init__(*args, **kwargs)
        if 'send_policy_freq' in ext_args:
            self._freq = ext_args['send_policy_freq']
        else:
            self._freq = 1

    def __call__(self, engine: 'BaseLearner') -> None:
        if False:
            i = 10
            return i + 15
        "\n        Overview:\n            Save learner's policy in corresponding path at interval iterations by calling ``engine``'s ``send_policy``.\n            Saved file includes model_state_dict, learner_last_iter.\n        Arguments:\n            - engine (:obj:`BaseLearner`): The BaseLearner.\n\n        .. note::\n            Only rank == 0 learner will save policy.\n        "
        last_iter = engine.last_iter.val
        if engine.rank == 0 and last_iter % self._freq == 0:
            state_dict = {'model': engine.policy.state_dict()['model'], 'iter': last_iter}
            engine.send_policy(state_dict)
            engine.debug('{} save iter{} policy'.format(engine.instance_name, last_iter))

class SendLearnInfoHook(LearnerHook):
    """
    Overview:
        Hook to send learn info
    Interfaces:
        __init__, __call__
    Property:
        name, priority, position
    """

    def __init__(self, *args, ext_args: dict, **kwargs) -> None:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            init SendLearnInfoHook\n        Arguments:\n            - ext_args (:obj:`dict`): extended_args, use ext_args.freq\n        '
        super().__init__(*args, **kwargs)
        self._freq = ext_args['freq']

    def __call__(self, engine: 'BaseLearner') -> None:
        if False:
            return 10
        '\n        Overview:\n            Send learn info including last_iter at interval iterations and priority info\n        Arguments:\n            - engine (:obj:`BaseLearner`): the BaseLearner\n        '
        last_iter = engine.last_iter.val
        engine.send_learn_info(engine.learn_info)
        if last_iter % self._freq == 0:
            engine.debug('{} save iter{} learn_info'.format(engine.instance_name, last_iter))