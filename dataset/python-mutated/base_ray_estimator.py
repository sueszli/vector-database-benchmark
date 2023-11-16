from bigdl.orca.learn.ray_estimator import BaseEstimator
from bigdl.orca.ray import OrcaRayContext
from bigdl.orca.data.file import enable_multi_fs_save, enable_multi_fs_load
import io
import torch
import ray
from ray.exceptions import RayActorError
from abc import abstractmethod, ABCMeta
from bigdl.orca.learn.pytorch.utils import find_free_port, check_for_failure
from bigdl.orca.learn.utils import get_driver_node_ip
from bigdl.dllib.utils.log4Error import invalidInputError, logging
logger = logging.getLogger(__name__)
from typing import Dict

class BaseRayEstimator(BaseEstimator, metaclass=ABCMeta):

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def fit(self, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Train the model with train data.\n\n        :return: predicted result.\n        '
        pass

    @abstractmethod
    def predict(self, **kwargs):
        if False:
            return 10
        '\n        Predict input data.\n\n        :return: predicted result.\n        '
        pass

    @abstractmethod
    def evaluate(self, **kwargs):
        if False:
            return 10
        "\n        Evaluate model.\n\n        :return: evaluation result as a dictionary of {'metric name': metric value}\n        "
        pass

    @abstractmethod
    def get_model(self):
        if False:
            print('Hello World!')
        '\n        Get the trained model.\n\n        :return: Trained model\n        '
        pass

    def setup(self, params, backend='ray', runner_cls=None, workers_per_node=1):
        if False:
            i = 10
            return i + 15
        ray_ctx = OrcaRayContext.get()
        if backend == 'ray':
            self.init_ddp_process = False
            self.cores_per_node = ray_ctx.ray_node_cpu_cores // workers_per_node
            self.num_nodes = ray_ctx.num_ray_nodes * workers_per_node
            RemoteRunner = ray.remote(num_cpus=self.cores_per_node)(runner_cls)
            self.remote_workers = [RemoteRunner.remote(**params) for i in range(self.num_nodes)]
            ray.get([worker.setup.remote(self.cores_per_node) for (i, worker) in enumerate(self.remote_workers)])
            ray.get([worker.setup_torch_estimator.remote(i, self.num_nodes) for (i, worker) in enumerate(self.remote_workers)])
        elif backend == 'horovod':
            from bigdl.orca.learn.horovod.horovod_ray_runner import HorovodRayRunner
            self.horovod_runner = HorovodRayRunner(ray_ctx, worker_cls=runner_cls, worker_param=params, workers_per_node=workers_per_node)
            self.remote_workers = self.horovod_runner.remote_workers
            cores_per_node = self.horovod_runner.cores_per_node
            ray.get([worker.setup.remote(cores_per_node) for (i, worker) in enumerate(self.remote_workers)])
            ray.get([worker.setup_horovod.remote() for (i, worker) in enumerate(self.remote_workers)])
        else:
            invalidInputError(False, 'Only "ray" and "horovod" are supported values of backend, but got {}'.format(backend))
        self.num_workers = len(self.remote_workers)

    def setup_torch_ddp(self):
        if False:
            i = 10
            return i + 15
        import torch.distributed as dist
        driver_ip = get_driver_node_ip()
        driver_tcp_store_port = find_free_port()
        _ = dist.TCPStore(driver_ip, driver_tcp_store_port, -1, True, dist.constants.default_pg_timeout)
        ray.get([worker.setup_torch_distribute.remote(driver_ip, driver_tcp_store_port, i, self.num_nodes) for (i, worker) in enumerate(self.remote_workers)])
        self.init_ddp_process = True

    def get_state_dict(self) -> Dict:
        if False:
            while True:
                i = 10
        stream_ids = [worker.get_state_stream.remote() for worker in self.remote_workers]
        ([stream_id], stream_ids) = ray.wait(stream_ids, num_returns=1, timeout=None)
        byte_obj = ray.get(stream_id)
        _buffer = io.BytesIO(byte_obj)
        state_dict = torch.load(_buffer, map_location='cpu')
        return state_dict

    def load_state_dict(self, state_dict: Dict, blocking: bool=True):
        if False:
            print('Hello World!')
        _buffer = io.BytesIO()
        torch.save(state_dict, _buffer)
        state_stream = _buffer.getvalue()
        state_id = ray.put(state_stream)
        remote_calls = [worker.load_state_stream.remote(state_id) for worker in self.remote_workers]
        if blocking:
            ray.get(remote_calls)

    @enable_multi_fs_save
    def save(self, model_path: str) -> str:
        if False:
            while True:
                i = 10
        '\n        Saves the Estimator state (including model and optimizer) to the provided model_path.\n\n        :param model_path: (str) Path to save the model.\n        :return:\n        '
        state_dict = self.get_state_dict()
        torch.save(state_dict, model_path)
        return model_path

    @enable_multi_fs_load
    def load(self, model_path: str):
        if False:
            while True:
                i = 10
        '\n        Loads the Estimator state (including model and optimizer) from the provided model_path.\n\n        :param model_path: (str) Path to the existing model.\n        '
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict)

    def save_checkpoint(self, model_path: str):
        if False:
            i = 10
            return i + 15
        '\n        Manually saves the Estimator state (including model and optimizer) to the provided\n        model_path.\n\n        :param model_path: (str) Path to save the model. Both local and remote path are supported.\n               e.g. "/tmp/estimator.ckpt" or "hdfs:///tmp/estimator.ckpt"\n        :return: None\n        '
        from bigdl.dllib.utils.file_utils import is_local_path
        if is_local_path(model_path):
            self.save(model_path)
        else:
            results = [worker.save_checkpoint.remote(model_path) for worker in self.remote_workers]
            ray.get(results)

    def load_checkpoint(self, model_path: str):
        if False:
            return 10
        '\n        Loads the Estimator state (including model and optimizer) from the provided model_path.\n\n        :param model_path: (str) Path to the existing model. Both local and remote path are\n               supported. e.g. "/tmp/estimator.ckpt" or "hdfs:///tmp/estimator.ckpt"\n        :return: None\n        '
        from bigdl.dllib.utils.file_utils import is_local_path
        if is_local_path(model_path):
            self.load(model_path)
        else:
            results = [worker.load_checkpoint.remote(model_path) for worker in self.remote_workers]
            ray.get(results)

    def shutdown(self, force: bool=False):
        if False:
            print('Hello World!')
        '\n        Shuts down workers and releases resources.\n\n        :return:\n        '
        if not force:
            cleanup = [worker.shutdown.remote() for worker in self.remote_workers]
            try:
                ray.get(cleanup)
                [worker.__ray_terminate__.remote() for worker in self.remote_workers]
            except RayActorError:
                logger.warning('Failed to shutdown gracefully, forcing a shutdown.')
                for worker in self.remote_workers:
                    logger.warning('Killing worker {}.'.format(worker))
                    ray.kill(worker)
        else:
            for worker in self.remote_workers:
                logger.debug('Killing worker {}.'.format(worker))
                ray.kill(worker)
        self.remote_workers = []

    def _train_epochs(self, **params):
        if False:
            return 10
        remote_worker_stats = []
        for (i, w) in enumerate(self.remote_workers):
            stats = w.train_epochs.remote(**params)
            remote_worker_stats.append(stats)
        success = check_for_failure(remote_worker_stats)
        if success:
            return (success, ray.get(remote_worker_stats))
        else:
            return (success, None)