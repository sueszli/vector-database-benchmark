import types
import copy
import ray
from bigdl.dllib.utils.log4Error import invalidInputError
from bigdl.orca.learn.pytorch.core.base_ray_estimator import BaseRayEstimator
from bigdl.orca.learn.pytorch.experimential.mmcv.mmcv_ray_runner import MMCVRayEpochRunner
from bigdl.orca.learn.pytorch.utils import process_stats
from typing import Dict, List, Optional, Tuple, Callable, Union

class MMCVRayEstimator(BaseRayEstimator):

    def __init__(self, *, mmcv_runner_creator: Callable, backend: str='ray', workers_per_node: int=1, config: Optional[Dict]=None) -> None:
        if False:
            print('Hello World!')
        if not isinstance(mmcv_runner_creator, types.FunctionType):
            invalidInputError(False, 'Must provide a function for mmcv_runner_creator')
        self.mmcv_runner_creator = mmcv_runner_creator
        self.backend = backend
        self.runner_cls = MMCVRayEpochRunner
        self.config = {} if config is None else config
        worker_config = copy.copy(self.config)
        params = dict(mmcv_runner_creator=self.mmcv_runner_creator, config=worker_config)
        self.setup(params, self.backend, self.runner_cls, workers_per_node)

    def fit(self, data_loaders_creators: List[Callable], workflow: List[Tuple[str, int]], max_epochs: Optional[int]=None, reduce_results: bool=True, **kwargs) -> List:
        if False:
            while True:
                i = 10
        "Trains a MMCV model given training and val data for several epochs.\n\n        :param data_loaders_creators: Dataloader creators for training and validation.\n        :param workflow: A list of (phase, epochs) to specify the\n               running order and epochs. E.g, [('train', 2), ('val', 1)] means\n               running 2 epochs for training and 1 epoch for validation,\n               iteratively.\n        :param max_epochs: Set max_epochs for MMCV runner is deprecated\n        :param reduce_results: Whether to average all metrics across all workers into\n               one dict. If a metric is a non-numerical value, the one value will be randomly\n               selected among the workers. If False, returns a list of dicts for\n               all workers. Default is True.\n        "
        for creator in data_loaders_creators:
            if not isinstance(creator, types.FunctionType):
                invalidInputError(False, 'Must provide a function for all dataloader creator')
        params = dict(data_loaders_creators=data_loaders_creators, workflow=workflow, max_epochs=max_epochs, **kwargs)
        self.setup_torch_ddp()
        (success, worker_stats) = self._train_epochs(**params)
        epoch_stats = list(map(list, zip(*worker_stats)))
        if reduce_results:
            for i in range(len(epoch_stats)):
                epoch_stats[i] = process_stats(epoch_stats[i])
            return epoch_stats
        else:
            return epoch_stats

    def run(self, data_loaders_creators: List[Callable], workflow: List[Tuple[str, int]], max_epochs: Optional[int]=None, reduce_results: bool=True, **kwargs) -> List:
        if False:
            i = 10
            return i + 15
        '\n        Same as fit method, the parameters are consistent with MMCV runner.run()\n        '
        return self.fit(data_loaders_creators, workflow, max_epochs, reduce_results, **kwargs)

    def predict(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        pass

    def evaluate(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        pass

    def get_model(self) -> Dict:
        if False:
            i = 10
            return i + 15
        state = self.get_state_dict()
        model_state = state['state_dict']
        return model_state

    def load_checkpoint(self, filename: str, map_location: Union[str, Callable]='cpu', strict: bool=False, revise_keys: List=[('^module.', '')]) -> None:
        if False:
            print('Hello World!')
        "Load checkpoint from a file or URI. The filename should either be a\n        local path on driver or a HDFS path\n\n        Args:\n            filename (str): Local path on Driver or HDFS path.\n            map_location (str): Same as :func:`torch.load`.\n            strict (bool): Whether to allow different params for the model and\n                checkpoint.\n            revise_keys (list): A list of customized keywords to modify the\n                state_dict in checkpoint. Each item is a (pattern, replacement)\n                pair of the regular expression operations. Default: strip\n                the prefix 'module.' by [(r'^module\\.', '')].\n\n        Returns:\n            None\n        "
        from bigdl.dllib.utils.file_utils import is_local_path
        if is_local_path(filename):
            self.load(filename)
        else:
            params = dict(filename=filename, map_location=map_location, strict=strict, revise_keys=revise_keys)
            results = [worker.load_checkpoint.remote(**params) for worker in self.remote_workers]
            ray.get(results)