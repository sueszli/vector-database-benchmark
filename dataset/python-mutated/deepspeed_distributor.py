import json
import sys
import tempfile
from typing import Union, Callable, List, Dict, Optional, Any
from pyspark.ml.torch.distributor import TorchDistributor

class DeepspeedTorchDistributor(TorchDistributor):
    _DEEPSPEED_SSL_CONF = 'deepspeed.spark.distributor.ignoreSsl'

    def __init__(self, numGpus: int=1, nnodes: int=1, localMode: bool=True, useGpu: bool=True, deepspeedConfig: Optional[Union[str, Dict[str, Any]]]=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        This class is used to run deepspeed training workloads with spark clusters.\n        The user has the option to specify the number of gpus per node\n        and the number of nodes (the same as if running from terminal),\n        as well as specify a deepspeed configuration file.\n\n        Parameters\n        ----------\n        numGpus: int\n            The number of GPUs to use per node (analagous to num_gpus in deepspeed command).\n        nnodes: int\n            The number of nodes that should be used for the run.\n        localMode: bool\n            Whether or not to run the training in a distributed fashion or just locally.\n        useGpu: bool\n            Boolean flag to determine whether to utilize gpus.\n        deepspeedConfig: Union[Dict[str,Any], str] or None:\n            The configuration file to be used for launching the deepspeed application.\n            If it\'s a dictionary containing the parameters, then we will create the file.\n            If None, deepspeed will fall back to default parameters.\n\n        Examples\n        --------\n        Run Deepspeed training function on a single node\n\n        >>> def train(learning_rate):\n        ...     import deepspeed\n        ...     # rest of training function\n        ...     return model\n        >>> distributor = DeepspeedTorchDistributor(\n        ...     numGpus=4,\n        ...     nnodes=1,\n        ...     useGpu=True,\n        ...     localMode=True,\n        ...     deepspeedConfig="path/to/config.json")\n        >>> output = distributor.run(train, 0.01)\n\n        Run Deepspeed training function on multiple nodes\n\n        >>> distributor = DeepspeedTorchDistributor(\n        ...     numGpus=4,\n        ...     nnodes=3,\n        ...     useGpu=True,\n        ...     localMode=False,\n        ...     deepspeedConfig="path/to/config.json")\n        >>> output = distributor.run(train, 0.01)\n        '
        num_processes = numGpus * nnodes
        self.deepspeed_config = deepspeedConfig
        super().__init__(num_processes, localMode, useGpu, _ssl_conf=DeepspeedTorchDistributor._DEEPSPEED_SSL_CONF)
        self.cleanup_deepspeed_conf = False

    @staticmethod
    def _get_deepspeed_config_path(deepspeed_config: Union[str, Dict[str, Any]]) -> str:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(deepspeed_config, dict):
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as file:
                json.dump(deepspeed_config, file)
                return file.name
        deepspeed_config_path = deepspeed_config
        if deepspeed_config is None:
            return ''
        return deepspeed_config_path

    @staticmethod
    def _create_torchrun_command(input_params: Dict[str, Any], train_path: str, *args: Any) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        local_mode = input_params['local_mode']
        num_processes = input_params['num_processes']
        deepspeed_config = input_params['deepspeed_config']
        deepspeed_config_path = DeepspeedTorchDistributor._get_deepspeed_config_path(deepspeed_config)
        (torchrun_args, processes_per_node) = TorchDistributor._get_torchrun_args(local_mode, num_processes)
        args_string = list(map(str, args))
        command_to_run = [sys.executable, '-m', 'torch.distributed.run', *torchrun_args, f'--nproc_per_node={processes_per_node}', train_path, *args_string, '--deepspeed']
        if deepspeed_config_path == '':
            return command_to_run
        return command_to_run + ['--deepspeed_config', deepspeed_config_path]

    @staticmethod
    def _run_training_on_pytorch_file(input_params: Dict[str, Any], train_path: str, *args: Any, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        if kwargs:
            raise ValueError("DeepspeedTorchDistributor with pytorch file doesn't support keyword arguments")
        log_streaming_client = input_params.get('log_streaming_client', None)
        training_command = DeepspeedTorchDistributor._create_torchrun_command(input_params, train_path, *args)
        DeepspeedTorchDistributor._execute_command(training_command, log_streaming_client=log_streaming_client)

    def run(self, train_object: Union[Callable, str], *args: Any, **kwargs: Any) -> Optional[Any]:
        if False:
            print('Hello World!')
        return self._run(train_object, DeepspeedTorchDistributor._run_training_on_pytorch_file, *args, **kwargs)