import copy
from logging import warning
from typing import Any, List, Optional, Union
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from torchmetrics.metric import Metric
from torch.optim.lr_scheduler import _LRScheduler
from bigdl.nano.pytorch import InferenceOptimizer
from bigdl.nano.utils.pytorch import TORCH_VERSION_LESS_1_11, check_ccl
from bigdl.nano.utils.pytorch import ChannelsLastCallback
from bigdl.nano.utils.pytorch import save_model, load_model
from bigdl.nano.pytorch.algorithms import SelectiveBackprop
from bigdl.nano.pytorch.lightning import LightningModule
from bigdl.nano.pytorch.strategies import IPEXStrategy, DDPSpawnStrategy, DDPSubprocessStrategy, DDPK8sStrategy
from bigdl.nano.deps.automl.hpo_api import create_hpo_searcher, check_hpo_status
from bigdl.nano.deps.ray.ray_api import create_ray_strategy
from bigdl.nano.utils.common import invalidInputError
from bigdl.nano.utils.common import _avx512_checker
from bigdl.nano.utils.common import deprecated
distributed_backends = ['spawn', 'ray', 'subprocess', 'k8s']
backends_class_map = {'spawn': DDPSpawnStrategy, 'subprocess': DDPSubprocessStrategy, 'ray': create_ray_strategy, 'k8s': DDPK8sStrategy}

class Trainer(pl.Trainer):
    """
    Trainer for BigDL-Nano pytorch.

    This Trainer extends PyTorch Lightning Trainer by adding
    various options to accelerate pytorch training.
    """

    def __init__(self, num_processes: Optional[int]=None, use_ipex: bool=False, distributed_backend='subprocess', process_group_backend: Optional[str]=None, cpu_for_each_process: Optional[List[List[int]]]=None, use_hpo=False, channels_last: bool=False, auto_lr: Union[dict, bool]=True, precision: Union[str, int]=32, *args: Any, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        "\n        A pytorch lightning trainer that uses bigdl-nano optimization.\n\n        :param num_processes: number of processes in distributed training. default: ``1``.\n        :param use_ipex: whether we use ipex as accelerator for trainer. default: ``False``.\n        :param distributed_backend: use which backend in distributed mode, defaults to\n            ``'subprocess'``, now avaiable backends are ``'spawn'``, ``'subprocess'`` and ``'ray'``\n        :param process_group_backend: use which process group backend in distributed mode, defaults\n            to ``None``, means using ``'gloo'`` with CPU, while using ``'nccl'`` with GPU, now\n            avaiable backends are ``None`` and ``'ccl'``.\n        :param cpu_for_each_process: A list of length ``num_processes``, each containing a list of\n            indices of cpus each process will be using. default: ``None``, and the cpu will be\n            automatically and evenly distributed among processes.\n        :param channels_last: whether convert input to channels last memory formats,\n            defaults to ``False``.\n        :param auto_lr: whether to scale the learning rate linearly by ``num_processes`` times.\n            Defaults to ``True``.\n            A dict with ``warmup_epochs`` as key is also accepted to control the number of epochs\n            needed for the learning rate to be scaled by ``num_processes`` times.\n            If ``auto_lr=Ture``, ``warmup_epochs`` will by default be ``max_epochs // 10``.\n            If ``num_processes=1`` or other ``lr_scheduler`` is set, ``auto_lr`` will be ignored.\n        :param precision: Double precision (``64``), full precision (``32``),\n            half precision (``16``) or bfloat16 precision (``'bf16'``), defaults to ``32``.\n            Enable ipex bfloat16 weight prepack when ``use_ipex=True`` and ``precision='bf16'``\n        "
        if 'accelerator' in kwargs:
            warning(f"Accelerator will be specified by bigdl-nano,\n            accelerator entered {kwargs['accelerator']} will be ignored. ")
            kwargs.pop('accelerator')
        if 'plugins' in kwargs:
            warning(f"Plugins will be specified by bigdl-nano,\n             plugines entered {kwargs['plugins']} will be ignored. ")
            kwargs.pop('plugins')
        if cpu_for_each_process is not None:
            if len(cpu_for_each_process) != num_processes:
                invalidInputError(False, f'The length of `cpu_for_each_process` ({len(cpu_for_each_process)}) is not equal to the number of processes {num_processes}.')
        if 'algorithms' in kwargs:
            kwargs = self._add_algorithms(kwargs)
        if channels_last:
            callbacks = kwargs.get('callbacks')
            if callbacks:
                callbacks.append(ChannelsLastCallback())
            else:
                kwargs['callbacks'] = [ChannelsLastCallback()]
        self.use_ipex = use_ipex
        dtype = None
        if self.use_ipex and precision == 'bf16':
            dtype = torch.bfloat16
            precision = 32
        if self.use_ipex and (not _avx512_checker()):
            if TORCH_VERSION_LESS_1_11:
                warning('Enable ipex<=1.11 in a cpu instruction set without avx512 will crash.Fall back to regular pytorch.')
                self.use_ipex = False
            elif dtype == torch.bfloat16:
                warning('Enable IPEX bfloat16 in a cpu instruction set without avx512 will crash. Using 32-bit precision')
                dtype = None
        kwargs['precision'] = precision
        if num_processes is None and distributed_backend != 'k8s':
            num_processes = 1
        if num_processes == 1:
            kwargs['strategy'] = IPEXStrategy(dtype=dtype) if self.use_ipex else None
            super().__init__(*args, **kwargs)
        else:
            invalidInputError(distributed_backend in distributed_backends, f'Distributed backends supported now are {distributed_backends}, but get {distributed_backend}.')
            check_ccl(process_group_backend)
            if 'checkpoint_callback' in kwargs:
                if not kwargs['checkpoint_callback']:
                    invalidInputError(False, f'`checkpoint_callback` set to False. Currently, disable checkpoint callback make distributed training backend work incorrect')
            strategy_cls = backends_class_map[distributed_backend]
            strategy = strategy_cls(num_processes=num_processes, cpu_for_each_process=cpu_for_each_process, use_ipex=self.use_ipex, dtype=dtype, auto_lr=auto_lr, process_group_backend=process_group_backend)
            kwargs['strategy'] = strategy
            super().__init__(*args, **kwargs)
        if use_hpo:
            self.hposearcher = create_hpo_searcher(trainer=self, num_processes=num_processes)
        else:
            self.hposearcher = None

    def _add_algorithms(self, kwargs):
        if False:
            for i in range(10):
                print('nop')
        callbacks = kwargs.get('callbacks')
        for algorithm in kwargs['algorithms']:
            if isinstance(algorithm, SelectiveBackprop):
                if callbacks:
                    callbacks.append(algorithm)
                else:
                    kwargs['callbacks'] = [algorithm]
        del kwargs['algorithms']
        return kwargs

    @staticmethod
    def compile(model: nn.Module, loss: _Loss=None, optimizer: torch.optim.Optimizer=None, scheduler: _LRScheduler=None, metrics: List[Metric]=None):
        if False:
            i = 10
            return i + 15
        '\n        Construct a pytorch-lightning model.\n\n        If model is already a pytorch-lightning model,\n        return model. If model is pytorch model, construct a new pytorch-lightning module\n        with model, loss and optimizer.\n\n        :param model:       A model instance.\n        :param loss:        Loss to construct pytorch-lightning model.\n                            Should be None if model is instance of pl.LightningModule.\n        :param optimizer:   Optimizer to construct pytorch-lightning model Should be None.\n                            if model is instance of pl.LightningModule.\n        :param metrics:     A list of torchmetrics to validate/test performance.\n        :return:            A LightningModule object.\n        '
        invalidInputError(isinstance(model, nn.Module), 'Model must be instance of nn.Module but got {}'.format(model.__class__))
        pl_model = None
        if isinstance(model, pl.LightningModule):
            invalidInputError(not (loss or optimizer), 'Loss and optimizer should be None if model is a pytorch-lightning model.')
            pl_model = model
        else:
            pl_model = LightningModule(model, loss, optimizer, scheduler, metrics)
        return pl_model

    def search(self, model, resume: bool=False, target_metric=None, mode: str='best', n_parallels=1, acceleration=False, input_sample=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Run HPO search. It will be called in Trainer.search().\n\n        :param model: The model to be searched. It should be an auto model.\n        :param resume: whether to resume the previous or start a new one,\n            defaults to False.\n        :param target_metric: the object metric to optimize,\n            defaults to None.\n        :param mode: use last epoch\'s result as trial\'s score or use best epoch\'s.\n            defaults to \'best\', you can change it to \'last\'.\n        :param n_parallels: the number of parallel processes for running trials.\n        :param acceleration: Whether to automatically consider the model after\n            inference acceleration in the search process. It will only take\n            effect if target_metric contains "latency". Default value is False.\n        :param input_sample: A set of inputs for trace, defaults to None if you have\n            trace before or model is a LightningModule with any dataloader attached.\n        :return: the model with study meta info attached.\n        '
        if not check_hpo_status(self.hposearcher):
            return None
        Trainer._log_api_event('search')
        return self.hposearcher.search(model, resume=resume, target_metric=target_metric, mode=mode, n_parallels=n_parallels, acceleration=acceleration, input_sample=input_sample, **kwargs)

    def search_summary(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Retrive a summary of trials.\n\n        :return: A summary of all the trials. Currently the entire study is\n            returned to allow more flexibility for further analysis and visualization.\n        '
        if not check_hpo_status(self.hposearcher):
            return None
        return self.hposearcher.search_summary()

    @staticmethod
    @deprecated(func_name='bigdl.nano.pytorch.Trainer.trace', message='Please use `bigdl.nano.pytorch.InferenceOptimizer.trace` instead.')
    def trace(model: nn.Module, input_sample=None, accelerator: str=None, use_ipex: bool=False, thread_num: int=None, onnxruntime_session_options=None, logging: bool=True, **export_kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Trace a pytorch model and convert it into an accelerated module for inference.\n\n        For example, this function returns a PytorchOpenVINOModel when accelerator=='openvino'.\n\n        :param model: An torch.nn.Module model, including pl.LightningModule.\n        :param input_sample: A set of inputs for trace, defaults to None if you have trace before or\n                             model is a LightningModule with any dataloader attached.\n        :param accelerator: The accelerator to use, defaults to None meaning staying in Pytorch\n                            backend. 'openvino', 'onnxruntime' and 'jit' are supported for now.\n        :param use_ipex: whether we use ipex as accelerator for inferencing. default: False.\n        :param thread_num: (optional) a int represents how many threads(cores) is needed for\n                           inference, only valid for accelerator='onnxruntime'\n                           or accelerator='openvino'.\n        :param onnxruntime_session_options: The session option for onnxruntime, only valid when\n                                            accelerator='onnxruntime', otherwise will be ignored.\n        :param logging: whether to log detailed information of model conversion, only valid when\n                        accelerator='openvino', otherwise will be ignored. default: True.\n        :param **kwargs: other extra advanced settings include\n                         1. those be passed to torch.onnx.export function, only valid when\n                         accelerator='onnxruntime'/'openvino', otherwise will be ignored.\n                         2. if channels_last is set and use_ipex=True, we will transform the\n                         data to be channels last according to the setting. Defaultly, channels_last\n                         will be set to True if use_ipex=True.\n        :return: Model with different acceleration.\n\n        .. warning::\n             ``bigdl.nano.pytorch.Trainer.trace`` will be deprecated in future release.\n\n             Please use ``bigdl.nano.pytorch.InferenceOptimizer.trace`` instead.\n        "
        return InferenceOptimizer.trace(model=model, input_sample=input_sample, accelerator=accelerator, use_ipex=use_ipex, thread_num=thread_num, onnxruntime_session_options=onnxruntime_session_options, logging=logging, **export_kwargs)

    @staticmethod
    @deprecated(func_name='bigdl.nano.pytorch.Trainer.quantize', message='Please use `bigdl.nano.pytorch.InferenceOptimizer.quantize` instead.')
    def quantize(model: nn.Module, precision: str='int8', accelerator: str=None, use_ipex: bool=False, calib_dataloader: DataLoader=None, metric: Metric=None, accuracy_criterion: dict=None, approach: str='static', method: str=None, conf: str=None, tuning_strategy: str=None, timeout: int=None, max_trials: int=None, input_sample=None, thread_num: int=None, onnxruntime_session_options=None, logging: bool=True, **export_kwargs):
        if False:
            return 10
        '\n        Calibrate a Pytorch-Lightning model for post-training quantization.\n\n        :param model:           A model to be quantized. Model type should be an instance of\n                                nn.Module.\n        :param precision:       Global precision of quantized model,\n                                supported type: \'int8\', \'bf16\', \'fp16\', defaults to \'int8\'.\n        :param accelerator:     Use accelerator \'None\', \'onnxruntime\', \'openvino\', defaults to None.\n                                None means staying in pytorch.\n        :param calib_dataloader:    A torch.utils.data.dataloader.DataLoader object for calibration.\n                                    Required for static quantization.\n                                    It\'s also used as validation dataloader.\n        :param metric:              A torchmetrics.metric.Metric object for evaluation.\n        :param accuracy_criterion:  Tolerable accuracy drop, defaults to None meaning no\n                                    accuracy control.\n                                    accuracy_criterion = {\'relative\': 0.1, \'higher_is_better\': True}\n                                    allows relative accuracy loss: 1%. accuracy_criterion =\n                                    {\'absolute\': 0.99, \'higher_is_better\':False} means accuracy\n                                    must be smaller than 0.99.\n        :param approach:    \'static\' or \'dynamic\'.\n                            \'static\': post_training_static_quant,\n                            \'dynamic\': post_training_dynamic_quant.\n                            Default: \'static\'. OpenVINO supports static mode only.\n        :param method:          Method to do quantization. When accelerator=None, supported\n            methods: \'fx\', \'eager\', \'ipex\', defaults to \'fx\'. If you don\'t use ipex, suggest using\n            \'fx\' which executes automatic optimizations like fusion. For more information, please\n            refer to https://pytorch.org/docs/stable/quantization.html#eager-mode-quantization.\n            When accelerator=\'onnxruntime\', supported methods: \'qlinear\', \'integer\', defaults\n            to \'qlinear\'. Suggest \'qlinear\' for lower accuracy drop if using static quantization.\n            More details in https://onnxruntime.ai/docs/performance/quantization.html.\n            This argument doesn\'t take effect for OpenVINO, don\'t change it for OpenVINO.\n        :param conf:        A path to conf yaml file for quantization.\n                            Default: None, using default config.\n        :param tuning_strategy:    \'bayesian\', \'basic\', \'mse\', \'sigopt\'. Default: \'bayesian\'.\n        :param timeout:     Tuning timeout (seconds). Default: None,  which means early stop.\n                            Combine with max_trials field to decide when to exit.\n        :param max_trials:  Max tune times. Default: None, which means no tuning.\n                            Combine with timeout field to decide when to exit.\n                            "timeout=0, max_trials=1" means it will try quantization only once and\n                            return satisfying best model.\n        :param input_sample:      An input example to convert pytorch model into ONNX/OpenVINO.\n        :param thread_num: (optional) a int represents how many threads(cores) is needed for\n                           inference, only valid for accelerator=\'onnxruntime\'\n                           or accelerator=\'openvino\'.\n        :param onnxruntime_session_options: The session option for onnxruntime, only valid when\n                                            accelerator=\'onnxruntime\', otherwise will be ignored.\n        :param logging: whether to log detailed information of model conversion, only valid when\n                        accelerator=\'openvino\', otherwise will be ignored. default: True.\n        :param **export_kwargs: will be passed to torch.onnx.export function.\n        :return:            A accelerated Pytorch-Lightning Model if quantization is sucessful.\n\n        .. warning::\n             ``bigdl.nano.pytorch.Trainer.quantize`` will be deprecated in future release.\n\n             Please use ``bigdl.nano.pytorch.InferenceOptimizer.quantize`` instead.\n        '
        return InferenceOptimizer.quantize(model=model, precision=precision, accelerator=accelerator, use_ipex=use_ipex, calib_dataloader=calib_dataloader, metric=metric, accuracy_criterion=accuracy_criterion, approach=approach, method=method, conf=conf, tuning_strategy=tuning_strategy, timeout=timeout, max_trials=max_trials, input_sample=input_sample, thread_num=thread_num, onnxruntime_session_options=onnxruntime_session_options, logging=logging, **export_kwargs)

    @staticmethod
    def save(model: nn.Module, path):
        if False:
            while True:
                i = 10
        '\n        Save the model to local file.\n\n        :param model: Any model of torch.nn.Module, including all models accelareted by\n               Trainer.trace/Trainer.quantize.\n        :param path: Path to saved model. Path should be a directory.\n        '
        save_model(model, path)

    @staticmethod
    def load(path, model: Optional[nn.Module]=None, input_sample=None, inplace=False, device=None):
        if False:
            i = 10
            return i + 15
        "\n        Load a model from local.\n\n        :param path: Path to model to be loaded. Path should be a directory.\n        :param model: Required FP32 model to load pytorch model, it is needed if:\n               1. you accelerate the model with accelerator=None by\n               InferenceOptimizer.trace()/InferenceOptimizer.quantize().\n               2. you accelerate the model with InferenceOptimizer.optimize() and\n               get_model()/get_best_model(), and the best method or the method you\n               specify don't contain accelerator 'onnxruntime'/'openvino'/'jit'.\n               If you are not sure what optimization method is used, we recommend that\n               you always pass in the original model for this case.\n               3. you want to the loaded model contains the attributes of original model.\n        :param input_sample: Input sample for your model, could be a Tensor or a tuple.\n               Only valid for inc ipex quantization model, otherwise will be ignored.\n        :param inplace: whether to perform inplace optimization. Default: ``False``.\n        :param device: A string represents the device of the inference. Default to None.\n               Only valid for openvino model, otherwise will be ignored.\n        :return: Model with different acceleration(None/OpenVINO/ONNX Runtime/JIT) or\n                 precision(FP32/FP16/BF16/INT8).\n        "
        return load_model(path, model, input_sample=input_sample, inplace=inplace, device=device)