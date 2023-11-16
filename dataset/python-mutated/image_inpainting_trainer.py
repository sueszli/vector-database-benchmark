import time
from collections.abc import Mapping
from torch import distributed as dist
from modelscope.metainfo import Trainers
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.trainer import EpochBasedTrainer
from modelscope.utils.constant import DEFAULT_MODEL_REVISION, ConfigFields, ConfigKeys, Hubs, ModeKeys, ModelFile, Tasks, TrainerStages
from modelscope.utils.data_utils import to_device
from modelscope.utils.file_utils import func_receive_dict_inputs

@TRAINERS.register_module(module_name=Trainers.image_inpainting)
class ImageInpaintingTrainer(EpochBasedTrainer):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)

    def train(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().train(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        if False:
            print('Hello World!')
        metric_values = super().evaluate(*args, **kwargs)
        return metric_values

    def prediction_step(self, model, inputs):
        if False:
            print('Hello World!')
        pass

    def train_loop(self, data_loader):
        if False:
            print('Hello World!')
        ' Training loop used by `EpochBasedTrainer.train()`\n        '
        self.invoke_hook(TrainerStages.before_run)
        self._epoch = 0
        self.model.train()
        for _ in range(self._epoch, self._max_epochs):
            self.invoke_hook(TrainerStages.before_train_epoch)
            for (i, data_batch) in enumerate(data_loader):
                data_batch = to_device(data_batch, self.device)
                self.data_batch = data_batch
                self._inner_iter = i
                for idx in range(2):
                    self.invoke_hook(TrainerStages.before_train_iter)
                    self.train_step(self.model, data_batch, idx)
                    self.invoke_hook(TrainerStages.after_train_iter)
                del self.data_batch
                self._iter += 1
                self._mode = ModeKeys.TRAIN
                if i + 1 >= self.iters_per_epoch:
                    break
            self.invoke_hook(TrainerStages.after_train_epoch)
            self._epoch += 1
        self.invoke_hook(TrainerStages.after_run)

    def train_step(self, model, inputs, idx):
        if False:
            for i in range(10):
                print('nop')
        " Perform a training step on a batch of inputs.\n\n        Subclass and override to inject custom behavior.\n\n        Args:\n            model (`TorchModel`): The model to train.\n            inputs (`Dict[str, Union[torch.Tensor, Any]]`):\n                The inputs and targets of the model.\n\n                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the\n                argument `labels`. Check your model's documentation for all accepted arguments.\n\n        Return:\n            `torch.Tensor`: The tensor with training loss on this batch.\n        "
        model.train()
        self._mode = ModeKeys.TRAIN
        if isinstance(inputs, Mapping) and (not func_receive_dict_inputs(model.forward)):
            train_outputs = model.model._do_step(**inputs, optimizer_idx=idx)
        else:
            train_outputs = model.model._do_step(inputs, optimizer_idx=idx)
        if not isinstance(train_outputs, dict):
            raise TypeError('"model.forward()" must return a dict')
        if 'log_vars' not in train_outputs:
            default_keys_pattern = ['loss']
            match_keys = set([])
            for key_p in default_keys_pattern:
                match_keys.update([key for key in train_outputs.keys() if key_p in key])
            log_vars = {}
            for key in match_keys:
                value = train_outputs.get(key, None)
                if value is not None:
                    if dist.is_available() and dist.is_initialized():
                        value = value.data.clone()
                        dist.all_reduce(value.div_(dist.get_world_size()))
                    log_vars.update({key: value.item()})
            self.log_buffer.update(log_vars)
        else:
            self.log_buffer.update(train_outputs['log_vars'])
        self.train_outputs = train_outputs