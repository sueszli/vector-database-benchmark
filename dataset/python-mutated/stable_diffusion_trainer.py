import os
from typing import Union
import torch
from torch import nn
from modelscope.metainfo import Trainers
from modelscope.models.base import Model, TorchModel
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.hooks.checkpoint.checkpoint_hook import CheckpointHook
from modelscope.trainers.hooks.checkpoint.checkpoint_processor import CheckpointProcessor
from modelscope.trainers.optimizer.builder import build_optimizer
from modelscope.trainers.trainer import EpochBasedTrainer
from modelscope.utils.config import ConfigDict

class SwiftDiffusionCheckpointProcessor(CheckpointProcessor):

    def save_checkpoints(self, trainer, checkpoint_path_prefix, output_dir, meta=None, save_optimizers=True):
        if False:
            for i in range(10):
                print('nop')
        'Save the state dict for swift lora tune model.\n        '
        trainer.model.unet.save_pretrained(os.path.join(output_dir))

@TRAINERS.register_module(module_name=Trainers.stable_diffusion)
class StableDiffusionTrainer(EpochBasedTrainer):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Stable Diffusion trainers for fine-tuning.\n\n        Args:\n            use_swift: Whether to use swift.\n\n        '
        super().__init__(*args, **kwargs)
        use_swift = kwargs.pop('use_swift', False)
        if use_swift:
            ckpt_hook = list(filter(lambda hook: isinstance(hook, CheckpointHook), self.hooks))[0]
            ckpt_hook.set_processor(SwiftDiffusionCheckpointProcessor())

    def build_optimizer(self, cfg: ConfigDict, default_args: dict=None):
        if False:
            print('Hello World!')
        try:
            return build_optimizer(self.model.unet, cfg=cfg, default_args=default_args)
        except KeyError as e:
            self.logger.error(f'Build optimizer error, the optimizer {cfg} is a torch native component, please check if your torch with version: {torch.__version__} matches the config.')
            raise e