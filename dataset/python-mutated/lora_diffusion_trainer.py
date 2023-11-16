from typing import Union
import torch
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from modelscope.metainfo import Trainers
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.hooks.checkpoint.checkpoint_hook import CheckpointHook
from modelscope.trainers.hooks.checkpoint.checkpoint_processor import CheckpointProcessor
from modelscope.trainers.optimizer.builder import build_optimizer
from modelscope.trainers.trainer import EpochBasedTrainer
from modelscope.utils.config import ConfigDict

class LoraDiffusionCheckpointProcessor(CheckpointProcessor):

    def __init__(self, torch_type=torch.float32, safe_serialization=False):
        if False:
            while True:
                i = 10
        'Checkpoint processor for lora diffusion.\n\n        Args:\n            torch_type: The torch type, default is float32.\n            safe_serialization: Whether to save the model using safetensors or the traditional PyTorch way with pickle.\n\n        '
        self.torch_type = torch_type
        self.safe_serialization = safe_serialization

    def save_checkpoints(self, trainer, checkpoint_path_prefix, output_dir, meta=None, save_optimizers=True):
        if False:
            i = 10
            return i + 15
        'Save the state dict for lora tune model.\n        '
        trainer.model.unet = trainer.model.unet.to(self.torch_type)
        trainer.model.unet.save_attn_procs(output_dir, safe_serialization=self.safe_serialization)

@TRAINERS.register_module(module_name=Trainers.lora_diffusion)
class LoraDiffusionTrainer(EpochBasedTrainer):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        'Lora trainers for fine-tuning stable diffusion\n\n        Args:\n            lora_rank: The rank size of lora intermediate linear.\n            torch_type: The torch type, default is float32.\n            safe_serialization: Whether to save the model using safetensors or the traditional PyTorch way with pickle.\n\n        '
        lora_rank = kwargs.pop('lora_rank', 4)
        torch_type = kwargs.pop('torch_type', torch.float32)
        safe_serialization = kwargs.pop('safe_serialization', False)
        ckpt_hook = list(filter(lambda hook: isinstance(hook, CheckpointHook), self.hooks))[0]
        ckpt_hook.set_processor(LoraDiffusionCheckpointProcessor(torch_type=torch_type, safe_serialization=safe_serialization))
        lora_attn_procs = {}
        for name in self.model.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith('attn1.processor') else self.model.unet.config.cross_attention_dim
            if name.startswith('mid_block'):
                hidden_size = self.model.unet.config.block_out_channels[-1]
            elif name.startswith('up_blocks'):
                block_id = int(name[len('up_blocks.')])
                hidden_size = list(reversed(self.model.unet.config.block_out_channels))[block_id]
            elif name.startswith('down_blocks'):
                block_id = int(name[len('down_blocks.')])
                hidden_size = self.model.unet.config.block_out_channels[block_id]
            lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)
        self.model.unet.set_attn_processor(lora_attn_procs)
        self.lora_layers = AttnProcsLayers(self.model.unet.attn_processors)

    def build_optimizer(self, cfg: ConfigDict, default_args: dict=None):
        if False:
            i = 10
            return i + 15
        try:
            return build_optimizer(self.lora_layers.parameters(), cfg=cfg, default_args=default_args)
        except KeyError as e:
            self.logger.error(f'Build optimizer error, the optimizer {cfg} is a torch native component, please check if your torch with version: {torch.__version__} matches the config.')
            raise e