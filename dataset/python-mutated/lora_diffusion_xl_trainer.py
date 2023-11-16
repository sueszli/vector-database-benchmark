from typing import Dict, Union
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAAttnProcessor2_0
from modelscope.metainfo import Trainers
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.hooks.checkpoint.checkpoint_hook import CheckpointHook
from modelscope.trainers.hooks.checkpoint.checkpoint_processor import CheckpointProcessor
from modelscope.trainers.optimizer.builder import build_optimizer
from modelscope.trainers.trainer import EpochBasedTrainer
from modelscope.utils.config import ConfigDict

def unet_attn_processors_state_dict(unet) -> Dict[str, torch.tensor]:
    if False:
        i = 10
        return i + 15
    '\n    Returns:\n        a state dict containing just the attention processor parameters.\n    '
    attn_processors = unet.attn_processors
    attn_processors_state_dict = {}
    for (attn_processor_key, attn_processor) in attn_processors.items():
        for (parameter_key, parameter) in attn_processor.state_dict().items():
            attn_processors_state_dict[f'{attn_processor_key}.{parameter_key}'] = parameter
    return attn_processors_state_dict

class LoraDiffusionXLCheckpointProcessor(CheckpointProcessor):

    def __init__(self, safe_serialization=False):
        if False:
            i = 10
            return i + 15
        'Checkpoint processor for lora diffusion.\n\n        Args:\n            safe_serialization: Whether to save the model using safetensors or the traditional PyTorch way with pickle.\n\n        '
        self.safe_serialization = safe_serialization

    def save_checkpoints(self, trainer, checkpoint_path_prefix, output_dir, meta=None, save_optimizers=True):
        if False:
            return 10
        'Save the state dict for lora tune stable diffusion xl model.\n        '
        attn_processors = trainer.model.unet.attn_processors
        unet_lora_layers_to_save = {}
        for (attn_processor_key, attn_processor) in attn_processors.items():
            for (parameter_key, parameter) in attn_processor.state_dict().items():
                unet_lora_layers_to_save[f'{attn_processor_key}.{parameter_key}'] = parameter
        StableDiffusionXLPipeline.save_lora_weights(output_dir, unet_lora_layers=unet_lora_layers_to_save, safe_serialization=self.safe_serialization)

@TRAINERS.register_module(module_name=Trainers.lora_diffusion_xl)
class LoraDiffusionXLTrainer(EpochBasedTrainer):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        'Lora trainers for fine-tuning stable diffusion xl.\n\n        Args:\n            lora_rank: The rank size of lora intermediate linear.\n            safe_serialization: Whether to save the model using safetensors or the traditional PyTorch way with pickle.\n\n        '
        lora_rank = kwargs.pop('lora_rank', 16)
        safe_serialization = kwargs.pop('safe_serialization', False)
        ckpt_hook = list(filter(lambda hook: isinstance(hook, CheckpointHook), self.hooks))[0]
        ckpt_hook.set_processor(LoraDiffusionXLCheckpointProcessor(safe_serialization=safe_serialization))
        unet_lora_attn_procs = {}
        unet_lora_parameters = []
        for (name, attn_processor) in self.model.unet.attn_processors.items():
            cross_attention_dim = None if name.endswith('attn1.processor') else self.model.unet.config.cross_attention_dim
            if name.startswith('mid_block'):
                hidden_size = self.model.unet.config.block_out_channels[-1]
            elif name.startswith('up_blocks'):
                block_id = int(name[len('up_blocks.')])
                hidden_size = list(reversed(self.model.unet.config.block_out_channels))[block_id]
            elif name.startswith('down_blocks'):
                block_id = int(name[len('down_blocks.')])
                hidden_size = self.model.unet.config.block_out_channels[block_id]
            lora_attn_processor_class = LoRAAttnProcessor2_0 if hasattr(F, 'scaled_dot_product_attention') else LoRAAttnProcessor
            module = lora_attn_processor_class(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)
            unet_lora_attn_procs[name] = module
            unet_lora_parameters.extend(module.parameters())
        self.model.unet.set_attn_processor(unet_lora_attn_procs)

    def build_optimizer(self, cfg: ConfigDict, default_args: dict=None):
        if False:
            while True:
                i = 10
        try:
            return build_optimizer(self.model.unet.parameters(), cfg=cfg, default_args=default_args)
        except KeyError as e:
            self.logger.error(f'Build optimizer error, the optimizer {cfg} is a torch native component, please check if your torch with version: {torch.__version__} matches the config.')
            raise e