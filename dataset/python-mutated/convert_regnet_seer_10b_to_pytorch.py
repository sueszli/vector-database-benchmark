"""Convert RegNet 10B checkpoints vissl."""
import argparse
import json
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from classy_vision.models.regnet import RegNet, RegNetParams
from huggingface_hub import cached_download, hf_hub_url
from torch import Tensor
from vissl.models.model_helpers import get_trunk_forward_outputs
from transformers import AutoImageProcessor, RegNetConfig, RegNetForImageClassification, RegNetModel
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
logging.set_verbosity_info()
logger = logging.get_logger()

@dataclass
class Tracker:
    module: nn.Module
    traced: List[nn.Module] = field(default_factory=list)
    handles: list = field(default_factory=list)
    name2module: Dict[str, nn.Module] = field(default_factory=OrderedDict)

    def _forward_hook(self, m, inputs: Tensor, outputs: Tensor, name: str):
        if False:
            while True:
                i = 10
        has_not_submodules = len(list(m.modules())) == 1 or isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d)
        if has_not_submodules:
            self.traced.append(m)
            self.name2module[name] = m

    def __call__(self, x: Tensor):
        if False:
            print('Hello World!')
        for (name, m) in self.module.named_modules():
            self.handles.append(m.register_forward_hook(partial(self._forward_hook, name=name)))
        self.module(x)
        [x.remove() for x in self.handles]
        return self

    @property
    def parametrized(self):
        if False:
            for i in range(10):
                print('nop')
        return {k: v for (k, v) in self.name2module.items() if len(list(v.state_dict().keys())) > 0}

class FakeRegNetVisslWrapper(nn.Module):
    """
    Fake wrapper for RegNet that mimics what vissl does without the need to pass a config file.
    """

    def __init__(self, model: nn.Module):
        if False:
            i = 10
            return i + 15
        super().__init__()
        feature_blocks: List[Tuple[str, nn.Module]] = []
        feature_blocks.append(('conv1', model.stem))
        for (k, v) in model.trunk_output.named_children():
            assert k.startswith('block'), f'Unexpected layer name {k}'
            block_index = len(feature_blocks) + 1
            feature_blocks.append((f'res{block_index}', v))
        self._feature_blocks = nn.ModuleDict(feature_blocks)

    def forward(self, x: Tensor):
        if False:
            print('Hello World!')
        return get_trunk_forward_outputs(x, out_feat_keys=None, feature_blocks=self._feature_blocks)

class FakeRegNetParams(RegNetParams):
    """
    Used to instantiace a RegNet model from classy vision with the same depth as the 10B one but with super small
    parameters, so we can trace it in memory.
    """

    def get_expanded_params(self):
        if False:
            while True:
                i = 10
        return [(8, 2, 2, 8, 1.0), (8, 2, 7, 8, 1.0), (8, 2, 17, 8, 1.0), (8, 2, 1, 8, 1.0)]

def get_from_to_our_keys(model_name: str) -> Dict[str, str]:
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns a dictionary that maps from original model's key -> our implementation's keys\n    "
    our_config = RegNetConfig(depths=[2, 7, 17, 1], hidden_sizes=[8, 8, 8, 8], groups_width=8)
    if 'in1k' in model_name:
        our_model = RegNetForImageClassification(our_config)
    else:
        our_model = RegNetModel(our_config)
    from_model = FakeRegNetVisslWrapper(RegNet(FakeRegNetParams(depth=27, group_width=1010, w_0=1744, w_a=620.83, w_m=2.52)))
    with torch.no_grad():
        from_model = from_model.eval()
        our_model = our_model.eval()
        x = torch.randn((1, 3, 32, 32))
        dest_tracker = Tracker(our_model)
        dest_traced = dest_tracker(x).parametrized
        pprint(dest_tracker.name2module)
        src_tracker = Tracker(from_model)
        src_traced = src_tracker(x).parametrized

    def to_params_dict(dict_with_modules):
        if False:
            i = 10
            return i + 15
        params_dict = OrderedDict()
        for (name, module) in dict_with_modules.items():
            for (param_name, param) in module.state_dict().items():
                params_dict[f'{name}.{param_name}'] = param
        return params_dict
    from_to_ours_keys = {}
    src_state_dict = to_params_dict(src_traced)
    dst_state_dict = to_params_dict(dest_traced)
    for ((src_key, src_param), (dest_key, dest_param)) in zip(src_state_dict.items(), dst_state_dict.items()):
        from_to_ours_keys[src_key] = dest_key
        logger.info(f'{src_key} -> {dest_key}')
    if 'in1k' in model_name:
        from_to_ours_keys['0.clf.0.weight'] = 'classifier.1.weight'
        from_to_ours_keys['0.clf.0.bias'] = 'classifier.1.bias'
    return from_to_ours_keys

def convert_weights_and_push(save_directory: Path, model_name: str=None, push_to_hub: bool=True):
    if False:
        while True:
            i = 10
    filename = 'imagenet-1k-id2label.json'
    num_labels = 1000
    repo_id = 'huggingface/label-files'
    num_labels = num_labels
    id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename, repo_type='dataset')), 'r'))
    id2label = {int(k): v for (k, v) in id2label.items()}
    id2label = id2label
    label2id = {v: k for (k, v) in id2label.items()}
    ImageNetPreTrainedConfig = partial(RegNetConfig, num_labels=num_labels, id2label=id2label, label2id=label2id)
    names_to_config = {'regnet-y-10b-seer': ImageNetPreTrainedConfig(depths=[2, 7, 17, 1], hidden_sizes=[2020, 4040, 11110, 28280], groups_width=1010), 'regnet-y-10b-seer-in1k': ImageNetPreTrainedConfig(depths=[2, 7, 17, 1], hidden_sizes=[2020, 4040, 11110, 28280], groups_width=1010)}

    def load_using_classy_vision(checkpoint_url: str) -> Tuple[Dict, Dict]:
        if False:
            while True:
                i = 10
        files = torch.hub.load_state_dict_from_url(checkpoint_url, model_dir=str(save_directory), map_location='cpu')
        model_state_dict = files['classy_state_dict']['base_model']['model']
        return (model_state_dict['trunk'], model_state_dict['heads'])
    names_to_from_model = {'regnet-y-10b-seer': partial(load_using_classy_vision, 'https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_regnet10B/model_iteration124500_conso.torch'), 'regnet-y-10b-seer-in1k': partial(load_using_classy_vision, 'https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_finetuned/seer_10b_finetuned_in1k_model_phase28_conso.torch')}
    from_to_ours_keys = get_from_to_our_keys(model_name)
    if not (save_directory / f'{model_name}.pth').exists():
        logger.info('Loading original state_dict.')
        (from_state_dict_trunk, from_state_dict_head) = names_to_from_model[model_name]()
        from_state_dict = from_state_dict_trunk
        if 'in1k' in model_name:
            from_state_dict = {**from_state_dict_trunk, **from_state_dict_head}
        logger.info('Done!')
        converted_state_dict = {}
        not_used_keys = list(from_state_dict.keys())
        regex = '\\.block.-part.'
        for key in from_state_dict.keys():
            src_key = re.sub(regex, '', key)
            dest_key = from_to_ours_keys[src_key]
            converted_state_dict[dest_key] = from_state_dict[key]
            not_used_keys.remove(key)
        assert len(not_used_keys) == 0, f"Some keys where not used {','.join(not_used_keys)}"
        logger.info(f"The following keys were not used: {','.join(not_used_keys)}")
        torch.save(converted_state_dict, save_directory / f'{model_name}.pth')
        del converted_state_dict
    else:
        logger.info('The state_dict was already stored on disk.')
    if push_to_hub:
        logger.info(f"Token is {os.environ['HF_TOKEN']}")
        logger.info('Loading our model.')
        our_config = names_to_config[model_name]
        our_model_func = RegNetModel
        if 'in1k' in model_name:
            our_model_func = RegNetForImageClassification
        our_model = our_model_func(our_config)
        our_model.to(torch.device('meta'))
        logger.info('Loading state_dict in our model.')
        state_dict_keys = our_model.state_dict().keys()
        PreTrainedModel._load_pretrained_model_low_mem(our_model, state_dict_keys, [save_directory / f'{model_name}.pth'])
        logger.info('Finally, pushing!')
        our_model.push_to_hub(repo_path_or_name=save_directory / model_name, commit_message='Add model', output_dir=save_directory / model_name)
        size = 384
        image_processor = AutoImageProcessor.from_pretrained('facebook/convnext-base-224-22k-1k', size=size)
        image_processor.push_to_hub(repo_path_or_name=save_directory / model_name, commit_message='Add image processor', output_dir=save_directory / model_name)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default=None, type=str, help='The name of the model you wish to convert, it must be one of the supported regnet* architecture, currently: regnetx-*, regnety-*. If `None`, all of them will the converted.')
    parser.add_argument('--pytorch_dump_folder_path', default=None, type=Path, required=True, help='Path to the output PyTorch model directory.')
    parser.add_argument('--push_to_hub', default=True, type=bool, required=False, help='If True, push model and image processor to the hub.')
    args = parser.parse_args()
    pytorch_dump_folder_path: Path = args.pytorch_dump_folder_path
    pytorch_dump_folder_path.mkdir(exist_ok=True, parents=True)
    convert_weights_and_push(pytorch_dump_folder_path, args.model_name, args.push_to_hub)