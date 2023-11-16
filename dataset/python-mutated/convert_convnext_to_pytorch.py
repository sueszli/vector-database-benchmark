"""Convert ConvNext checkpoints from the original repository.

URL: https://github.com/facebookresearch/ConvNeXt"""
import argparse
import json
from pathlib import Path
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import ConvNextConfig, ConvNextForImageClassification, ConvNextImageProcessor
from transformers.utils import logging
logging.set_verbosity_info()
logger = logging.get_logger(__name__)

def get_convnext_config(checkpoint_url):
    if False:
        return 10
    config = ConvNextConfig()
    if 'tiny' in checkpoint_url:
        depths = [3, 3, 9, 3]
        hidden_sizes = [96, 192, 384, 768]
    if 'small' in checkpoint_url:
        depths = [3, 3, 27, 3]
        hidden_sizes = [96, 192, 384, 768]
    if 'base' in checkpoint_url:
        depths = [3, 3, 27, 3]
        hidden_sizes = [128, 256, 512, 1024]
    if 'large' in checkpoint_url:
        depths = [3, 3, 27, 3]
        hidden_sizes = [192, 384, 768, 1536]
    if 'xlarge' in checkpoint_url:
        depths = [3, 3, 27, 3]
        hidden_sizes = [256, 512, 1024, 2048]
    if '1k' in checkpoint_url:
        num_labels = 1000
        filename = 'imagenet-1k-id2label.json'
        expected_shape = (1, 1000)
    else:
        num_labels = 21841
        filename = 'imagenet-22k-id2label.json'
        expected_shape = (1, 21841)
    repo_id = 'huggingface/label-files'
    config.num_labels = num_labels
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type='dataset'), 'r'))
    id2label = {int(k): v for (k, v) in id2label.items()}
    if '1k' not in checkpoint_url:
        del id2label[9205]
        del id2label[15027]
    config.id2label = id2label
    config.label2id = {v: k for (k, v) in id2label.items()}
    config.hidden_sizes = hidden_sizes
    config.depths = depths
    return (config, expected_shape)

def rename_key(name):
    if False:
        i = 10
        return i + 15
    if 'downsample_layers.0.0' in name:
        name = name.replace('downsample_layers.0.0', 'embeddings.patch_embeddings')
    if 'downsample_layers.0.1' in name:
        name = name.replace('downsample_layers.0.1', 'embeddings.norm')
    if 'downsample_layers.1.0' in name:
        name = name.replace('downsample_layers.1.0', 'stages.1.downsampling_layer.0')
    if 'downsample_layers.1.1' in name:
        name = name.replace('downsample_layers.1.1', 'stages.1.downsampling_layer.1')
    if 'downsample_layers.2.0' in name:
        name = name.replace('downsample_layers.2.0', 'stages.2.downsampling_layer.0')
    if 'downsample_layers.2.1' in name:
        name = name.replace('downsample_layers.2.1', 'stages.2.downsampling_layer.1')
    if 'downsample_layers.3.0' in name:
        name = name.replace('downsample_layers.3.0', 'stages.3.downsampling_layer.0')
    if 'downsample_layers.3.1' in name:
        name = name.replace('downsample_layers.3.1', 'stages.3.downsampling_layer.1')
    if 'stages' in name and 'downsampling_layer' not in name:
        name = name[:len('stages.0')] + '.layers' + name[len('stages.0'):]
    if 'stages' in name:
        name = name.replace('stages', 'encoder.stages')
    if 'norm' in name:
        name = name.replace('norm', 'layernorm')
    if 'gamma' in name:
        name = name.replace('gamma', 'layer_scale_parameter')
    if 'head' in name:
        name = name.replace('head', 'classifier')
    return name

def prepare_img():
    if False:
        print('Hello World!')
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    im = Image.open(requests.get(url, stream=True).raw)
    return im

@torch.no_grad()
def convert_convnext_checkpoint(checkpoint_url, pytorch_dump_folder_path):
    if False:
        return 10
    "\n    Copy/paste/tweak model's weights to our ConvNext structure.\n    "
    (config, expected_shape) = get_convnext_config(checkpoint_url)
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)['model']
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        state_dict[rename_key(key)] = val
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        if not key.startswith('classifier'):
            key = 'convnext.' + key
        state_dict[key] = val
    model = ConvNextForImageClassification(config)
    model.load_state_dict(state_dict)
    model.eval()
    size = 224 if '224' in checkpoint_url else 384
    image_processor = ConvNextImageProcessor(size=size)
    pixel_values = image_processor(images=prepare_img(), return_tensors='pt').pixel_values
    logits = model(pixel_values).logits
    if checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth':
        expected_logits = torch.tensor([-0.121, -0.6605, 0.1918])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth':
        expected_logits = torch.tensor([-0.4473, -0.1847, -0.6365])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth':
        expected_logits = torch.tensor([0.4525, 0.7539, 0.0308])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_384.pth':
        expected_logits = torch.tensor([0.3561, 0.635, -0.0384])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth':
        expected_logits = torch.tensor([0.4174, -0.0989, 0.1489])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_384.pth':
        expected_logits = torch.tensor([0.2513, -0.1349, -0.1613])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth':
        expected_logits = torch.tensor([1.298, 0.3631, -0.1198])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth':
        expected_logits = torch.tensor([1.2963, 0.1227, 0.1723])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth':
        expected_logits = torch.tensor([1.7956, 0.839, 0.282])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth':
        expected_logits = torch.tensor([-0.2822, -0.0502, -0.0878])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_384.pth':
        expected_logits = torch.tensor([-0.5672, -0.073, -0.4348])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_224.pth':
        expected_logits = torch.tensor([0.2681, 0.2365, 0.6246])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_384.pth':
        expected_logits = torch.tensor([-0.2642, 0.3931, 0.5116])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_224_ema.pth':
        expected_logits = torch.tensor([-0.6677, -0.1873, -0.8379])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_384_ema.pth':
        expected_logits = torch.tensor([-0.7749, -0.2967, -0.6444])
    else:
        raise ValueError(f'Unknown URL: {checkpoint_url}')
    assert torch.allclose(logits[0, :3], expected_logits, atol=0.001)
    assert logits.shape == expected_shape
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f'Saving model to {pytorch_dump_folder_path}')
    model.save_pretrained(pytorch_dump_folder_path)
    print(f'Saving image processor to {pytorch_dump_folder_path}')
    image_processor.save_pretrained(pytorch_dump_folder_path)
    print('Pushing model to the hub...')
    model_name = 'convnext'
    if 'tiny' in checkpoint_url:
        model_name += '-tiny'
    elif 'small' in checkpoint_url:
        model_name += '-small'
    elif 'base' in checkpoint_url:
        model_name += '-base'
    elif 'xlarge' in checkpoint_url:
        model_name += '-xlarge'
    elif 'large' in checkpoint_url:
        model_name += '-large'
    if '224' in checkpoint_url:
        model_name += '-224'
    elif '384' in checkpoint_url:
        model_name += '-384'
    if '22k' in checkpoint_url and '1k' not in checkpoint_url:
        model_name += '-22k'
    if '22k' in checkpoint_url and '1k' in checkpoint_url:
        model_name += '-22k-1k'
    model.push_to_hub(repo_path_or_name=Path(pytorch_dump_folder_path, model_name), organization='nielsr', commit_message='Add model')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_url', default='https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth', type=str, help="URL of the original ConvNeXT checkpoint you'd like to convert.")
    parser.add_argument('--pytorch_dump_folder_path', default=None, type=str, required=True, help='Path to the output PyTorch model directory.')
    args = parser.parse_args()
    convert_convnext_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path)