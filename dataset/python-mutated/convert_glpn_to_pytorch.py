"""Convert GLPN checkpoints."""
import argparse
from collections import OrderedDict
from pathlib import Path
import requests
import torch
from PIL import Image
from transformers import GLPNConfig, GLPNForDepthEstimation, GLPNImageProcessor
from transformers.utils import logging
logging.set_verbosity_info()
logger = logging.get_logger(__name__)

def rename_keys(state_dict):
    if False:
        i = 10
        return i + 15
    new_state_dict = OrderedDict()
    for (key, value) in state_dict.items():
        if key.startswith('module.encoder'):
            key = key.replace('module.encoder', 'glpn.encoder')
        if key.startswith('module.decoder'):
            key = key.replace('module.decoder', 'decoder.stages')
        if 'patch_embed' in key:
            idx = key[key.find('patch_embed') + len('patch_embed')]
            key = key.replace(f'patch_embed{idx}', f'patch_embeddings.{int(idx) - 1}')
        if 'norm' in key:
            key = key.replace('norm', 'layer_norm')
        if 'glpn.encoder.layer_norm' in key:
            idx = key[key.find('glpn.encoder.layer_norm') + len('glpn.encoder.layer_norm')]
            key = key.replace(f'layer_norm{idx}', f'layer_norm.{int(idx) - 1}')
        if 'layer_norm1' in key:
            key = key.replace('layer_norm1', 'layer_norm_1')
        if 'layer_norm2' in key:
            key = key.replace('layer_norm2', 'layer_norm_2')
        if 'block' in key:
            idx = key[key.find('block') + len('block')]
            key = key.replace(f'block{idx}', f'block.{int(idx) - 1}')
        if 'attn.q' in key:
            key = key.replace('attn.q', 'attention.self.query')
        if 'attn.proj' in key:
            key = key.replace('attn.proj', 'attention.output.dense')
        if 'attn' in key:
            key = key.replace('attn', 'attention.self')
        if 'fc1' in key:
            key = key.replace('fc1', 'dense1')
        if 'fc2' in key:
            key = key.replace('fc2', 'dense2')
        if 'linear_pred' in key:
            key = key.replace('linear_pred', 'classifier')
        if 'linear_fuse' in key:
            key = key.replace('linear_fuse.conv', 'linear_fuse')
            key = key.replace('linear_fuse.bn', 'batch_norm')
        if 'linear_c' in key:
            idx = key[key.find('linear_c') + len('linear_c')]
            key = key.replace(f'linear_c{idx}', f'linear_c.{int(idx) - 1}')
        if 'bot_conv' in key:
            key = key.replace('bot_conv', '0.convolution')
        if 'skip_conv1' in key:
            key = key.replace('skip_conv1', '1.convolution')
        if 'skip_conv2' in key:
            key = key.replace('skip_conv2', '2.convolution')
        if 'fusion1' in key:
            key = key.replace('fusion1', '1.fusion')
        if 'fusion2' in key:
            key = key.replace('fusion2', '2.fusion')
        if 'fusion3' in key:
            key = key.replace('fusion3', '3.fusion')
        if 'fusion' in key and 'conv' in key:
            key = key.replace('conv', 'convolutional_layer')
        if key.startswith('module.last_layer_depth'):
            key = key.replace('module.last_layer_depth', 'head.head')
        new_state_dict[key] = value
    return new_state_dict

def read_in_k_v(state_dict, config):
    if False:
        return 10
    for i in range(config.num_encoder_blocks):
        for j in range(config.depths[i]):
            kv_weight = state_dict.pop(f'glpn.encoder.block.{i}.{j}.attention.self.kv.weight')
            kv_bias = state_dict.pop(f'glpn.encoder.block.{i}.{j}.attention.self.kv.bias')
            state_dict[f'glpn.encoder.block.{i}.{j}.attention.self.key.weight'] = kv_weight[:config.hidden_sizes[i], :]
            state_dict[f'glpn.encoder.block.{i}.{j}.attention.self.key.bias'] = kv_bias[:config.hidden_sizes[i]]
            state_dict[f'glpn.encoder.block.{i}.{j}.attention.self.value.weight'] = kv_weight[config.hidden_sizes[i]:, :]
            state_dict[f'glpn.encoder.block.{i}.{j}.attention.self.value.bias'] = kv_bias[config.hidden_sizes[i]:]

def prepare_img():
    if False:
        print('Hello World!')
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    return image

@torch.no_grad()
def convert_glpn_checkpoint(checkpoint_path, pytorch_dump_folder_path, push_to_hub=False, model_name=None):
    if False:
        return 10
    "\n    Copy/paste/tweak model's weights to our GLPN structure.\n    "
    config = GLPNConfig(hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=64, depths=[3, 8, 27, 3])
    image_processor = GLPNImageProcessor()
    image = prepare_img()
    pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
    logger.info('Converting model...')
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict = rename_keys(state_dict)
    read_in_k_v(state_dict, config)
    model = GLPNForDepthEstimation(config)
    model.load_state_dict(state_dict)
    model.eval()
    outputs = model(pixel_values)
    predicted_depth = outputs.predicted_depth
    if model_name is not None:
        if 'nyu' in model_name:
            expected_slice = torch.tensor([[4.4147, 4.0873, 4.0673], [3.789, 3.2881, 3.1525], [3.7674, 3.5423, 3.4913]])
        elif 'kitti' in model_name:
            expected_slice = torch.tensor([[3.4291, 2.7865, 2.5151], [3.2841, 2.7021, 2.3502], [3.1147, 2.4625, 2.2481]])
        else:
            raise ValueError(f'Unknown model name: {model_name}')
        expected_shape = torch.Size([1, 480, 640])
        assert predicted_depth.shape == expected_shape
        assert torch.allclose(predicted_depth[0, :3, :3], expected_slice, atol=0.0001)
        print('Looks ok!')
    if push_to_hub:
        logger.info('Pushing model and image processor to the hub...')
        model.push_to_hub(repo_path_or_name=Path(pytorch_dump_folder_path, model_name), organization='nielsr', commit_message='Add model', use_temp_dir=True)
        image_processor.push_to_hub(repo_path_or_name=Path(pytorch_dump_folder_path, model_name), organization='nielsr', commit_message='Add image processor', use_temp_dir=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to the original PyTorch checkpoint (.pth file).')
    parser.add_argument('--pytorch_dump_folder_path', default=None, type=str, help='Path to the folder to output PyTorch model.')
    parser.add_argument('--push_to_hub', action='store_true', help='Whether to upload the model to the HuggingFace hub.')
    parser.add_argument('--model_name', default='glpn-kitti', type=str, help="Name of the model in case you're pushing to the hub.")
    args = parser.parse_args()
    convert_glpn_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub, args.model_name)