"""Convert VitMatte checkpoints from the original repository.

URL: https://github.com/hustvl/ViTMatte
"""
import argparse
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import VitDetConfig, VitMatteConfig, VitMatteForImageMatting, VitMatteImageProcessor

def get_config(model_name):
    if False:
        for i in range(10):
            print('nop')
    hidden_size = 384 if 'small' in model_name else 768
    num_attention_heads = 6 if 'small' in model_name else 12
    backbone_config = VitDetConfig(num_channels=4, image_size=512, pretrain_image_size=224, patch_size=16, hidden_size=hidden_size, num_attention_heads=num_attention_heads, use_absolute_position_embeddings=True, use_relative_position_embeddings=True, window_size=14, window_block_indices=[0, 1, 3, 4, 6, 7, 9, 10], residual_block_indices=[2, 5, 8, 11], out_features=['stage12'])
    return VitMatteConfig(backbone_config=backbone_config, hidden_size=hidden_size)

def create_rename_keys(config):
    if False:
        for i in range(10):
            print('nop')
    rename_keys = []
    rename_keys.append(('backbone.pos_embed', 'backbone.embeddings.position_embeddings'))
    rename_keys.append(('backbone.patch_embed.proj.weight', 'backbone.embeddings.projection.weight'))
    rename_keys.append(('backbone.patch_embed.proj.bias', 'backbone.embeddings.projection.bias'))
    return rename_keys

def rename_key(dct, old, new):
    if False:
        return 10
    val = dct.pop(old)
    dct[new] = val

def convert_vitmatte_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    if False:
        for i in range(10):
            print('nop')
    config = get_config(model_name)
    model_name_to_filename = {'vitmatte-small-composition-1k': 'ViTMatte_S_Com.pth', 'vitmatte-base-composition-1k': 'ViTMatte_B_Com.pth', 'vitmatte-small-distinctions-646': 'ViTMatte_S_DIS.pth', 'vitmatte-base-distinctions-646': 'ViTMatte_B_DIS.pth'}
    filename = model_name_to_filename[model_name]
    filepath = hf_hub_download(repo_id='nielsr/vitmatte-checkpoints', filename=filename, repo_type='model')
    state_dict = torch.load(filepath, map_location='cpu')
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        if 'backbone.blocks' in key:
            key = key.replace('backbone.blocks', 'backbone.encoder.layer')
        if 'attn' in key:
            key = key.replace('attn', 'attention')
        if 'fusion_blks' in key:
            key = key.replace('fusion_blks', 'fusion_blocks')
        if 'bn' in key:
            key = key.replace('bn', 'batch_norm')
        state_dict[key] = val
    rename_keys = create_rename_keys(config)
    for (src, dest) in rename_keys:
        rename_key(state_dict, src, dest)
    processor = VitMatteImageProcessor()
    model = VitMatteForImageMatting(config)
    model.eval()
    model.load_state_dict(state_dict)
    url = 'https://github.com/hustvl/ViTMatte/blob/main/demo/bulb_rgb.png?raw=true'
    image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    url = 'https://github.com/hustvl/ViTMatte/blob/main/demo/bulb_trimap.png?raw=true'
    trimap = Image.open(requests.get(url, stream=True).raw)
    pixel_values = processor(images=image, trimaps=trimap.convert('L'), return_tensors='pt').pixel_values
    with torch.no_grad():
        alphas = model(pixel_values).alphas
    if model_name == 'vitmatte-small-composition-1k':
        expected_slice = torch.tensor([[0.9977, 0.9987, 0.999], [0.998, 0.9998, 0.9998], [0.9983, 0.9998, 0.9998]])
    elif model_name == 'vitmatte-base-composition-1k':
        expected_slice = torch.tensor([[0.9972, 0.9971, 0.9981], [0.9948, 0.9987, 0.9994], [0.9963, 0.9992, 0.9995]])
    elif model_name == 'vitmatte-small-distinctions-646':
        expected_slice = torch.tensor([[0.988, 0.997, 0.9972], [0.996, 0.9996, 0.9997], [0.9963, 0.9996, 0.9997]])
    elif model_name == 'vitmatte-base-distinctions-646':
        expected_slice = torch.tensor([[0.9963, 0.9998, 0.9999], [0.9995, 1.0, 1.0], [0.9992, 0.9999, 1.0]])
    assert torch.allclose(alphas[0, 0, :3, :3], expected_slice, atol=0.0001)
    print('Looks ok!')
    if pytorch_dump_folder_path is not None:
        print(f'Saving model and processor of {model_name} to {pytorch_dump_folder_path}')
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)
    if push_to_hub:
        print(f'Pushing model and processor for {model_name} to hub')
        model.push_to_hub(f'hustvl/{model_name}')
        processor.push_to_hub(f'hustvl/{model_name}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='vitmatte-small-composition-1k', type=str, choices=['vitmatte-small-composition-1k', 'vitmatte-base-composition-1k', 'vitmatte-small-distinctions-646', 'vitmatte-base-distinctions-646'], help="Name of the VitMatte model you'd like to convert.")
    parser.add_argument('--pytorch_dump_folder_path', default=None, type=str, help='Path to the output PyTorch model directory.')
    parser.add_argument('--push_to_hub', action='store_true', help='Whether or not to push the converted model to the 🤗 hub.')
    args = parser.parse_args()
    convert_vitmatte_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)