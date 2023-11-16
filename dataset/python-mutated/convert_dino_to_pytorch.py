"""Convert ViT checkpoints trained with the DINO method."""
import argparse
import json
from pathlib import Path
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor, ViTModel
from transformers.utils import logging
logging.set_verbosity_info()
logger = logging.get_logger(__name__)

def create_rename_keys(config, base_model=False):
    if False:
        print('Hello World!')
    rename_keys = []
    for i in range(config.num_hidden_layers):
        rename_keys.append((f'blocks.{i}.norm1.weight', f'vit.encoder.layer.{i}.layernorm_before.weight'))
        rename_keys.append((f'blocks.{i}.norm1.bias', f'vit.encoder.layer.{i}.layernorm_before.bias'))
        rename_keys.append((f'blocks.{i}.attn.proj.weight', f'vit.encoder.layer.{i}.attention.output.dense.weight'))
        rename_keys.append((f'blocks.{i}.attn.proj.bias', f'vit.encoder.layer.{i}.attention.output.dense.bias'))
        rename_keys.append((f'blocks.{i}.norm2.weight', f'vit.encoder.layer.{i}.layernorm_after.weight'))
        rename_keys.append((f'blocks.{i}.norm2.bias', f'vit.encoder.layer.{i}.layernorm_after.bias'))
        rename_keys.append((f'blocks.{i}.mlp.fc1.weight', f'vit.encoder.layer.{i}.intermediate.dense.weight'))
        rename_keys.append((f'blocks.{i}.mlp.fc1.bias', f'vit.encoder.layer.{i}.intermediate.dense.bias'))
        rename_keys.append((f'blocks.{i}.mlp.fc2.weight', f'vit.encoder.layer.{i}.output.dense.weight'))
        rename_keys.append((f'blocks.{i}.mlp.fc2.bias', f'vit.encoder.layer.{i}.output.dense.bias'))
    rename_keys.extend([('cls_token', 'vit.embeddings.cls_token'), ('patch_embed.proj.weight', 'vit.embeddings.patch_embeddings.projection.weight'), ('patch_embed.proj.bias', 'vit.embeddings.patch_embeddings.projection.bias'), ('pos_embed', 'vit.embeddings.position_embeddings')])
    if base_model:
        rename_keys.extend([('norm.weight', 'layernorm.weight'), ('norm.bias', 'layernorm.bias')])
        rename_keys = [(pair[0], pair[1][4:]) if pair[1].startswith('vit') else pair for pair in rename_keys]
    else:
        rename_keys.extend([('norm.weight', 'vit.layernorm.weight'), ('norm.bias', 'vit.layernorm.bias'), ('head.weight', 'classifier.weight'), ('head.bias', 'classifier.bias')])
    return rename_keys

def read_in_q_k_v(state_dict, config, base_model=False):
    if False:
        i = 10
        return i + 15
    for i in range(config.num_hidden_layers):
        if base_model:
            prefix = ''
        else:
            prefix = 'vit.'
        in_proj_weight = state_dict.pop(f'blocks.{i}.attn.qkv.weight')
        in_proj_bias = state_dict.pop(f'blocks.{i}.attn.qkv.bias')
        state_dict[f'{prefix}encoder.layer.{i}.attention.attention.query.weight'] = in_proj_weight[:config.hidden_size, :]
        state_dict[f'{prefix}encoder.layer.{i}.attention.attention.query.bias'] = in_proj_bias[:config.hidden_size]
        state_dict[f'{prefix}encoder.layer.{i}.attention.attention.key.weight'] = in_proj_weight[config.hidden_size:config.hidden_size * 2, :]
        state_dict[f'{prefix}encoder.layer.{i}.attention.attention.key.bias'] = in_proj_bias[config.hidden_size:config.hidden_size * 2]
        state_dict[f'{prefix}encoder.layer.{i}.attention.attention.value.weight'] = in_proj_weight[-config.hidden_size:, :]
        state_dict[f'{prefix}encoder.layer.{i}.attention.attention.value.bias'] = in_proj_bias[-config.hidden_size:]

def remove_classification_head_(state_dict):
    if False:
        print('Hello World!')
    ignore_keys = ['head.weight', 'head.bias']
    for k in ignore_keys:
        state_dict.pop(k, None)

def rename_key(dct, old, new):
    if False:
        while True:
            i = 10
    val = dct.pop(old)
    dct[new] = val

def prepare_img():
    if False:
        i = 10
        return i + 15
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    im = Image.open(requests.get(url, stream=True).raw)
    return im

@torch.no_grad()
def convert_vit_checkpoint(model_name, pytorch_dump_folder_path, base_model=True):
    if False:
        print('Hello World!')
    "\n    Copy/paste/tweak model's weights to our ViT structure.\n    "
    config = ViTConfig()
    if model_name[-1] == '8':
        config.patch_size = 8
    if not base_model:
        config.num_labels = 1000
        repo_id = 'huggingface/label-files'
        filename = 'imagenet-1k-id2label.json'
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type='dataset'), 'r'))
        id2label = {int(k): v for (k, v) in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for (k, v) in id2label.items()}
    if model_name in ['dino_vits8', 'dino_vits16']:
        config.hidden_size = 384
        config.intermediate_size = 1536
        config.num_hidden_layers = 12
        config.num_attention_heads = 6
    original_model = torch.hub.load('facebookresearch/dino:main', model_name)
    original_model.eval()
    state_dict = original_model.state_dict()
    if base_model:
        remove_classification_head_(state_dict)
    rename_keys = create_rename_keys(config, base_model=base_model)
    for (src, dest) in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, config, base_model)
    if base_model:
        model = ViTModel(config, add_pooling_layer=False).eval()
    else:
        model = ViTForImageClassification(config).eval()
    model.load_state_dict(state_dict)
    image_processor = ViTImageProcessor()
    encoding = image_processor(images=prepare_img(), return_tensors='pt')
    pixel_values = encoding['pixel_values']
    outputs = model(pixel_values)
    if base_model:
        final_hidden_state_cls_token = original_model(pixel_values)
        assert torch.allclose(final_hidden_state_cls_token, outputs.last_hidden_state[:, 0, :], atol=0.1)
    else:
        logits = original_model(pixel_values)
        assert logits.shape == outputs.logits.shape
        assert torch.allclose(logits, outputs.logits, atol=0.001)
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f'Saving model {model_name} to {pytorch_dump_folder_path}')
    model.save_pretrained(pytorch_dump_folder_path)
    print(f'Saving image processor to {pytorch_dump_folder_path}')
    image_processor.save_pretrained(pytorch_dump_folder_path)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='dino_vitb16', type=str, help="Name of the model trained with DINO you'd like to convert.")
    parser.add_argument('--pytorch_dump_folder_path', default=None, type=str, help='Path to the output PyTorch model directory.')
    parser.add_argument('--base_model', action='store_true', help='Whether to only convert the base model (no projection head weights).')
    parser.set_defaults(base_model=True)
    args = parser.parse_args()
    convert_vit_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.base_model)