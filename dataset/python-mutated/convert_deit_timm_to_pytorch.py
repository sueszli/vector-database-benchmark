"""Convert DeiT distilled checkpoints from the timm library."""
import argparse
import json
from pathlib import Path
import requests
import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import DeiTConfig, DeiTForImageClassificationWithTeacher, DeiTImageProcessor
from transformers.utils import logging
logging.set_verbosity_info()
logger = logging.get_logger(__name__)

def create_rename_keys(config, base_model=False):
    if False:
        return 10
    rename_keys = []
    for i in range(config.num_hidden_layers):
        rename_keys.append((f'blocks.{i}.norm1.weight', f'deit.encoder.layer.{i}.layernorm_before.weight'))
        rename_keys.append((f'blocks.{i}.norm1.bias', f'deit.encoder.layer.{i}.layernorm_before.bias'))
        rename_keys.append((f'blocks.{i}.attn.proj.weight', f'deit.encoder.layer.{i}.attention.output.dense.weight'))
        rename_keys.append((f'blocks.{i}.attn.proj.bias', f'deit.encoder.layer.{i}.attention.output.dense.bias'))
        rename_keys.append((f'blocks.{i}.norm2.weight', f'deit.encoder.layer.{i}.layernorm_after.weight'))
        rename_keys.append((f'blocks.{i}.norm2.bias', f'deit.encoder.layer.{i}.layernorm_after.bias'))
        rename_keys.append((f'blocks.{i}.mlp.fc1.weight', f'deit.encoder.layer.{i}.intermediate.dense.weight'))
        rename_keys.append((f'blocks.{i}.mlp.fc1.bias', f'deit.encoder.layer.{i}.intermediate.dense.bias'))
        rename_keys.append((f'blocks.{i}.mlp.fc2.weight', f'deit.encoder.layer.{i}.output.dense.weight'))
        rename_keys.append((f'blocks.{i}.mlp.fc2.bias', f'deit.encoder.layer.{i}.output.dense.bias'))
    rename_keys.extend([('cls_token', 'deit.embeddings.cls_token'), ('dist_token', 'deit.embeddings.distillation_token'), ('patch_embed.proj.weight', 'deit.embeddings.patch_embeddings.projection.weight'), ('patch_embed.proj.bias', 'deit.embeddings.patch_embeddings.projection.bias'), ('pos_embed', 'deit.embeddings.position_embeddings')])
    if base_model:
        rename_keys.extend([('norm.weight', 'layernorm.weight'), ('norm.bias', 'layernorm.bias'), ('pre_logits.fc.weight', 'pooler.dense.weight'), ('pre_logits.fc.bias', 'pooler.dense.bias')])
        rename_keys = [(pair[0], pair[1][4:]) if pair[1].startswith('deit') else pair for pair in rename_keys]
    else:
        rename_keys.extend([('norm.weight', 'deit.layernorm.weight'), ('norm.bias', 'deit.layernorm.bias'), ('head.weight', 'cls_classifier.weight'), ('head.bias', 'cls_classifier.bias'), ('head_dist.weight', 'distillation_classifier.weight'), ('head_dist.bias', 'distillation_classifier.bias')])
    return rename_keys

def read_in_q_k_v(state_dict, config, base_model=False):
    if False:
        return 10
    for i in range(config.num_hidden_layers):
        if base_model:
            prefix = ''
        else:
            prefix = 'deit.'
        in_proj_weight = state_dict.pop(f'blocks.{i}.attn.qkv.weight')
        in_proj_bias = state_dict.pop(f'blocks.{i}.attn.qkv.bias')
        state_dict[f'{prefix}encoder.layer.{i}.attention.attention.query.weight'] = in_proj_weight[:config.hidden_size, :]
        state_dict[f'{prefix}encoder.layer.{i}.attention.attention.query.bias'] = in_proj_bias[:config.hidden_size]
        state_dict[f'{prefix}encoder.layer.{i}.attention.attention.key.weight'] = in_proj_weight[config.hidden_size:config.hidden_size * 2, :]
        state_dict[f'{prefix}encoder.layer.{i}.attention.attention.key.bias'] = in_proj_bias[config.hidden_size:config.hidden_size * 2]
        state_dict[f'{prefix}encoder.layer.{i}.attention.attention.value.weight'] = in_proj_weight[-config.hidden_size:, :]
        state_dict[f'{prefix}encoder.layer.{i}.attention.attention.value.bias'] = in_proj_bias[-config.hidden_size:]

def rename_key(dct, old, new):
    if False:
        i = 10
        return i + 15
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
def convert_deit_checkpoint(deit_name, pytorch_dump_folder_path):
    if False:
        i = 10
        return i + 15
    "\n    Copy/paste/tweak model's weights to our DeiT structure.\n    "
    config = DeiTConfig()
    base_model = False
    config.num_labels = 1000
    repo_id = 'huggingface/label-files'
    filename = 'imagenet-1k-id2label.json'
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type='dataset'), 'r'))
    id2label = {int(k): v for (k, v) in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for (k, v) in id2label.items()}
    config.patch_size = int(deit_name[-6:-4])
    config.image_size = int(deit_name[-3:])
    if deit_name[9:].startswith('tiny'):
        config.hidden_size = 192
        config.intermediate_size = 768
        config.num_hidden_layers = 12
        config.num_attention_heads = 3
    elif deit_name[9:].startswith('small'):
        config.hidden_size = 384
        config.intermediate_size = 1536
        config.num_hidden_layers = 12
        config.num_attention_heads = 6
    if deit_name[9:].startswith('base'):
        pass
    elif deit_name[4:].startswith('large'):
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 24
        config.num_attention_heads = 16
    timm_model = timm.create_model(deit_name, pretrained=True)
    timm_model.eval()
    state_dict = timm_model.state_dict()
    rename_keys = create_rename_keys(config, base_model)
    for (src, dest) in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, config, base_model)
    model = DeiTForImageClassificationWithTeacher(config).eval()
    model.load_state_dict(state_dict)
    size = int(256 / 224 * config.image_size)
    image_processor = DeiTImageProcessor(size=size, crop_size=config.image_size)
    encoding = image_processor(images=prepare_img(), return_tensors='pt')
    pixel_values = encoding['pixel_values']
    outputs = model(pixel_values)
    timm_logits = timm_model(pixel_values)
    assert timm_logits.shape == outputs.logits.shape
    assert torch.allclose(timm_logits, outputs.logits, atol=0.001)
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f'Saving model {deit_name} to {pytorch_dump_folder_path}')
    model.save_pretrained(pytorch_dump_folder_path)
    print(f'Saving image processor to {pytorch_dump_folder_path}')
    image_processor.save_pretrained(pytorch_dump_folder_path)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--deit_name', default='vit_deit_base_distilled_patch16_224', type=str, help="Name of the DeiT timm model you'd like to convert.")
    parser.add_argument('--pytorch_dump_folder_path', default=None, type=str, help='Path to the output PyTorch model directory.')
    args = parser.parse_args()
    convert_deit_checkpoint(args.deit_name, args.pytorch_dump_folder_path)