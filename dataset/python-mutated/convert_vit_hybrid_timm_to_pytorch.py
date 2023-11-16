"""Convert ViT hybrid checkpoints from the timm library."""
import argparse
import json
from pathlib import Path
import requests
import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from transformers import BitConfig, ViTHybridConfig, ViTHybridForImageClassification, ViTHybridImageProcessor, ViTHybridModel
from transformers.image_utils import PILImageResampling
from transformers.utils import logging
logging.set_verbosity_info()
logger = logging.get_logger(__name__)

def create_rename_keys(config, base_model=False):
    if False:
        for i in range(10):
            print('nop')
    rename_keys = []
    rename_keys.append(('cls_token', 'vit.embeddings.cls_token'))
    rename_keys.append(('pos_embed', 'vit.embeddings.position_embeddings'))
    rename_keys.append(('patch_embed.proj.weight', 'vit.embeddings.patch_embeddings.projection.weight'))
    rename_keys.append(('patch_embed.proj.bias', 'vit.embeddings.patch_embeddings.projection.bias'))
    rename_keys.append(('patch_embed.backbone.stem.conv.weight', 'vit.embeddings.patch_embeddings.backbone.bit.embedder.convolution.weight'))
    rename_keys.append(('patch_embed.backbone.stem.norm.weight', 'vit.embeddings.patch_embeddings.backbone.bit.embedder.norm.weight'))
    rename_keys.append(('patch_embed.backbone.stem.norm.bias', 'vit.embeddings.patch_embeddings.backbone.bit.embedder.norm.bias'))
    for stage_idx in range(len(config.backbone_config.depths)):
        for layer_idx in range(config.backbone_config.depths[stage_idx]):
            rename_keys.append((f'patch_embed.backbone.stages.{stage_idx}.blocks.{layer_idx}.conv1.weight', f'vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.{layer_idx}.conv1.weight'))
            rename_keys.append((f'patch_embed.backbone.stages.{stage_idx}.blocks.{layer_idx}.norm1.weight', f'vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.{layer_idx}.norm1.weight'))
            rename_keys.append((f'patch_embed.backbone.stages.{stage_idx}.blocks.{layer_idx}.norm1.bias', f'vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.{layer_idx}.norm1.bias'))
            rename_keys.append((f'patch_embed.backbone.stages.{stage_idx}.blocks.{layer_idx}.conv2.weight', f'vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.{layer_idx}.conv2.weight'))
            rename_keys.append((f'patch_embed.backbone.stages.{stage_idx}.blocks.{layer_idx}.norm2.weight', f'vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.{layer_idx}.norm2.weight'))
            rename_keys.append((f'patch_embed.backbone.stages.{stage_idx}.blocks.{layer_idx}.norm2.bias', f'vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.{layer_idx}.norm2.bias'))
            rename_keys.append((f'patch_embed.backbone.stages.{stage_idx}.blocks.{layer_idx}.conv3.weight', f'vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.{layer_idx}.conv3.weight'))
            rename_keys.append((f'patch_embed.backbone.stages.{stage_idx}.blocks.{layer_idx}.norm3.weight', f'vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.{layer_idx}.norm3.weight'))
            rename_keys.append((f'patch_embed.backbone.stages.{stage_idx}.blocks.{layer_idx}.norm3.bias', f'vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.{layer_idx}.norm3.bias'))
        rename_keys.append((f'patch_embed.backbone.stages.{stage_idx}.blocks.0.downsample.conv.weight', f'vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.0.downsample.conv.weight'))
        rename_keys.append((f'patch_embed.backbone.stages.{stage_idx}.blocks.0.downsample.norm.weight', f'vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.0.downsample.norm.weight'))
        rename_keys.append((f'patch_embed.backbone.stages.{stage_idx}.blocks.0.downsample.norm.bias', f'vit.embeddings.patch_embeddings.backbone.bit.encoder.stages.{stage_idx}.layers.0.downsample.norm.bias'))
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
    if base_model:
        rename_keys.extend([('norm.weight', 'layernorm.weight'), ('norm.bias', 'layernorm.bias'), ('pre_logits.fc.weight', 'pooler.dense.weight'), ('pre_logits.fc.bias', 'pooler.dense.bias')])
        rename_keys = [(pair[0], pair[1][4:]) if pair[1].startswith('vit') else pair for pair in rename_keys]
    else:
        rename_keys.extend([('norm.weight', 'vit.layernorm.weight'), ('norm.bias', 'vit.layernorm.bias'), ('head.weight', 'classifier.weight'), ('head.bias', 'classifier.bias')])
    return rename_keys

def read_in_q_k_v(state_dict, config, base_model=False):
    if False:
        print('Hello World!')
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
        return 10
    ignore_keys = ['head.weight', 'head.bias']
    for k in ignore_keys:
        state_dict.pop(k, None)

def rename_key(dct, old, new):
    if False:
        return 10
    val = dct.pop(old)
    dct[new] = val

def prepare_img():
    if False:
        return 10
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    im = Image.open(requests.get(url, stream=True).raw)
    return im

@torch.no_grad()
def convert_vit_checkpoint(vit_name, pytorch_dump_folder_path, push_to_hub=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Copy/paste/tweak model's weights to our ViT structure.\n    "
    backbone_config = BitConfig(global_padding='same', layer_type='bottleneck', depths=(3, 4, 9), out_features=['stage3'], embedding_dynamic_padding=True)
    config = ViTHybridConfig(backbone_config=backbone_config, image_size=384, num_labels=1000)
    base_model = False
    timm_model = timm.create_model(vit_name, pretrained=True)
    timm_model.eval()
    state_dict = timm_model.state_dict()
    if base_model:
        remove_classification_head_(state_dict)
    rename_keys = create_rename_keys(config, base_model)
    for (src, dest) in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, config, base_model)
    repo_id = 'huggingface/label-files'
    filename = 'imagenet-1k-id2label.json'
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type='dataset'), 'r'))
    id2label = {int(k): v for (k, v) in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for (k, v) in id2label.items()}
    if vit_name[-5:] == 'in21k':
        model = ViTHybridModel(config).eval()
    else:
        model = ViTHybridForImageClassification(config).eval()
    model.load_state_dict(state_dict)
    transform = create_transform(**resolve_data_config({}, model=timm_model))
    timm_transforms = transform.transforms
    pillow_resamplings = {'bilinear': PILImageResampling.BILINEAR, 'bicubic': PILImageResampling.BICUBIC, 'nearest': PILImageResampling.NEAREST}
    processor = ViTHybridImageProcessor(do_resize=True, size={'shortest_edge': timm_transforms[0].size}, resample=pillow_resamplings[timm_transforms[0].interpolation.value], do_center_crop=True, crop_size={'height': timm_transforms[1].size[0], 'width': timm_transforms[1].size[1]}, do_normalize=True, image_mean=timm_transforms[-1].mean.tolist(), image_std=timm_transforms[-1].std.tolist())
    image = prepare_img()
    timm_pixel_values = transform(image).unsqueeze(0)
    pixel_values = processor(image, return_tensors='pt').pixel_values
    assert torch.allclose(timm_pixel_values, pixel_values)
    with torch.no_grad():
        outputs = model(pixel_values)
        logits = outputs.logits
    print('Predicted class:', logits.argmax(-1).item())
    if base_model:
        timm_pooled_output = timm_model.forward_features(pixel_values)
        assert timm_pooled_output.shape == outputs.pooler_output.shape
        assert torch.allclose(timm_pooled_output, outputs.pooler_output, atol=0.001)
    else:
        timm_logits = timm_model(pixel_values)
        assert timm_logits.shape == outputs.logits.shape
        assert torch.allclose(timm_logits, outputs.logits, atol=0.001)
    print('Looks ok!')
    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f'Saving model {vit_name} to {pytorch_dump_folder_path}')
        model.save_pretrained(pytorch_dump_folder_path)
        print(f'Saving processor to {pytorch_dump_folder_path}')
        processor.save_pretrained(pytorch_dump_folder_path)
    if push_to_hub:
        print(f'Pushing model and processor to the hub {vit_name}')
        model.push_to_hub(f'ybelkada/{vit_name}')
        processor.push_to_hub(f'ybelkada/{vit_name}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vit_name', default='vit_base_r50_s16_384', type=str, help="Name of the hybrid ViT timm model you'd like to convert.")
    parser.add_argument('--pytorch_dump_folder_path', default=None, type=str, help='Path to the output PyTorch model directory.')
    parser.add_argument('--push_to_hub', action='store_true', help='Whether to upload the model to the HuggingFace hub.')
    args = parser.parse_args()
    convert_vit_checkpoint(args.vit_name, args.pytorch_dump_folder_path, args.push_to_hub)