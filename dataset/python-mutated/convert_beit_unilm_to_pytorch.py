"""Convert BEiT checkpoints from the unilm repository."""
import argparse
import json
from pathlib import Path
import requests
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import BeitConfig, BeitForImageClassification, BeitForMaskedImageModeling, BeitForSemanticSegmentation, BeitImageProcessor
from transformers.image_utils import PILImageResampling
from transformers.utils import logging
logging.set_verbosity_info()
logger = logging.get_logger(__name__)

def create_rename_keys(config, has_lm_head=False, is_semantic=False):
    if False:
        return 10
    prefix = 'backbone.' if is_semantic else ''
    rename_keys = []
    for i in range(config.num_hidden_layers):
        rename_keys.append((f'{prefix}blocks.{i}.norm1.weight', f'beit.encoder.layer.{i}.layernorm_before.weight'))
        rename_keys.append((f'{prefix}blocks.{i}.norm1.bias', f'beit.encoder.layer.{i}.layernorm_before.bias'))
        rename_keys.append((f'{prefix}blocks.{i}.attn.proj.weight', f'beit.encoder.layer.{i}.attention.output.dense.weight'))
        rename_keys.append((f'{prefix}blocks.{i}.attn.proj.bias', f'beit.encoder.layer.{i}.attention.output.dense.bias'))
        rename_keys.append((f'{prefix}blocks.{i}.norm2.weight', f'beit.encoder.layer.{i}.layernorm_after.weight'))
        rename_keys.append((f'{prefix}blocks.{i}.norm2.bias', f'beit.encoder.layer.{i}.layernorm_after.bias'))
        rename_keys.append((f'{prefix}blocks.{i}.mlp.fc1.weight', f'beit.encoder.layer.{i}.intermediate.dense.weight'))
        rename_keys.append((f'{prefix}blocks.{i}.mlp.fc1.bias', f'beit.encoder.layer.{i}.intermediate.dense.bias'))
        rename_keys.append((f'{prefix}blocks.{i}.mlp.fc2.weight', f'beit.encoder.layer.{i}.output.dense.weight'))
        rename_keys.append((f'{prefix}blocks.{i}.mlp.fc2.bias', f'beit.encoder.layer.{i}.output.dense.bias'))
    rename_keys.extend([(f'{prefix}cls_token', 'beit.embeddings.cls_token'), (f'{prefix}patch_embed.proj.weight', 'beit.embeddings.patch_embeddings.projection.weight'), (f'{prefix}patch_embed.proj.bias', 'beit.embeddings.patch_embeddings.projection.bias')])
    if has_lm_head:
        rename_keys.extend([('mask_token', 'beit.embeddings.mask_token'), ('rel_pos_bias.relative_position_bias_table', 'beit.encoder.relative_position_bias.relative_position_bias_table'), ('rel_pos_bias.relative_position_index', 'beit.encoder.relative_position_bias.relative_position_index'), ('norm.weight', 'layernorm.weight'), ('norm.bias', 'layernorm.bias')])
    elif is_semantic:
        rename_keys.extend([('decode_head.conv_seg.weight', 'decode_head.classifier.weight'), ('decode_head.conv_seg.bias', 'decode_head.classifier.bias'), ('auxiliary_head.conv_seg.weight', 'auxiliary_head.classifier.weight'), ('auxiliary_head.conv_seg.bias', 'auxiliary_head.classifier.bias')])
    else:
        rename_keys.extend([('fc_norm.weight', 'beit.pooler.layernorm.weight'), ('fc_norm.bias', 'beit.pooler.layernorm.bias'), ('head.weight', 'classifier.weight'), ('head.bias', 'classifier.bias')])
    return rename_keys

def read_in_q_k_v(state_dict, config, has_lm_head=False, is_semantic=False):
    if False:
        while True:
            i = 10
    for i in range(config.num_hidden_layers):
        prefix = 'backbone.' if is_semantic else ''
        in_proj_weight = state_dict.pop(f'{prefix}blocks.{i}.attn.qkv.weight')
        q_bias = state_dict.pop(f'{prefix}blocks.{i}.attn.q_bias')
        v_bias = state_dict.pop(f'{prefix}blocks.{i}.attn.v_bias')
        state_dict[f'beit.encoder.layer.{i}.attention.attention.query.weight'] = in_proj_weight[:config.hidden_size, :]
        state_dict[f'beit.encoder.layer.{i}.attention.attention.query.bias'] = q_bias
        state_dict[f'beit.encoder.layer.{i}.attention.attention.key.weight'] = in_proj_weight[config.hidden_size:config.hidden_size * 2, :]
        state_dict[f'beit.encoder.layer.{i}.attention.attention.value.weight'] = in_proj_weight[-config.hidden_size:, :]
        state_dict[f'beit.encoder.layer.{i}.attention.attention.value.bias'] = v_bias
        gamma_1 = state_dict.pop(f'{prefix}blocks.{i}.gamma_1')
        gamma_2 = state_dict.pop(f'{prefix}blocks.{i}.gamma_2')
        state_dict[f'beit.encoder.layer.{i}.lambda_1'] = gamma_1
        state_dict[f'beit.encoder.layer.{i}.lambda_2'] = gamma_2
        if not has_lm_head:
            table = state_dict.pop(f'{prefix}blocks.{i}.attn.relative_position_bias_table')
            index = state_dict.pop(f'{prefix}blocks.{i}.attn.relative_position_index')
            state_dict[f'beit.encoder.layer.{i}.attention.attention.relative_position_bias.relative_position_bias_table'] = table
            state_dict[f'beit.encoder.layer.{i}.attention.attention.relative_position_bias.relative_position_index'] = index

def rename_key(dct, old, new):
    if False:
        for i in range(10):
            print('nop')
    val = dct.pop(old)
    dct[new] = val

def prepare_img():
    if False:
        for i in range(10):
            print('nop')
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    im = Image.open(requests.get(url, stream=True).raw)
    return im

@torch.no_grad()
def convert_beit_checkpoint(checkpoint_url, pytorch_dump_folder_path):
    if False:
        while True:
            i = 10
    "\n    Copy/paste/tweak model's weights to our BEiT structure.\n    "
    config = BeitConfig()
    has_lm_head = False
    is_semantic = False
    repo_id = 'huggingface/label-files'
    if checkpoint_url[-9:-4] == 'pt22k':
        config.use_shared_relative_position_bias = True
        config.use_mask_token = True
        has_lm_head = True
    elif checkpoint_url[-9:-4] == 'ft22k':
        config.use_relative_position_bias = True
        config.num_labels = 21841
        filename = 'imagenet-22k-id2label.json'
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type='dataset'), 'r'))
        id2label = {int(k): v for (k, v) in id2label.items()}
        del id2label[9205]
        del id2label[15027]
        config.id2label = id2label
        config.label2id = {v: k for (k, v) in id2label.items()}
    elif checkpoint_url[-8:-4] == 'to1k':
        config.use_relative_position_bias = True
        config.num_labels = 1000
        filename = 'imagenet-1k-id2label.json'
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type='dataset'), 'r'))
        id2label = {int(k): v for (k, v) in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for (k, v) in id2label.items()}
        if '384' in checkpoint_url:
            config.image_size = 384
        if '512' in checkpoint_url:
            config.image_size = 512
    elif 'ade20k' in checkpoint_url:
        config.use_relative_position_bias = True
        config.num_labels = 150
        filename = 'ade20k-id2label.json'
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type='dataset'), 'r'))
        id2label = {int(k): v for (k, v) in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for (k, v) in id2label.items()}
        config.image_size = 640
        is_semantic = True
    else:
        raise ValueError("Checkpoint not supported, URL should either end with 'pt22k', 'ft22k', 'to1k' or 'ade20k'")
    if 'base' in checkpoint_url:
        pass
    elif 'large' in checkpoint_url:
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 24
        config.num_attention_heads = 16
        if 'ade20k' in checkpoint_url:
            config.image_size = 640
            config.out_indices = [7, 11, 15, 23]
    else:
        raise ValueError("Should either find 'base' or 'large' in checkpoint URL")
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location='cpu', check_hash=True)
    state_dict = state_dict['model'] if 'ade20k' not in checkpoint_url else state_dict['state_dict']
    rename_keys = create_rename_keys(config, has_lm_head=has_lm_head, is_semantic=is_semantic)
    for (src, dest) in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, config, has_lm_head=has_lm_head, is_semantic=is_semantic)
    if is_semantic:
        for (key, val) in state_dict.copy().items():
            val = state_dict.pop(key)
            if key.startswith('backbone.fpn'):
                key = key.replace('backbone.fpn', 'fpn')
            state_dict[key] = val
    if checkpoint_url[-9:-4] == 'pt22k':
        model = BeitForMaskedImageModeling(config)
    elif 'ade20k' in checkpoint_url:
        model = BeitForSemanticSegmentation(config)
    else:
        model = BeitForImageClassification(config)
    model.eval()
    model.load_state_dict(state_dict)
    if is_semantic:
        image_processor = BeitImageProcessor(size=config.image_size, do_center_crop=False)
        ds = load_dataset('hf-internal-testing/fixtures_ade20k', split='test')
        image = Image.open(ds[0]['file'])
    else:
        image_processor = BeitImageProcessor(size=config.image_size, resample=PILImageResampling.BILINEAR, do_center_crop=False)
        image = prepare_img()
    encoding = image_processor(images=image, return_tensors='pt')
    pixel_values = encoding['pixel_values']
    outputs = model(pixel_values)
    logits = outputs.logits
    expected_shape = torch.Size([1, 1000])
    if checkpoint_url[:-4].endswith('beit_base_patch16_224_pt22k'):
        expected_shape = torch.Size([1, 196, 8192])
    elif checkpoint_url[:-4].endswith('beit_large_patch16_224_pt22k'):
        expected_shape = torch.Size([1, 196, 8192])
    elif checkpoint_url[:-4].endswith('beit_base_patch16_224_pt22k_ft22k'):
        expected_shape = torch.Size([1, 21841])
        expected_logits = torch.tensor([2.2288, 2.4671, 0.7395])
        expected_class_idx = 2397
    elif checkpoint_url[:-4].endswith('beit_large_patch16_224_pt22k_ft22k'):
        expected_shape = torch.Size([1, 21841])
        expected_logits = torch.tensor([1.6881, -0.2787, 0.5901])
        expected_class_idx = 2396
    elif checkpoint_url[:-4].endswith('beit_base_patch16_224_pt22k_ft1k'):
        expected_logits = torch.tensor([0.1241, 0.0798, -0.6569])
        expected_class_idx = 285
    elif checkpoint_url[:-4].endswith('beit_base_patch16_224_pt22k_ft22kto1k'):
        expected_logits = torch.tensor([-1.2385, -1.0987, -1.0108])
        expected_class_idx = 281
    elif checkpoint_url[:-4].endswith('beit_base_patch16_384_pt22k_ft22kto1k'):
        expected_logits = torch.tensor([-1.5303, -0.9484, -0.3147])
        expected_class_idx = 761
    elif checkpoint_url[:-4].endswith('beit_large_patch16_224_pt22k_ft1k'):
        expected_logits = torch.tensor([0.461, -0.0928, 0.2086])
        expected_class_idx = 761
    elif checkpoint_url[:-4].endswith('beit_large_patch16_224_pt22k_ft22kto1k'):
        expected_logits = torch.tensor([-0.4804, 0.6257, -0.1837])
        expected_class_idx = 761
    elif checkpoint_url[:-4].endswith('beit_large_patch16_384_pt22k_ft22kto1k'):
        expected_logits = torch.tensor([[-0.5122, 0.5117, -0.2113]])
        expected_class_idx = 761
    elif checkpoint_url[:-4].endswith('beit_large_patch16_512_pt22k_ft22kto1k'):
        expected_logits = torch.tensor([-0.3062, 0.7261, 0.4852])
        expected_class_idx = 761
    elif checkpoint_url[:-4].endswith('beit_base_patch16_640_pt22k_ft22ktoade20k'):
        expected_shape = (1, 150, 160, 160)
        expected_logits = torch.tensor([[[-4.9225, -2.3954, -3.0522], [-2.8822, -1.0046, -1.7561], [-2.9549, -1.3228, -2.1347]], [[-5.8168, -3.4129, -4.0778], [-3.8651, -2.2214, -3.0277], [-3.8356, -2.4643, -3.3535]], [[-0.0078, 3.9952, 4.0754], [2.9856, 4.6944, 5.0035], [3.2413, 4.7813, 4.9969]]])
    elif checkpoint_url[:-4].endswith('beit_large_patch16_640_pt22k_ft22ktoade20k'):
        expected_shape = (1, 150, 160, 160)
        expected_logits = torch.tensor([[[-4.3305, -2.3049, -3.0161], [-2.9591, -1.5305, -2.2251], [-3.4198, -1.8004, -2.9062]], [[-5.8922, -3.7435, -4.3978], [-4.2063, -2.7872, -3.4755], [-4.2791, -3.1874, -4.1681]], [[0.9895, 4.3467, 4.7663], [4.2476, 5.683, 6.1518], [4.555, 6.2495, 6.5154]]])
    else:
        raise ValueError("Can't verify logits as model is not supported")
    if logits.shape != expected_shape:
        raise ValueError(f'Shape of logits not as expected. logits.shape={logits.shape!r}, expected_shape={expected_shape!r}')
    if not has_lm_head:
        if is_semantic:
            if not torch.allclose(logits[0, :3, :3, :3], expected_logits, atol=0.001):
                raise ValueError('First elements of logits not as expected')
        else:
            print('Predicted class idx:', logits.argmax(-1).item())
            if not torch.allclose(logits[0, :3], expected_logits, atol=0.001):
                raise ValueError('First elements of logits not as expected')
            if logits.argmax(-1).item() != expected_class_idx:
                raise ValueError('Predicted class index not as expected')
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f'Saving model to {pytorch_dump_folder_path}')
    model.save_pretrained(pytorch_dump_folder_path)
    print(f'Saving image processor to {pytorch_dump_folder_path}')
    image_processor.save_pretrained(pytorch_dump_folder_path)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_url', default='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_224_pt22k_ft22kto1k.pth', type=str, help='URL to the original PyTorch checkpoint (.pth file).')
    parser.add_argument('--pytorch_dump_folder_path', default=None, type=str, help='Path to the folder to output PyTorch model.')
    args = parser.parse_args()
    convert_beit_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path)