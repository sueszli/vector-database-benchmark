"""Convert Pvt checkpoints from the original library."""
import argparse
from pathlib import Path
import requests
import torch
from PIL import Image
from transformers import PvtConfig, PvtForImageClassification, PvtImageProcessor
from transformers.utils import logging
logging.set_verbosity_info()
logger = logging.get_logger(__name__)

def create_rename_keys(config):
    if False:
        return 10
    rename_keys = []
    for i in range(config.num_encoder_blocks):
        rename_keys.append((f'pos_embed{i + 1}', f'pvt.encoder.patch_embeddings.{i}.position_embeddings'))
        rename_keys.append((f'patch_embed{i + 1}.proj.weight', f'pvt.encoder.patch_embeddings.{i}.projection.weight'))
        rename_keys.append((f'patch_embed{i + 1}.proj.bias', f'pvt.encoder.patch_embeddings.{i}.projection.bias'))
        rename_keys.append((f'patch_embed{i + 1}.norm.weight', f'pvt.encoder.patch_embeddings.{i}.layer_norm.weight'))
        rename_keys.append((f'patch_embed{i + 1}.norm.bias', f'pvt.encoder.patch_embeddings.{i}.layer_norm.bias'))
        for j in range(config.depths[i]):
            rename_keys.append((f'block{i + 1}.{j}.attn.q.weight', f'pvt.encoder.block.{i}.{j}.attention.self.query.weight'))
            rename_keys.append((f'block{i + 1}.{j}.attn.q.bias', f'pvt.encoder.block.{i}.{j}.attention.self.query.bias'))
            rename_keys.append((f'block{i + 1}.{j}.attn.kv.weight', f'pvt.encoder.block.{i}.{j}.attention.self.kv.weight'))
            rename_keys.append((f'block{i + 1}.{j}.attn.kv.bias', f'pvt.encoder.block.{i}.{j}.attention.self.kv.bias'))
            if config.sequence_reduction_ratios[i] > 1:
                rename_keys.append((f'block{i + 1}.{j}.attn.norm.weight', f'pvt.encoder.block.{i}.{j}.attention.self.layer_norm.weight'))
                rename_keys.append((f'block{i + 1}.{j}.attn.norm.bias', f'pvt.encoder.block.{i}.{j}.attention.self.layer_norm.bias'))
                rename_keys.append((f'block{i + 1}.{j}.attn.sr.weight', f'pvt.encoder.block.{i}.{j}.attention.self.sequence_reduction.weight'))
                rename_keys.append((f'block{i + 1}.{j}.attn.sr.bias', f'pvt.encoder.block.{i}.{j}.attention.self.sequence_reduction.bias'))
            rename_keys.append((f'block{i + 1}.{j}.attn.proj.weight', f'pvt.encoder.block.{i}.{j}.attention.output.dense.weight'))
            rename_keys.append((f'block{i + 1}.{j}.attn.proj.bias', f'pvt.encoder.block.{i}.{j}.attention.output.dense.bias'))
            rename_keys.append((f'block{i + 1}.{j}.norm1.weight', f'pvt.encoder.block.{i}.{j}.layer_norm_1.weight'))
            rename_keys.append((f'block{i + 1}.{j}.norm1.bias', f'pvt.encoder.block.{i}.{j}.layer_norm_1.bias'))
            rename_keys.append((f'block{i + 1}.{j}.norm2.weight', f'pvt.encoder.block.{i}.{j}.layer_norm_2.weight'))
            rename_keys.append((f'block{i + 1}.{j}.norm2.bias', f'pvt.encoder.block.{i}.{j}.layer_norm_2.bias'))
            rename_keys.append((f'block{i + 1}.{j}.mlp.fc1.weight', f'pvt.encoder.block.{i}.{j}.mlp.dense1.weight'))
            rename_keys.append((f'block{i + 1}.{j}.mlp.fc1.bias', f'pvt.encoder.block.{i}.{j}.mlp.dense1.bias'))
            rename_keys.append((f'block{i + 1}.{j}.mlp.fc2.weight', f'pvt.encoder.block.{i}.{j}.mlp.dense2.weight'))
            rename_keys.append((f'block{i + 1}.{j}.mlp.fc2.bias', f'pvt.encoder.block.{i}.{j}.mlp.dense2.bias'))
    rename_keys.extend([('cls_token', 'pvt.encoder.patch_embeddings.3.cls_token')])
    rename_keys.extend([('norm.weight', 'pvt.encoder.layer_norm.weight'), ('norm.bias', 'pvt.encoder.layer_norm.bias'), ('head.weight', 'classifier.weight'), ('head.bias', 'classifier.bias')])
    return rename_keys

def read_in_k_v(state_dict, config):
    if False:
        return 10
    for i in range(config.num_encoder_blocks):
        for j in range(config.depths[i]):
            kv_weight = state_dict.pop(f'pvt.encoder.block.{i}.{j}.attention.self.kv.weight')
            kv_bias = state_dict.pop(f'pvt.encoder.block.{i}.{j}.attention.self.kv.bias')
            state_dict[f'pvt.encoder.block.{i}.{j}.attention.self.key.weight'] = kv_weight[:config.hidden_sizes[i], :]
            state_dict[f'pvt.encoder.block.{i}.{j}.attention.self.key.bias'] = kv_bias[:config.hidden_sizes[i]]
            state_dict[f'pvt.encoder.block.{i}.{j}.attention.self.value.weight'] = kv_weight[config.hidden_sizes[i]:, :]
            state_dict[f'pvt.encoder.block.{i}.{j}.attention.self.value.bias'] = kv_bias[config.hidden_sizes[i]:]

def rename_key(dct, old, new):
    if False:
        i = 10
        return i + 15
    val = dct.pop(old)
    dct[new] = val

def prepare_img():
    if False:
        print('Hello World!')
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    im = Image.open(requests.get(url, stream=True).raw)
    return im

@torch.no_grad()
def convert_pvt_checkpoint(pvt_size, pvt_checkpoint, pytorch_dump_folder_path):
    if False:
        return 10
    "\n    Copy/paste/tweak model's weights to our PVT structure.\n    "
    if pvt_size == 'tiny':
        config_path = 'Zetatech/pvt-tiny-224'
    elif pvt_size == 'small':
        config_path = 'Zetatech/pvt-small-224'
    elif pvt_size == 'medium':
        config_path = 'Zetatech/pvt-medium-224'
    elif pvt_size == 'large':
        config_path = 'Zetatech/pvt-large-224'
    else:
        raise ValueError(f"Available model's size: 'tiny', 'small', 'medium', 'large', but '{pvt_size}' was given")
    config = PvtConfig(name_or_path=config_path)
    state_dict = torch.load(pvt_checkpoint, map_location='cpu')
    rename_keys = create_rename_keys(config)
    for (src, dest) in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_k_v(state_dict, config)
    model = PvtForImageClassification(config).eval()
    model.load_state_dict(state_dict)
    image_processor = PvtImageProcessor(size=config.image_size)
    encoding = image_processor(images=prepare_img(), return_tensors='pt')
    pixel_values = encoding['pixel_values']
    outputs = model(pixel_values)
    logits = outputs.logits.detach().cpu()
    if pvt_size == 'tiny':
        expected_slice_logits = torch.tensor([-1.4192, -1.9158, -0.9702])
    elif pvt_size == 'small':
        expected_slice_logits = torch.tensor([0.4353, -0.196, -0.2373])
    elif pvt_size == 'medium':
        expected_slice_logits = torch.tensor([-0.2914, -0.2231, 0.0321])
    elif pvt_size == 'large':
        expected_slice_logits = torch.tensor([0.374, -0.7739, -0.4214])
    else:
        raise ValueError(f"Available model's size: 'tiny', 'small', 'medium', 'large', but '{pvt_size}' was given")
    assert torch.allclose(logits[0, :3], expected_slice_logits, atol=0.0001)
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f'Saving model pytorch_model.bin to {pytorch_dump_folder_path}')
    model.save_pretrained(pytorch_dump_folder_path)
    print(f'Saving image processor to {pytorch_dump_folder_path}')
    image_processor.save_pretrained(pytorch_dump_folder_path)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pvt_size', default='tiny', type=str, help="Size of the PVT pretrained model you'd like to convert.")
    parser.add_argument('--pvt_checkpoint', default='pvt_tiny.pth', type=str, help="Checkpoint of the PVT pretrained model you'd like to convert.")
    parser.add_argument('--pytorch_dump_folder_path', default=None, type=str, help='Path to the output PyTorch model directory.')
    args = parser.parse_args()
    convert_pvt_checkpoint(args.pvt_size, args.pvt_checkpoint, args.pytorch_dump_folder_path)