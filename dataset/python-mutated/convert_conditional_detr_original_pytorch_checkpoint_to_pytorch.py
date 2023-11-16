"""Convert Conditional DETR checkpoints."""
import argparse
import json
from collections import OrderedDict
from pathlib import Path
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import ConditionalDetrConfig, ConditionalDetrForObjectDetection, ConditionalDetrForSegmentation, ConditionalDetrImageProcessor
from transformers.utils import logging
logging.set_verbosity_info()
logger = logging.get_logger(__name__)
rename_keys = []
for i in range(6):
    rename_keys.append((f'transformer.encoder.layers.{i}.self_attn.out_proj.weight', f'encoder.layers.{i}.self_attn.out_proj.weight'))
    rename_keys.append((f'transformer.encoder.layers.{i}.self_attn.out_proj.bias', f'encoder.layers.{i}.self_attn.out_proj.bias'))
    rename_keys.append((f'transformer.encoder.layers.{i}.linear1.weight', f'encoder.layers.{i}.fc1.weight'))
    rename_keys.append((f'transformer.encoder.layers.{i}.linear1.bias', f'encoder.layers.{i}.fc1.bias'))
    rename_keys.append((f'transformer.encoder.layers.{i}.linear2.weight', f'encoder.layers.{i}.fc2.weight'))
    rename_keys.append((f'transformer.encoder.layers.{i}.linear2.bias', f'encoder.layers.{i}.fc2.bias'))
    rename_keys.append((f'transformer.encoder.layers.{i}.norm1.weight', f'encoder.layers.{i}.self_attn_layer_norm.weight'))
    rename_keys.append((f'transformer.encoder.layers.{i}.norm1.bias', f'encoder.layers.{i}.self_attn_layer_norm.bias'))
    rename_keys.append((f'transformer.encoder.layers.{i}.norm2.weight', f'encoder.layers.{i}.final_layer_norm.weight'))
    rename_keys.append((f'transformer.encoder.layers.{i}.norm2.bias', f'encoder.layers.{i}.final_layer_norm.bias'))
    rename_keys.append((f'transformer.decoder.layers.{i}.self_attn.out_proj.weight', f'decoder.layers.{i}.self_attn.out_proj.weight'))
    rename_keys.append((f'transformer.decoder.layers.{i}.self_attn.out_proj.bias', f'decoder.layers.{i}.self_attn.out_proj.bias'))
    rename_keys.append((f'transformer.decoder.layers.{i}.cross_attn.out_proj.weight', f'decoder.layers.{i}.encoder_attn.out_proj.weight'))
    rename_keys.append((f'transformer.decoder.layers.{i}.cross_attn.out_proj.bias', f'decoder.layers.{i}.encoder_attn.out_proj.bias'))
    rename_keys.append((f'transformer.decoder.layers.{i}.linear1.weight', f'decoder.layers.{i}.fc1.weight'))
    rename_keys.append((f'transformer.decoder.layers.{i}.linear1.bias', f'decoder.layers.{i}.fc1.bias'))
    rename_keys.append((f'transformer.decoder.layers.{i}.linear2.weight', f'decoder.layers.{i}.fc2.weight'))
    rename_keys.append((f'transformer.decoder.layers.{i}.linear2.bias', f'decoder.layers.{i}.fc2.bias'))
    rename_keys.append((f'transformer.decoder.layers.{i}.norm1.weight', f'decoder.layers.{i}.self_attn_layer_norm.weight'))
    rename_keys.append((f'transformer.decoder.layers.{i}.norm1.bias', f'decoder.layers.{i}.self_attn_layer_norm.bias'))
    rename_keys.append((f'transformer.decoder.layers.{i}.norm2.weight', f'decoder.layers.{i}.encoder_attn_layer_norm.weight'))
    rename_keys.append((f'transformer.decoder.layers.{i}.norm2.bias', f'decoder.layers.{i}.encoder_attn_layer_norm.bias'))
    rename_keys.append((f'transformer.decoder.layers.{i}.norm3.weight', f'decoder.layers.{i}.final_layer_norm.weight'))
    rename_keys.append((f'transformer.decoder.layers.{i}.norm3.bias', f'decoder.layers.{i}.final_layer_norm.bias'))
    rename_keys.append((f'transformer.decoder.layers.{i}.sa_qcontent_proj.weight', f'decoder.layers.{i}.sa_qcontent_proj.weight'))
    rename_keys.append((f'transformer.decoder.layers.{i}.sa_kcontent_proj.weight', f'decoder.layers.{i}.sa_kcontent_proj.weight'))
    rename_keys.append((f'transformer.decoder.layers.{i}.sa_qpos_proj.weight', f'decoder.layers.{i}.sa_qpos_proj.weight'))
    rename_keys.append((f'transformer.decoder.layers.{i}.sa_kpos_proj.weight', f'decoder.layers.{i}.sa_kpos_proj.weight'))
    rename_keys.append((f'transformer.decoder.layers.{i}.sa_v_proj.weight', f'decoder.layers.{i}.sa_v_proj.weight'))
    rename_keys.append((f'transformer.decoder.layers.{i}.ca_qcontent_proj.weight', f'decoder.layers.{i}.ca_qcontent_proj.weight'))
    rename_keys.append((f'transformer.decoder.layers.{i}.ca_kcontent_proj.weight', f'decoder.layers.{i}.ca_kcontent_proj.weight'))
    rename_keys.append((f'transformer.decoder.layers.{i}.ca_kpos_proj.weight', f'decoder.layers.{i}.ca_kpos_proj.weight'))
    rename_keys.append((f'transformer.decoder.layers.{i}.ca_v_proj.weight', f'decoder.layers.{i}.ca_v_proj.weight'))
    rename_keys.append((f'transformer.decoder.layers.{i}.ca_qpos_sine_proj.weight', f'decoder.layers.{i}.ca_qpos_sine_proj.weight'))
    rename_keys.append((f'transformer.decoder.layers.{i}.sa_qcontent_proj.bias', f'decoder.layers.{i}.sa_qcontent_proj.bias'))
    rename_keys.append((f'transformer.decoder.layers.{i}.sa_kcontent_proj.bias', f'decoder.layers.{i}.sa_kcontent_proj.bias'))
    rename_keys.append((f'transformer.decoder.layers.{i}.sa_qpos_proj.bias', f'decoder.layers.{i}.sa_qpos_proj.bias'))
    rename_keys.append((f'transformer.decoder.layers.{i}.sa_kpos_proj.bias', f'decoder.layers.{i}.sa_kpos_proj.bias'))
    rename_keys.append((f'transformer.decoder.layers.{i}.sa_v_proj.bias', f'decoder.layers.{i}.sa_v_proj.bias'))
    rename_keys.append((f'transformer.decoder.layers.{i}.ca_qcontent_proj.bias', f'decoder.layers.{i}.ca_qcontent_proj.bias'))
    rename_keys.append((f'transformer.decoder.layers.{i}.ca_kcontent_proj.bias', f'decoder.layers.{i}.ca_kcontent_proj.bias'))
    rename_keys.append((f'transformer.decoder.layers.{i}.ca_kpos_proj.bias', f'decoder.layers.{i}.ca_kpos_proj.bias'))
    rename_keys.append((f'transformer.decoder.layers.{i}.ca_v_proj.bias', f'decoder.layers.{i}.ca_v_proj.bias'))
    rename_keys.append((f'transformer.decoder.layers.{i}.ca_qpos_sine_proj.bias', f'decoder.layers.{i}.ca_qpos_sine_proj.bias'))
rename_keys.extend([('input_proj.weight', 'input_projection.weight'), ('input_proj.bias', 'input_projection.bias'), ('query_embed.weight', 'query_position_embeddings.weight'), ('transformer.decoder.norm.weight', 'decoder.layernorm.weight'), ('transformer.decoder.norm.bias', 'decoder.layernorm.bias'), ('class_embed.weight', 'class_labels_classifier.weight'), ('class_embed.bias', 'class_labels_classifier.bias'), ('bbox_embed.layers.0.weight', 'bbox_predictor.layers.0.weight'), ('bbox_embed.layers.0.bias', 'bbox_predictor.layers.0.bias'), ('bbox_embed.layers.1.weight', 'bbox_predictor.layers.1.weight'), ('bbox_embed.layers.1.bias', 'bbox_predictor.layers.1.bias'), ('bbox_embed.layers.2.weight', 'bbox_predictor.layers.2.weight'), ('bbox_embed.layers.2.bias', 'bbox_predictor.layers.2.bias'), ('transformer.decoder.ref_point_head.layers.0.weight', 'decoder.ref_point_head.layers.0.weight'), ('transformer.decoder.ref_point_head.layers.0.bias', 'decoder.ref_point_head.layers.0.bias'), ('transformer.decoder.ref_point_head.layers.1.weight', 'decoder.ref_point_head.layers.1.weight'), ('transformer.decoder.ref_point_head.layers.1.bias', 'decoder.ref_point_head.layers.1.bias'), ('transformer.decoder.query_scale.layers.0.weight', 'decoder.query_scale.layers.0.weight'), ('transformer.decoder.query_scale.layers.0.bias', 'decoder.query_scale.layers.0.bias'), ('transformer.decoder.query_scale.layers.1.weight', 'decoder.query_scale.layers.1.weight'), ('transformer.decoder.query_scale.layers.1.bias', 'decoder.query_scale.layers.1.bias'), ('transformer.decoder.layers.0.ca_qpos_proj.weight', 'decoder.layers.0.ca_qpos_proj.weight'), ('transformer.decoder.layers.0.ca_qpos_proj.bias', 'decoder.layers.0.ca_qpos_proj.bias')])

def rename_key(state_dict, old, new):
    if False:
        i = 10
        return i + 15
    val = state_dict.pop(old)
    state_dict[new] = val

def rename_backbone_keys(state_dict):
    if False:
        print('Hello World!')
    new_state_dict = OrderedDict()
    for (key, value) in state_dict.items():
        if 'backbone.0.body' in key:
            new_key = key.replace('backbone.0.body', 'backbone.conv_encoder.model')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

def read_in_q_k_v(state_dict, is_panoptic=False):
    if False:
        for i in range(10):
            print('nop')
    prefix = ''
    if is_panoptic:
        prefix = 'conditional_detr.'
    for i in range(6):
        in_proj_weight = state_dict.pop(f'{prefix}transformer.encoder.layers.{i}.self_attn.in_proj_weight')
        in_proj_bias = state_dict.pop(f'{prefix}transformer.encoder.layers.{i}.self_attn.in_proj_bias')
        state_dict[f'encoder.layers.{i}.self_attn.q_proj.weight'] = in_proj_weight[:256, :]
        state_dict[f'encoder.layers.{i}.self_attn.q_proj.bias'] = in_proj_bias[:256]
        state_dict[f'encoder.layers.{i}.self_attn.k_proj.weight'] = in_proj_weight[256:512, :]
        state_dict[f'encoder.layers.{i}.self_attn.k_proj.bias'] = in_proj_bias[256:512]
        state_dict[f'encoder.layers.{i}.self_attn.v_proj.weight'] = in_proj_weight[-256:, :]
        state_dict[f'encoder.layers.{i}.self_attn.v_proj.bias'] = in_proj_bias[-256:]

def prepare_img():
    if False:
        return 10
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    im = Image.open(requests.get(url, stream=True).raw)
    return im

@torch.no_grad()
def convert_conditional_detr_checkpoint(model_name, pytorch_dump_folder_path):
    if False:
        for i in range(10):
            print('nop')
    "\n    Copy/paste/tweak model's weights to our CONDITIONAL_DETR structure.\n    "
    config = ConditionalDetrConfig()
    if 'resnet101' in model_name:
        config.backbone = 'resnet101'
    if 'dc5' in model_name:
        config.dilation = True
    is_panoptic = 'panoptic' in model_name
    if is_panoptic:
        config.num_labels = 250
    else:
        config.num_labels = 91
        repo_id = 'huggingface/label-files'
        filename = 'coco-detection-id2label.json'
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type='dataset'), 'r'))
        id2label = {int(k): v for (k, v) in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for (k, v) in id2label.items()}
    format = 'coco_panoptic' if is_panoptic else 'coco_detection'
    image_processor = ConditionalDetrImageProcessor(format=format)
    img = prepare_img()
    encoding = image_processor(images=img, return_tensors='pt')
    pixel_values = encoding['pixel_values']
    logger.info(f'Converting model {model_name}...')
    conditional_detr = torch.hub.load('DeppMeng/ConditionalDETR', model_name, pretrained=True).eval()
    state_dict = conditional_detr.state_dict()
    for (src, dest) in rename_keys:
        if is_panoptic:
            src = 'conditional_detr.' + src
        rename_key(state_dict, src, dest)
    state_dict = rename_backbone_keys(state_dict)
    read_in_q_k_v(state_dict, is_panoptic=is_panoptic)
    prefix = 'conditional_detr.model.' if is_panoptic else 'model.'
    for key in state_dict.copy().keys():
        if is_panoptic:
            if key.startswith('conditional_detr') and (not key.startswith('class_labels_classifier')) and (not key.startswith('bbox_predictor')):
                val = state_dict.pop(key)
                state_dict['conditional_detr.model' + key[4:]] = val
            elif 'class_labels_classifier' in key or 'bbox_predictor' in key:
                val = state_dict.pop(key)
                state_dict['conditional_detr.' + key] = val
            elif key.startswith('bbox_attention') or key.startswith('mask_head'):
                continue
            else:
                val = state_dict.pop(key)
                state_dict[prefix + key] = val
        elif not key.startswith('class_labels_classifier') and (not key.startswith('bbox_predictor')):
            val = state_dict.pop(key)
            state_dict[prefix + key] = val
    model = ConditionalDetrForSegmentation(config) if is_panoptic else ConditionalDetrForObjectDetection(config)
    model.load_state_dict(state_dict)
    model.eval()
    model.push_to_hub(repo_id=model_name, organization='DepuMeng', commit_message='Add model')
    original_outputs = conditional_detr(pixel_values)
    outputs = model(pixel_values)
    assert torch.allclose(outputs.logits, original_outputs['pred_logits'], atol=0.0001)
    assert torch.allclose(outputs.pred_boxes, original_outputs['pred_boxes'], atol=0.0001)
    if is_panoptic:
        assert torch.allclose(outputs.pred_masks, original_outputs['pred_masks'], atol=0.0001)
    logger.info(f'Saving PyTorch model and image processor to {pytorch_dump_folder_path}...')
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)
    image_processor.save_pretrained(pytorch_dump_folder_path)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='conditional_detr_resnet50', type=str, help="Name of the CONDITIONAL_DETR model you'd like to convert.")
    parser.add_argument('--pytorch_dump_folder_path', default=None, type=str, help='Path to the folder to output PyTorch model.')
    args = parser.parse_args()
    convert_conditional_detr_checkpoint(args.model_name, args.pytorch_dump_folder_path)