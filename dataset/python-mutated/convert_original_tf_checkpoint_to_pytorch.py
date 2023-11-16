"""Convert MobileNetV1 checkpoints from the tensorflow/models library."""
import argparse
import json
import re
from pathlib import Path
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import MobileNetV1Config, MobileNetV1ForImageClassification, MobileNetV1ImageProcessor, load_tf_weights_in_mobilenet_v1
from transformers.utils import logging
logging.set_verbosity_info()
logger = logging.get_logger(__name__)

def get_mobilenet_v1_config(model_name):
    if False:
        print('Hello World!')
    config = MobileNetV1Config(layer_norm_eps=0.001)
    if '_quant' in model_name:
        raise ValueError('Quantized models are not supported.')
    matches = re.match('^mobilenet_v1_([^_]*)_([^_]*)$', model_name)
    if matches:
        config.depth_multiplier = float(matches[1])
        config.image_size = int(matches[2])
    config.num_labels = 1001
    filename = 'imagenet-1k-id2label.json'
    repo_id = 'huggingface/label-files'
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type='dataset'), 'r'))
    id2label = {int(k) + 1: v for (k, v) in id2label.items()}
    id2label[0] = 'background'
    config.id2label = id2label
    config.label2id = {v: k for (k, v) in id2label.items()}
    return config

def prepare_img():
    if False:
        i = 10
        return i + 15
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    im = Image.open(requests.get(url, stream=True).raw)
    return im

@torch.no_grad()
def convert_movilevit_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path, push_to_hub=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Copy/paste/tweak model's weights to our MobileNetV1 structure.\n    "
    config = get_mobilenet_v1_config(model_name)
    model = MobileNetV1ForImageClassification(config).eval()
    load_tf_weights_in_mobilenet_v1(model, config, checkpoint_path)
    image_processor = MobileNetV1ImageProcessor(crop_size={'width': config.image_size, 'height': config.image_size}, size={'shortest_edge': config.image_size + 32})
    encoding = image_processor(images=prepare_img(), return_tensors='pt')
    outputs = model(**encoding)
    logits = outputs.logits
    assert logits.shape == (1, 1001)
    if model_name == 'mobilenet_v1_1.0_224':
        expected_logits = torch.tensor([-4.1739, -1.1233, 3.1205])
    elif model_name == 'mobilenet_v1_0.75_192':
        expected_logits = torch.tensor([-3.944, -2.3141, -0.3333])
    else:
        expected_logits = None
    if expected_logits is not None:
        assert torch.allclose(logits[0, :3], expected_logits, atol=0.0001)
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f'Saving model {model_name} to {pytorch_dump_folder_path}')
    model.save_pretrained(pytorch_dump_folder_path)
    print(f'Saving image processor to {pytorch_dump_folder_path}')
    image_processor.save_pretrained(pytorch_dump_folder_path)
    if push_to_hub:
        print('Pushing to the hub...')
        repo_id = 'google/' + model_name
        image_processor.push_to_hub(repo_id)
        model.push_to_hub(repo_id)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='mobilenet_v1_1.0_224', type=str, help="Name of the MobileNetV1 model you'd like to convert. Should in the form 'mobilenet_v1_<depth>_<size>'.")
    parser.add_argument('--checkpoint_path', required=True, type=str, help='Path to the original TensorFlow checkpoint (.ckpt file).')
    parser.add_argument('--pytorch_dump_folder_path', required=True, type=str, help='Path to the output PyTorch model directory.')
    parser.add_argument('--push_to_hub', action='store_true', help='Whether or not to push the converted model to the 🤗 hub.')
    args = parser.parse_args()
    convert_movilevit_checkpoint(args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub)