"""Convert Funnel checkpoint."""
import argparse
import torch
from transformers import FunnelBaseModel, FunnelConfig, FunnelModel, load_tf_weights_in_funnel
from transformers.utils import logging
logging.set_verbosity_info()

def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, config_file, pytorch_dump_path, base_model):
    if False:
        while True:
            i = 10
    config = FunnelConfig.from_json_file(config_file)
    print(f'Building PyTorch model from configuration: {config}')
    model = FunnelBaseModel(config) if base_model else FunnelModel(config)
    load_tf_weights_in_funnel(model, config, tf_checkpoint_path)
    print(f'Save PyTorch model to {pytorch_dump_path}')
    torch.save(model.state_dict(), pytorch_dump_path)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf_checkpoint_path', default=None, type=str, required=True, help='Path to the TensorFlow checkpoint path.')
    parser.add_argument('--config_file', default=None, type=str, required=True, help='The config json file corresponding to the pre-trained model. \nThis specifies the model architecture.')
    parser.add_argument('--pytorch_dump_path', default=None, type=str, required=True, help='Path to the output PyTorch model.')
    parser.add_argument('--base_model', action='store_true', help='Whether you want just the base model (no decoder) or not.')
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.config_file, args.pytorch_dump_path, args.base_model)