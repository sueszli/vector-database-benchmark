"""Convert CANINE checkpoint."""
import argparse
from transformers import CanineConfig, CanineModel, CanineTokenizer, load_tf_weights_in_canine
from transformers.utils import logging
logging.set_verbosity_info()

def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, pytorch_dump_path):
    if False:
        while True:
            i = 10
    config = CanineConfig()
    model = CanineModel(config)
    model.eval()
    print(f'Building PyTorch model from configuration: {config}')
    load_tf_weights_in_canine(model, config, tf_checkpoint_path)
    print(f'Save PyTorch model to {pytorch_dump_path}')
    model.save_pretrained(pytorch_dump_path)
    tokenizer = CanineTokenizer()
    print(f'Save tokenizer files to {pytorch_dump_path}')
    tokenizer.save_pretrained(pytorch_dump_path)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf_checkpoint_path', default=None, type=str, required=True, help='Path to the TensorFlow checkpoint. Should end with model.ckpt')
    parser.add_argument('--pytorch_dump_path', default=None, type=str, required=True, help='Path to a folder where the PyTorch model will be placed.')
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.pytorch_dump_path)