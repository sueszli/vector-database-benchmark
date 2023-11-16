"""Convert T5 checkpoint."""
import argparse
from transformers import T5Config, T5ForConditionalGeneration, load_tf_weights_in_t5
from transformers.utils import logging
logging.set_verbosity_info()

def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, config_file, pytorch_dump_path):
    if False:
        return 10
    config = T5Config.from_json_file(config_file)
    print(f'Building PyTorch model from configuration: {config}')
    model = T5ForConditionalGeneration(config)
    load_tf_weights_in_t5(model, config, tf_checkpoint_path)
    print(f'Save PyTorch model to {pytorch_dump_path}')
    model.save_pretrained(pytorch_dump_path)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf_checkpoint_path', default=None, type=str, required=True, help='Path to the TensorFlow checkpoint path.')
    parser.add_argument('--config_file', default=None, type=str, required=True, help='The config json file corresponding to the pre-trained T5 model. \nThis specifies the model architecture.')
    parser.add_argument('--pytorch_dump_path', default=None, type=str, required=True, help='Path to the output PyTorch model.')
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.config_file, args.pytorch_dump_path)