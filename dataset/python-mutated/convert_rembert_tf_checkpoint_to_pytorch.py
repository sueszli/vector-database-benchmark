"""Convert RemBERT checkpoint."""
import argparse
import torch
from transformers import RemBertConfig, RemBertModel, load_tf_weights_in_rembert
from transformers.utils import logging
logging.set_verbosity_info()

def convert_rembert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    if False:
        i = 10
        return i + 15
    config = RemBertConfig.from_json_file(bert_config_file)
    print('Building PyTorch model from configuration: {}'.format(str(config)))
    model = RemBertModel(config)
    load_tf_weights_in_rembert(model, config, tf_checkpoint_path)
    print('Save PyTorch model to {}'.format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf_checkpoint_path', default=None, type=str, required=True, help='Path to the TensorFlow checkpoint path.')
    parser.add_argument('--rembert_config_file', default=None, type=str, required=True, help='The config json file corresponding to the pre-trained RemBERT model. \nThis specifies the model architecture.')
    parser.add_argument('--pytorch_dump_path', default=None, type=str, required=True, help='Path to the output PyTorch model.')
    args = parser.parse_args()
    convert_rembert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.rembert_config_file, args.pytorch_dump_path)