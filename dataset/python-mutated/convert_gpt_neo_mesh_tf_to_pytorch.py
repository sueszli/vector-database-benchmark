"""Convert GPT Neo checkpoint."""
import argparse
import json
from transformers import GPTNeoConfig, GPTNeoForCausalLM, load_tf_weights_in_gpt_neo
from transformers.utils import logging
logging.set_verbosity_info()

def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, config_file, pytorch_dump_path):
    if False:
        while True:
            i = 10
    config_json = json.load(open(config_file, 'r'))
    config = GPTNeoConfig(hidden_size=config_json['n_embd'], num_layers=config_json['n_layer'], num_heads=config_json['n_head'], attention_types=config_json['attention_types'], max_position_embeddings=config_json['n_positions'], resid_dropout=config_json['res_dropout'], embed_dropout=config_json['embed_dropout'], attention_dropout=config_json['attn_dropout'])
    print(f'Building PyTorch model from configuration: {config}')
    model = GPTNeoForCausalLM(config)
    load_tf_weights_in_gpt_neo(model, config, tf_checkpoint_path)
    print(f'Save PyTorch model to {pytorch_dump_path}')
    model.save_pretrained(pytorch_dump_path)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf_checkpoint_path', default=None, type=str, required=True, help='Path to the TensorFlow checkpoint path.')
    parser.add_argument('--config_file', default=None, type=str, required=True, help='The config json file corresponding to the pre-trained mesh-tf model. \nThis specifies the model architecture.')
    parser.add_argument('--pytorch_dump_path', default=None, type=str, required=True, help='Path to the output PyTorch model.')
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.config_file, args.pytorch_dump_path)