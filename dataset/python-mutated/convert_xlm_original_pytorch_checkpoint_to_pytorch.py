"""Convert OpenAI GPT checkpoint."""
import argparse
import json
import numpy
import torch
from transformers.models.xlm.tokenization_xlm import VOCAB_FILES_NAMES
from transformers.utils import CONFIG_NAME, WEIGHTS_NAME, logging
logging.set_verbosity_info()

def convert_xlm_checkpoint_to_pytorch(xlm_checkpoint_path, pytorch_dump_folder_path):
    if False:
        while True:
            i = 10
    chkpt = torch.load(xlm_checkpoint_path, map_location='cpu')
    state_dict = chkpt['model']
    two_levels_state_dict = {}
    for (k, v) in state_dict.items():
        if 'pred_layer' in k:
            two_levels_state_dict[k] = v
        else:
            two_levels_state_dict['transformer.' + k] = v
    config = chkpt['params']
    config = {n: v for (n, v) in config.items() if not isinstance(v, (torch.FloatTensor, numpy.ndarray))}
    vocab = chkpt['dico_word2id']
    vocab = {s + '</w>' if s.find('@@') == -1 and i > 13 else s.replace('@@', ''): i for (s, i) in vocab.items()}
    pytorch_weights_dump_path = pytorch_dump_folder_path + '/' + WEIGHTS_NAME
    pytorch_config_dump_path = pytorch_dump_folder_path + '/' + CONFIG_NAME
    pytorch_vocab_dump_path = pytorch_dump_folder_path + '/' + VOCAB_FILES_NAMES['vocab_file']
    print(f'Save PyTorch model to {pytorch_weights_dump_path}')
    torch.save(two_levels_state_dict, pytorch_weights_dump_path)
    print(f'Save configuration file to {pytorch_config_dump_path}')
    with open(pytorch_config_dump_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(config, indent=2) + '\n')
    print(f'Save vocab file to {pytorch_config_dump_path}')
    with open(pytorch_vocab_dump_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(vocab, indent=2) + '\n')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--xlm_checkpoint_path', default=None, type=str, required=True, help='Path the official PyTorch dump.')
    parser.add_argument('--pytorch_dump_folder_path', default=None, type=str, required=True, help='Path to the output PyTorch model.')
    args = parser.parse_args()
    convert_xlm_checkpoint_to_pytorch(args.xlm_checkpoint_path, args.pytorch_dump_folder_path)