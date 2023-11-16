"""Convert Transformer XL checkpoint and datasets."""
import argparse
import os
import pickle
import sys
import torch
from transformers import TransfoXLConfig, TransfoXLLMHeadModel, load_tf_weights_in_transfo_xl
from transformers.models.transfo_xl import tokenization_transfo_xl as data_utils
from transformers.models.transfo_xl.tokenization_transfo_xl import CORPUS_NAME, VOCAB_FILES_NAMES
from transformers.utils import CONFIG_NAME, WEIGHTS_NAME, logging
logging.set_verbosity_info()
data_utils.Vocab = data_utils.TransfoXLTokenizer
data_utils.Corpus = data_utils.TransfoXLCorpus
sys.modules['data_utils'] = data_utils
sys.modules['vocabulary'] = data_utils

def convert_transfo_xl_checkpoint_to_pytorch(tf_checkpoint_path, transfo_xl_config_file, pytorch_dump_folder_path, transfo_xl_dataset_file):
    if False:
        while True:
            i = 10
    if transfo_xl_dataset_file:
        with open(transfo_xl_dataset_file, 'rb') as fp:
            corpus = pickle.load(fp, encoding='latin1')
        pytorch_vocab_dump_path = pytorch_dump_folder_path + '/' + VOCAB_FILES_NAMES['pretrained_vocab_file']
        print(f'Save vocabulary to {pytorch_vocab_dump_path}')
        corpus_vocab_dict = corpus.vocab.__dict__
        torch.save(corpus_vocab_dict, pytorch_vocab_dump_path)
        corpus_dict_no_vocab = corpus.__dict__
        corpus_dict_no_vocab.pop('vocab', None)
        pytorch_dataset_dump_path = pytorch_dump_folder_path + '/' + CORPUS_NAME
        print(f'Save dataset to {pytorch_dataset_dump_path}')
        torch.save(corpus_dict_no_vocab, pytorch_dataset_dump_path)
    if tf_checkpoint_path:
        config_path = os.path.abspath(transfo_xl_config_file)
        tf_path = os.path.abspath(tf_checkpoint_path)
        print(f'Converting Transformer XL checkpoint from {tf_path} with config at {config_path}.')
        if transfo_xl_config_file == '':
            config = TransfoXLConfig()
        else:
            config = TransfoXLConfig.from_json_file(transfo_xl_config_file)
        print(f'Building PyTorch model from configuration: {config}')
        model = TransfoXLLMHeadModel(config)
        model = load_tf_weights_in_transfo_xl(model, config, tf_path)
        pytorch_weights_dump_path = os.path.join(pytorch_dump_folder_path, WEIGHTS_NAME)
        pytorch_config_dump_path = os.path.join(pytorch_dump_folder_path, CONFIG_NAME)
        print(f'Save PyTorch model to {os.path.abspath(pytorch_weights_dump_path)}')
        torch.save(model.state_dict(), pytorch_weights_dump_path)
        print(f'Save configuration file to {os.path.abspath(pytorch_config_dump_path)}')
        with open(pytorch_config_dump_path, 'w', encoding='utf-8') as f:
            f.write(config.to_json_string())
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pytorch_dump_folder_path', default=None, type=str, required=True, help='Path to the folder to store the PyTorch model or dataset/vocab.')
    parser.add_argument('--tf_checkpoint_path', default='', type=str, help='An optional path to a TensorFlow checkpoint path to be converted.')
    parser.add_argument('--transfo_xl_config_file', default='', type=str, help='An optional config json file corresponding to the pre-trained BERT model. \nThis specifies the model architecture.')
    parser.add_argument('--transfo_xl_dataset_file', default='', type=str, help='An optional dataset file to be converted in a vocabulary.')
    args = parser.parse_args()
    convert_transfo_xl_checkpoint_to_pytorch(args.tf_checkpoint_path, args.transfo_xl_config_file, args.pytorch_dump_folder_path, args.transfo_xl_dataset_file)