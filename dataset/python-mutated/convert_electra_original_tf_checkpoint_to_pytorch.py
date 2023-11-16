"""Convert ELECTRA checkpoint."""
import argparse
import torch
from transformers import ElectraConfig, ElectraForMaskedLM, ElectraForPreTraining, load_tf_weights_in_electra
from transformers.utils import logging
logging.set_verbosity_info()

def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, config_file, pytorch_dump_path, discriminator_or_generator):
    if False:
        print('Hello World!')
    config = ElectraConfig.from_json_file(config_file)
    print(f'Building PyTorch model from configuration: {config}')
    if discriminator_or_generator == 'discriminator':
        model = ElectraForPreTraining(config)
    elif discriminator_or_generator == 'generator':
        model = ElectraForMaskedLM(config)
    else:
        raise ValueError("The discriminator_or_generator argument should be either 'discriminator' or 'generator'")
    load_tf_weights_in_electra(model, config, tf_checkpoint_path, discriminator_or_generator=discriminator_or_generator)
    print(f'Save PyTorch model to {pytorch_dump_path}')
    torch.save(model.state_dict(), pytorch_dump_path)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf_checkpoint_path', default=None, type=str, required=True, help='Path to the TensorFlow checkpoint path.')
    parser.add_argument('--config_file', default=None, type=str, required=True, help='The config json file corresponding to the pre-trained model. \nThis specifies the model architecture.')
    parser.add_argument('--pytorch_dump_path', default=None, type=str, required=True, help='Path to the output PyTorch model.')
    parser.add_argument('--discriminator_or_generator', default=None, type=str, required=True, help="Whether to export the generator or the discriminator. Should be a string, either 'discriminator' or 'generator'.")
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.config_file, args.pytorch_dump_path, args.discriminator_or_generator)