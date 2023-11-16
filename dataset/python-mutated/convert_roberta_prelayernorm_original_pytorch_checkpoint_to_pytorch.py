"""Convert RoBERTa-PreLayerNorm checkpoint."""
import argparse
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, RobertaPreLayerNormConfig, RobertaPreLayerNormForMaskedLM
from transformers.utils import logging
logging.set_verbosity_info()
logger = logging.get_logger(__name__)

def convert_roberta_prelayernorm_checkpoint_to_pytorch(checkpoint_repo: str, pytorch_dump_folder_path: str):
    if False:
        print('Hello World!')
    "\n    Copy/paste/tweak roberta_prelayernorm's weights to our BERT structure.\n    "
    config = RobertaPreLayerNormConfig.from_pretrained(checkpoint_repo, architectures=['RobertaPreLayerNormForMaskedLM'])
    original_state_dict = torch.load(hf_hub_download(repo_id=checkpoint_repo, filename='pytorch_model.bin'))
    state_dict = {}
    for (tensor_key, tensor_value) in original_state_dict.items():
        if tensor_key.startswith('roberta.'):
            tensor_key = 'roberta_prelayernorm.' + tensor_key[len('roberta.'):]
        if tensor_key.endswith('.self.LayerNorm.weight') or tensor_key.endswith('.self.LayerNorm.bias'):
            continue
        state_dict[tensor_key] = tensor_value
    model = RobertaPreLayerNormForMaskedLM.from_pretrained(pretrained_model_name_or_path=None, config=config, state_dict=state_dict)
    model.save_pretrained(pytorch_dump_folder_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_repo)
    tokenizer.save_pretrained(pytorch_dump_folder_path)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-repo', default=None, type=str, required=True, help="Path the official PyTorch dump, e.g. 'andreasmadsen/efficient_mlm_m0.40'.")
    parser.add_argument('--pytorch_dump_folder_path', default=None, type=str, required=True, help='Path to the output PyTorch model.')
    args = parser.parse_args()
    convert_roberta_prelayernorm_checkpoint_to_pytorch(args.checkpoint_repo, args.pytorch_dump_folder_path)