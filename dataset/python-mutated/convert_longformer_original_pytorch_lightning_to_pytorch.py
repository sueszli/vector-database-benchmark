"""Convert RoBERTa checkpoint."""
import argparse
import pytorch_lightning as pl
import torch
from torch import nn
from transformers import LongformerForQuestionAnswering, LongformerModel

class LightningModel(pl.LightningModule):

    def __init__(self, model):
        if False:
            while True:
                i = 10
        super().__init__()
        self.model = model
        self.num_labels = 2
        self.qa_outputs = nn.Linear(self.model.config.hidden_size, self.num_labels)

    def forward(self):
        if False:
            print('Hello World!')
        pass

def convert_longformer_qa_checkpoint_to_pytorch(longformer_model: str, longformer_question_answering_ckpt_path: str, pytorch_dump_folder_path: str):
    if False:
        for i in range(10):
            print('nop')
    longformer = LongformerModel.from_pretrained(longformer_model)
    lightning_model = LightningModel(longformer)
    ckpt = torch.load(longformer_question_answering_ckpt_path, map_location=torch.device('cpu'))
    lightning_model.load_state_dict(ckpt['state_dict'])
    longformer_for_qa = LongformerForQuestionAnswering.from_pretrained(longformer_model)
    longformer_for_qa.longformer.load_state_dict(lightning_model.model.state_dict())
    longformer_for_qa.qa_outputs.load_state_dict(lightning_model.qa_outputs.state_dict())
    longformer_for_qa.eval()
    longformer_for_qa.save_pretrained(pytorch_dump_folder_path)
    print(f'Conversion successful. Model saved under {pytorch_dump_folder_path}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--longformer_model', default=None, type=str, required=True, help='model identifier of longformer. Should be either `longformer-base-4096` or `longformer-large-4096`.')
    parser.add_argument('--longformer_question_answering_ckpt_path', default=None, type=str, required=True, help='Path the official PyTorch Lightning Checkpoint.')
    parser.add_argument('--pytorch_dump_folder_path', default=None, type=str, required=True, help='Path to the output PyTorch model.')
    args = parser.parse_args()
    convert_longformer_qa_checkpoint_to_pytorch(args.longformer_model, args.longformer_question_answering_ckpt_path, args.pytorch_dump_folder_path)