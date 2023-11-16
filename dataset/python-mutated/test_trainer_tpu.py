import sys
from typing import Dict
from transformers import EvalPrediction, HfArgumentParser, TrainingArguments, is_torch_available
from transformers.utils import logging
logger = logging.get_logger(__name__)
if is_torch_available():
    import torch
    from torch import nn
    from torch.utils.data import Dataset
    from transformers import Trainer

    class DummyDataset(Dataset):

        def __init__(self, length: int=101):
            if False:
                while True:
                    i = 10
            self.length = length

        def __len__(self):
            if False:
                while True:
                    i = 10
            return self.length

        def __getitem__(self, i) -> int:
            if False:
                print('Hello World!')
            return i

    class DummyDataCollator:

        def __call__(self, features):
            if False:
                return 10
            return {'input_ids': torch.tensor(features), 'labels': torch.tensor(features)}

    class DummyModel(nn.Module):

        def __init__(self):
            if False:
                i = 10
                return i + 15
            super().__init__()
            self.fc = nn.Linear(120, 80)

        def forward(self, input_ids, labels=None):
            if False:
                return 10
            if labels is not None:
                return (torch.tensor(0.0, device=input_ids.device), input_ids)
            else:
                return input_ids

def main():
    if False:
        i = 10
        return i + 15
    parser = HfArgumentParser((TrainingArguments,))
    sys.argv += ['--output_dir', './examples']
    training_args = parser.parse_args_into_dataclasses()[0]
    logger.warning(f'Process rank: {training_args.local_rank}, device: {training_args.device}, tpu_num_cores: {training_args.tpu_num_cores}')
    for dataset_length in [1001, 256, 15]:
        dataset = DummyDataset(dataset_length)

        def compute_metrics(p: EvalPrediction) -> Dict:
            if False:
                for i in range(10):
                    print('nop')
            sequential = list(range(len(dataset)))
            success = p.predictions.tolist() == sequential and p.label_ids.tolist() == sequential
            return {'success': success}
        trainer = Trainer(model=DummyModel(), args=training_args, data_collator=DummyDataCollator(), eval_dataset=dataset, compute_metrics=compute_metrics)
        metrics = trainer.evaluate()
        logger.info(metrics)
        if metrics['eval_success'] is not True:
            logger.error(metrics)
            exit(1)
        p = trainer.predict(dataset)
        logger.info(p.metrics)
        if p.metrics['test_success'] is not True:
            logger.error(p.metrics)
            exit(1)
        trainer.args.eval_accumulation_steps = 2
        metrics = trainer.evaluate()
        logger.info(metrics)
        if metrics['eval_success'] is not True:
            logger.error(metrics)
            exit(1)
        p = trainer.predict(dataset)
        logger.info(p.metrics)
        if p.metrics['test_success'] is not True:
            logger.error(p.metrics)
            exit(1)
        trainer.args.eval_accumulation_steps = None
    logger.info('🔥 All distributed tests successful')

def _mp_fn(index):
    if False:
        print('Hello World!')
    main()
if __name__ == '__main__':
    main()