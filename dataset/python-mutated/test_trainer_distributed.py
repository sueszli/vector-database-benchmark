from typing import Dict
import numpy as np
from transformers import EvalPrediction, HfArgumentParser, TrainingArguments, is_torch_available
from transformers.testing_utils import TestCasePlus, execute_subprocess_async, get_torch_dist_unique_port, require_torch_multi_gpu, require_torch_multi_xpu, require_torch_neuroncore, require_torch_npu
from transformers.training_args import ParallelMode
from transformers.utils import logging
logger = logging.get_logger(__name__)
if is_torch_available():
    import torch
    from torch import nn
    from torch.utils.data import Dataset, IterableDataset
    from transformers import Trainer

    class DummyDataset(Dataset):

        def __init__(self, length: int=101):
            if False:
                print('Hello World!')
            self.length = length

        def __len__(self):
            if False:
                i = 10
                return i + 15
            return self.length

        def __getitem__(self, i) -> int:
            if False:
                print('Hello World!')
            return i

    class DummyDataCollator:

        def __call__(self, features):
            if False:
                while True:
                    i = 10
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
                i = 10
                return i + 15
            if labels is not None:
                return (torch.tensor(0.0, device=input_ids.device), input_ids)
            else:
                return input_ids

    class RegressionModel(nn.Module):

        def __init__(self, a=0, b=0, double_output=False):
            if False:
                i = 10
                return i + 15
            super().__init__()
            self.a = nn.Parameter(torch.tensor(a).float())
            self.b = nn.Parameter(torch.tensor(b).float())
            self.double_output = double_output
            self.config = None

        def forward(self, input_x, labels=None, **kwargs):
            if False:
                i = 10
                return i + 15
            y = input_x * self.a + self.b
            if labels is None:
                return (y, y) if self.double_output else (y,)
            loss = nn.functional.mse_loss(y, labels)
            return (loss, y, y) if self.double_output else (loss, y)

    class SampleIterableDataset(IterableDataset):

        def __init__(self, a=2, b=3, length=64, seed=42, label_names=None):
            if False:
                for i in range(10):
                    print('nop')
            self.dataset = RegressionDataset(a=a, b=b, length=length, seed=seed, label_names=label_names)

        def __iter__(self):
            if False:
                return 10
            for i in range(len(self.dataset)):
                yield self.dataset[i]

    class FiniteIterableDataset(SampleIterableDataset):

        def __init__(self, a=2, b=3, length=64, seed=42, label_names=None):
            if False:
                while True:
                    i = 10
            super().__init__(a, b, length, seed, label_names)
            self.current_sample = 0

        def __iter__(self):
            if False:
                i = 10
                return i + 15
            while self.current_sample < len(self.dataset):
                yield self.dataset[self.current_sample]
                self.current_sample += 1

    class RegressionDataset:

        def __init__(self, a=2, b=3, length=64, seed=42, label_names=None):
            if False:
                i = 10
                return i + 15
            np.random.seed(seed)
            self.label_names = ['labels'] if label_names is None else label_names
            self.length = length
            self.x = np.random.normal(size=(length,)).astype(np.float32)
            self.ys = [a * self.x + b + np.random.normal(scale=0.1, size=(length,)) for _ in self.label_names]
            self.ys = [y.astype(np.float32) for y in self.ys]

        def __len__(self):
            if False:
                return 10
            return self.length

        def __getitem__(self, i):
            if False:
                i = 10
                return i + 15
            result = {name: y[i] for (name, y) in zip(self.label_names, self.ys)}
            result['input_x'] = self.x[i]
            return result

class TestTrainerDistributedNeuronCore(TestCasePlus):

    @require_torch_neuroncore
    def test_trainer(self):
        if False:
            print('Hello World!')
        distributed_args = f'--nproc_per_node=2\n            --master_port={get_torch_dist_unique_port()}\n            {self.test_file_dir}/test_trainer_distributed.py\n        '.split()
        output_dir = self.get_auto_remove_tmp_dir()
        args = f'--output_dir {output_dir}'.split()
        cmd = ['torchrun'] + distributed_args + args
        execute_subprocess_async(cmd, env=self.get_env())

class TestTrainerDistributedNPU(TestCasePlus):

    @require_torch_npu
    def test_trainer(self):
        if False:
            print('Hello World!')
        distributed_args = f'--nproc_per_node=2\n            --master_port={get_torch_dist_unique_port()}\n            {self.test_file_dir}/test_trainer_distributed.py\n        '.split()
        output_dir = self.get_auto_remove_tmp_dir()
        args = f'--output_dir {output_dir}'.split()
        cmd = ['torchrun'] + distributed_args + args
        execute_subprocess_async(cmd, env=self.get_env())

class TestTrainerDistributed(TestCasePlus):

    @require_torch_multi_gpu
    def test_trainer(self):
        if False:
            print('Hello World!')
        distributed_args = f'--nproc_per_node={torch.cuda.device_count()}\n            --master_port={get_torch_dist_unique_port()}\n            {self.test_file_dir}/test_trainer_distributed.py\n        '.split()
        output_dir = self.get_auto_remove_tmp_dir()
        args = f'--output_dir {output_dir}'.split()
        cmd = ['torchrun'] + distributed_args + args
        execute_subprocess_async(cmd, env=self.get_env())

@require_torch_multi_xpu
class TestTrainerDistributedXPU(TestCasePlus):

    def test_trainer(self):
        if False:
            i = 10
            return i + 15
        distributed_args = f'--nproc_per_node={torch.xpu.device_count()}\n            --master_port={get_torch_dist_unique_port()}\n            {self.test_file_dir}/test_trainer_distributed.py\n        '.split()
        output_dir = self.get_auto_remove_tmp_dir()
        args = f'--output_dir {output_dir}'.split()
        cmd = ['torchrun'] + distributed_args + args
        execute_subprocess_async(cmd, env=self.get_env())
if __name__ == '__main__':
    parser = HfArgumentParser((TrainingArguments,))
    training_args = parser.parse_args_into_dataclasses()[0]
    logger.warning(f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, distributed training: {training_args.parallel_mode != ParallelMode.NOT_DISTRIBUTED}')
    for dataset_length in [101, 40, 7]:
        dataset = DummyDataset(dataset_length)

        def compute_metrics(p: EvalPrediction) -> Dict:
            if False:
                while True:
                    i = 10
            sequential = list(range(len(dataset)))
            success = p.predictions.tolist() == sequential and p.label_ids.tolist() == sequential
            if not success and training_args.local_rank == 0:
                logger.warning(f'Predictions and/or labels do not match expected results:\n  - predictions: {p.predictions.tolist()}\n  - labels: {p.label_ids.tolist()}\n  - expected: {sequential}')
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
    train_dataset = FiniteIterableDataset(label_names=['labels', 'extra'], length=1)
    model = RegressionModel()
    training_args.per_device_train_batch_size = 1
    training_args.max_steps = 1
    training_args.dispatch_batches = False
    trainer = Trainer(model, training_args, train_dataset=train_dataset)
    trainer.train()