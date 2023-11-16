import logging
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator, Optional, Tuple, Type
from torch.utils.data import DataLoader, Dataset, IterableDataset
import ray
from ray import train
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
from ray.data import DataIterator
from ray.data._internal.iterator.stream_split_iterator import StreamSplitDataIterator
from ray.data.dataset import MaterializedDataset
from ray.data.iterator import _IterableFromIterator
from ray.train import Checkpoint
from ray.train.huggingface.transformers.transformers_checkpoint import TransformersCheckpoint
from ray.util import PublicAPI
logger = logging.getLogger(__name__)
TRANSFORMERS_IMPORT_ERROR: Optional[ImportError] = None
try:
    import datasets.iterable_dataset
    import transformers.trainer
    from transformers import Trainer
    from transformers.trainer_callback import TrainerCallback
    from transformers.trainer_utils import IntervalStrategy
except ImportError as e:
    TRANSFORMERS_IMPORT_ERROR = e
    TrainerCallback = object

def maybe_add_length(obj: Any, length: Optional[int]) -> Any:
    if False:
        while True:
            i = 10
    'Change the class of obj to a subclass with predefined __len__ if needed.'
    if not length or hasattr(obj, '__len__'):
        return obj

    def __len__(self):
        if False:
            while True:
                i = 10
        return length
    new_class = type(f'{obj.__class__.__name__}WithLength', (obj.__class__,), {'__len__': __len__})
    obj.__class__ = new_class
    return obj

def wrap_transformers_trainer(trainer: 'Trainer') -> 'Trainer':
    if False:
        print('Hello World!')
    'Change the class of trainer to a subclass implementing Ray-specific logic.'
    base_trainer_class: Type[transformers.trainer.Trainer] = trainer.__class__

    class RayTrainer(base_trainer_class):

        def get_train_dataloader(self):
            if False:
                i = 10
                return i + 15
            data_loader = super().get_train_dataloader()
            if isinstance(data_loader.dataset, transformers.trainer.IterableDatasetShard) and getattr(data_loader.dataset.dataset, '_do_not_split', False):
                data_loader.dataset.num_processes = 1
                data_loader.dataset.process_index = 0
            return data_loader
    trainer.__class__ = RayTrainer
    return trainer

class RayDatasetHFIterable(datasets.iterable_dataset._BaseExamplesIterable):
    """HF ``_BaseExamplesIterable`` backed by a ``ray.data.DataIterator``.

    The other abstract methods of shuffling and sharding the data are not implemented,
    since those operations should be done by Ray Data. For example, the dataset
    is already sharded to each data parallel worker and is disabled
    (see ``wrap_transformers_trainer`` above).
    """

    def __init__(self, dataset: DataIterator) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self.dataset = dataset

    def __iter__(self) -> Iterator[Tuple[int, dict]]:
        if False:
            for i in range(10):
                print('nop')
        for (idx, row) in enumerate(self.dataset.iter_rows()):
            yield (idx, {k: v for (k, v) in row.items()})

def process_dataset_for_hf(dataset: DataIterator, disable_transformers_splitting: bool=False) -> 'IterableDataset':
    if False:
        for i in range(10):
            print('nop')
    'Converts a Ray Dataset into a HF IterableDataset.'
    hf_iterable = RayDatasetHFIterable(dataset)
    iterable_dataset = datasets.iterable_dataset.IterableDataset(hf_iterable, format_type='torch').with_format('torch')
    if isinstance(dataset, StreamSplitDataIterator):
        if isinstance(dataset._base_dataset, MaterializedDataset):
            dataset_length = dataset._base_dataset.count() // dataset.world_size()
        else:
            dataset_length = None
            logger.warning(f'The length for {dataset._base_dataset} cannot be determined since it is a streaming dataset. HF transformers requires `max_steps` to be passed in this case, or you can materialize the dataset with `ds.materialize()`.')
    else:
        try:
            dataset_length = dataset._base_dataset.count()
        except (ValueError, AttributeError):
            dataset_length = None
    iterable_dataset = maybe_add_length(iterable_dataset, dataset_length)
    iterable_dataset._do_not_split = disable_transformers_splitting
    return iterable_dataset

def process_datasets(train_dataset: DataIterator, eval_dataset: DataIterator) -> Tuple['IterableDataset', 'IterableDataset']:
    if False:
        print('Hello World!')
    'Convert Ray train and validation to HF IterableDatasets.'
    if train_dataset:
        train_torch_dataset = process_dataset_for_hf(train_dataset, disable_transformers_splitting=True)
    else:
        train_torch_dataset = None
    if eval_dataset:
        eval_torch_dataset = process_dataset_for_hf(eval_dataset)
    else:
        eval_torch_dataset = None
    return (train_torch_dataset, eval_torch_dataset)

class TrainReportCallback(TrainerCallback):
    """HF TrainerCallback for Ray Train metric reporting & checkpointing."""

    def __init__(self) -> None:
        if False:
            return 10
        self.delayed_report = {'metrics': {}, 'checkpoint': None}
        self.last_metrics = {}
        self.last_step = 0
        super().__init__()

    def on_epoch_end(self, args, state, control, **kwargs):
        if False:
            return 10
        if control.should_training_stop:
            if state.global_step != self.last_step:
                if args.evaluation_strategy not in ('no', IntervalStrategy.NO):
                    control.should_evaluate = True
                control.should_log = True
            control.should_save = True
        return control

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if False:
            return 10
        report = {**logs, 'step': state.global_step, 'epoch': state.epoch}
        self.delayed_report['metrics'].update(report)
        self.last_step = state.global_step

    def on_save(self, args, state, control, **kwargs):
        if False:
            i = 10
            return i + 15
        checkpoint_path = Path(transformers.trainer.get_last_checkpoint(args.output_dir)).absolute()
        if checkpoint_path:
            checkpoint = TransformersCheckpoint.from_directory(str(checkpoint_path))
            self.delayed_report['checkpoint'] = checkpoint

    def _report(self):
        if False:
            for i in range(10):
                print('nop')
        if self.delayed_report['metrics']:
            train.report(**self.delayed_report)
            self.last_metrics = self.delayed_report['metrics']
            self.delayed_report = {'metrics': {}, 'checkpoint': None}

    def on_epoch_begin(self, args, state, control, **kwargs):
        if False:
            return 10
        self._report()

    def on_step_begin(self, args, state, control, **kwargs):
        if False:
            return 10
        self._report()

    def on_train_end(self, args, state, control, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.delayed_report['metrics'] = {**self.last_metrics, **self.delayed_report['metrics']}
        self._report()

@PublicAPI(stability='beta')
class RayTrainReportCallback(TrainerCallback):
    """A simple callback to report checkpoints and metrics to Ray Tarin.

    This callback is a subclass of `transformers.TrainerCallback
    <https://huggingface.co/docs/transformers/main/en/main_classes/callback#transformers.TrainerCallback>`_
    and overrides the `TrainerCallback.on_save()` method. After
    a new checkpoint get saved, it fetches the latest metric dictionary
    from `TrainerState.log_history` and reports it with the latest checkpoint
    to Ray Train.

    Checkpoints will be saved in the following structure::

        checkpoint_00000*/   Ray Train Checkpoint
        └─ checkpoint/       Hugging Face Transformers Checkpoint

    For customized reporting and checkpointing logic, implement your own
    `transformers.TrainerCallback` following this user
    guide: :ref:`Saving and Loading Checkpoints <train-dl-saving-checkpoints>`.

    Note that users should ensure that the logging, evaluation, and saving frequencies
    are properly configured so that the monitoring metric is always up-to-date
    when `transformers.Trainer` saves a checkpoint.

    Suppose the monitoring metric is reported from evaluation stage:

    Some valid configurations:
        - evaluation_strategy == save_strategy == "epoch"
        - evaluation_strategy == save_strategy == "steps", save_steps % eval_steps == 0

    Some invalid configurations:
        - evaluation_strategy != save_strategy
        - evaluation_strategy == save_strategy == "steps", save_steps % eval_steps != 0

    """

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        record_extra_usage_tag(TagKey.TRAIN_TRANSFORMERS_RAYTRAINREPORTCALLBACK, '1')

    def on_save(self, args, state, control, **kwargs):
        if False:
            while True:
                i = 10
        'Event called after a checkpoint save.'
        with TemporaryDirectory() as tmpdir:
            metrics = {}
            for log in state.log_history:
                metrics.update(log)
            source_ckpt_path = transformers.trainer.get_last_checkpoint(args.output_dir)
            target_ckpt_path = os.path.join(tmpdir, 'checkpoint')
            shutil.copytree(source_ckpt_path, target_ckpt_path)
            checkpoint = Checkpoint.from_directory(tmpdir)
            ray.train.report(metrics=metrics, checkpoint=checkpoint)

class RayTorchIterableDataset(IterableDataset):
    """Wrapper class for ray data iterables."""

    def __init__(self, data_iterable) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.data_iterable = data_iterable

    def __iter__(self) -> Iterator:
        if False:
            while True:
                i = 10
        return iter(self.data_iterable)

@PublicAPI(stability='beta')
def prepare_trainer(trainer: 'Trainer') -> 'Trainer':
    if False:
        for i in range(10):
            print('nop')
    'Prepare your HuggingFace Transformer Trainer for Ray Train.\n\n    This utility function enable the trainer integrates with Ray Data Integration.\n    Internally, it overrides the `get_train_dataloader` and `get_eval_dataloader`\n    methods and inject the data integration logics if the `train_dataset` and\n    `eval_dataset` are Ray Data Iterables.\n    '
    if TRANSFORMERS_IMPORT_ERROR is not None:
        raise TRANSFORMERS_IMPORT_ERROR
    base_trainer_class: Type[transformers.trainer.Trainer] = trainer.__class__

    class RayTransformersTrainer(base_trainer_class):
        """A Wrapper of `transformers.Trainer` for Ray Data Integration."""

        def get_train_dataloader(self) -> DataLoader:
            if False:
                print('Hello World!')
            if isinstance(self.train_dataset, _IterableFromIterator):
                dataset = RayTorchIterableDataset(self.train_dataset)
                return DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0])
            else:
                return super().get_train_dataloader()

        def get_eval_dataloader(self, eval_dataset: Optional[Dataset]=None) -> DataLoader:
            if False:
                for i in range(10):
                    print('nop')
            if eval_dataset is None:
                eval_dataset = self.eval_dataset
            if isinstance(eval_dataset, _IterableFromIterator):
                dataset = RayTorchIterableDataset(eval_dataset)
                return DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0])
            else:
                return super().get_eval_dataloader(eval_dataset)
    trainer.__class__ = RayTransformersTrainer
    record_extra_usage_tag(TagKey.TRAIN_TRANSFORMERS_PREPARE_TRAINER, '1')
    return trainer