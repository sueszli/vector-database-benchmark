from typing import Any, Dict, List, Optional, Set, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from snorkel.analysis import Scorer
from snorkel.classification import DictDataLoader, DictDataset, Operation, Task
from snorkel.classification.data import DEFAULT_INPUT_DATA_KEY, DEFAULT_TASK_NAME
from snorkel.classification.multitask_classifier import MultitaskClassifier
from .utils import add_slice_labels, convert_to_slice_tasks

class SliceAwareClassifier(MultitaskClassifier):
    """A slice-aware classifier that supports training + scoring on slice labels.

    NOTE: This model currently only supports binary classification.

    Parameters
    ----------
    base_architecture
        A network architecture that accepts input data and outputs a representation
    head_dim
        Output feature dimension of the base_architecture, and input dimension of the
        internal prediction head: ``nn.Linear(head_dim, 2)``.
    slice_names
        A list of slice names that the model will accept initialize as tasks
        and accept as corresponding labels
    scorer
        A Scorer to be used for initialization of the ``MultitaskClassifier`` superclass.
    **multitask_kwargs
        Arbitrary keyword arguments to be passed to the ``MultitaskClassifier`` superclass.

    Attributes
    ----------
    base_task
        A base ``snorkel.classification.Task`` that the model will learn.
        This becomes a ``master_head_module`` that combines slice tasks information.
        For more, see ``snorkel.slicing.convert_to_slice_tasks``.
    slice_names
        See above
    """

    def __init__(self, base_architecture: nn.Module, head_dim: int, slice_names: List[str], input_data_key: str=DEFAULT_INPUT_DATA_KEY, task_name: str=DEFAULT_TASK_NAME, scorer: Scorer=Scorer(metrics=['accuracy', 'f1']), **multitask_kwargs: Any) -> None:
        if False:
            print('Hello World!')
        module_pool = nn.ModuleDict({'base_architecture': base_architecture, 'prediction_head': nn.Linear(head_dim, 2)})
        op_sequence = [Operation(name='input_op', module_name='base_architecture', inputs=[('_input_', input_data_key)]), Operation(name='head_op', module_name='prediction_head', inputs=['input_op'])]
        self.base_task = Task(name=task_name, module_pool=module_pool, op_sequence=op_sequence, scorer=scorer)
        slice_tasks = convert_to_slice_tasks(self.base_task, slice_names)
        model_name = f'{task_name}_sliceaware_classifier'
        super().__init__(tasks=slice_tasks, name=model_name, **multitask_kwargs)
        self.slice_names = slice_names

    def make_slice_dataloader(self, dataset: DictDataset, S: np.recarray, **dataloader_kwargs: Any) -> DictDataLoader:
        if False:
            while True:
                i = 10
        'Create DictDataLoader with slice labels, initialized from specified dataset.\n\n        Parameters\n        ----------\n        dataset\n            A DictDataset that will be converted into a slice-aware dataloader\n        S\n            A [num_examples, num_slices] slice matrix indicating whether\n            each example is in every slice\n        slice_names\n            A list of slice names corresponding to columns of ``S``\n\n        dataloader_kwargs\n            Arbitrary kwargs to be passed to DictDataLoader\n            See ``DictDataLoader.__init__``.\n        '
        if self.base_task.name not in dataset.Y_dict:
            raise ValueError(f'Base task ({self.base_task.name}) labels missing from {dataset}')
        dataloader = DictDataLoader(dataset, **dataloader_kwargs)
        add_slice_labels(dataloader, self.base_task, S)
        return dataloader

    @torch.no_grad()
    def score_slices(self, dataloaders: List[DictDataLoader], as_dataframe: bool=False) -> Union[Dict[str, float], pd.DataFrame]:
        if False:
            i = 10
            return i + 15
        'Scores appropriate slice labels using the overall prediction head.\n\n        In other words, uses ``base_task`` (NOT ``slice_tasks``) to evaluate slices.\n\n        In practice, we\'d like to use a final prediction from a _single_ task head.\n        To do so, ``self.base_task`` leverages reweighted slice representation to\n        make a prediction. In this method, we remap all slice-specific ``pred``\n        labels to ``self.base_task`` for evaluation.\n\n        Parameters\n        ----------\n        dataloaders\n            A list of DictDataLoaders to calculate scores for\n        as_dataframe\n            A boolean indicating whether to return results as pandas\n            DataFrame (True) or dict (False)\n        eval_slices_on_base_task\n            A boolean indicating whether to remap slice labels to base task.\n            Otherwise, keeps evaluation of slice labels on slice-specific heads.\n\n        Returns\n        -------\n        Dict[str, float]\n            A dictionary mapping metric¡ names to corresponding scores\n            Metric names will be of the form "task/dataset/split/metric"\n        '
        eval_mapping: Dict[str, Optional[str]] = {}
        all_labels: Union[List, Set] = []
        for dl in dataloaders:
            all_labels.extend(dl.dataset.Y_dict.keys())
        all_labels = set(all_labels)
        for label in all_labels:
            if 'pred' in label:
                eval_mapping[label] = self.base_task.name
            elif 'ind' in label:
                eval_mapping[label] = None
        return super().score(dataloaders=dataloaders, remap_labels=eval_mapping, as_dataframe=as_dataframe)