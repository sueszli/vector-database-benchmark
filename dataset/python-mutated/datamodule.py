"""LightningDataModule for loading DataLoaders with ease."""
import inspect
from typing import IO, Any, Dict, Iterable, Optional, Union, cast
from lightning_utilities import apply_to_collection
from torch.utils.data import DataLoader, Dataset, IterableDataset
from typing_extensions import Self
import lightning.pytorch as pl
from lightning.fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
from lightning.pytorch.core.hooks import DataHooks
from lightning.pytorch.core.mixins import HyperparametersMixin
from lightning.pytorch.core.saving import _load_from_checkpoint
from lightning.pytorch.utilities.model_helpers import _restricted_classmethod
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

class LightningDataModule(DataHooks, HyperparametersMixin):
    """A DataModule standardizes the training, val, test splits, data preparation and transforms. The main advantage is
    consistent data splits, data preparation and transforms across models.

    Example::

        import lightning as L
        import torch.utils.data as data
        from lightning.pytorch.demos.boring_classes import RandomDataset

        class MyDataModule(L.LightningDataModule):
            def prepare_data(self):
                # download, IO, etc. Useful with shared filesystems
                # only called on 1 GPU/TPU in distributed
                ...

            def setup(self, stage):
                # make assignments here (val/train/test split)
                # called on every process in DDP
                dataset = RandomDataset(1, 100)
                self.train, self.val, self.test = data.random_split(
                    dataset, [80, 10, 10], generator=torch.Generator().manual_seed(42)
                )

            def train_dataloader(self):
                return data.DataLoader(self.train)

            def val_dataloader(self):
                return data.DataLoader(self.val)

            def test_dataloader(self):
                return data.DataLoader(self.test)

            def teardown(self):
                # clean up state after the trainer stops, delete files...
                # called on every process in DDP
                ...

    """
    name: Optional[str] = None
    CHECKPOINT_HYPER_PARAMS_KEY = 'datamodule_hyper_parameters'
    CHECKPOINT_HYPER_PARAMS_NAME = 'datamodule_hparams_name'
    CHECKPOINT_HYPER_PARAMS_TYPE = 'datamodule_hparams_type'

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.trainer: Optional['pl.Trainer'] = None

    @classmethod
    def from_datasets(cls, train_dataset: Optional[Union[Dataset, Iterable[Dataset]]]=None, val_dataset: Optional[Union[Dataset, Iterable[Dataset]]]=None, test_dataset: Optional[Union[Dataset, Iterable[Dataset]]]=None, predict_dataset: Optional[Union[Dataset, Iterable[Dataset]]]=None, batch_size: int=1, num_workers: int=0, **datamodule_kwargs: Any) -> 'LightningDataModule':
        if False:
            while True:
                i = 10
        "Create an instance from torch.utils.data.Dataset.\n\n        Args:\n            train_dataset: Optional dataset or iterable of datasets to be used for train_dataloader()\n            val_dataset: Optional dataset or iterable of datasets to be used for val_dataloader()\n            test_dataset: Optional dataset or iterable of datasets to be used for test_dataloader()\n            predict_dataset: Optional dataset or iterable of datasets to be used for predict_dataloader()\n            batch_size: Batch size to use for each dataloader. Default is 1. This parameter gets forwarded to the\n                ``__init__`` if the datamodule has such a name defined in its signature.\n            num_workers: Number of subprocesses to use for data loading. 0 means that the\n                data will be loaded in the main process. Number of CPUs available. This parameter gets forwarded to the\n                ``__init__`` if the datamodule has such a name defined in its signature.\n            **datamodule_kwargs: Additional parameters that get passed down to the datamodule's ``__init__``.\n\n        "

        def dataloader(ds: Dataset, shuffle: bool=False) -> DataLoader:
            if False:
                i = 10
                return i + 15
            shuffle &= not isinstance(ds, IterableDataset)
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

        def train_dataloader() -> TRAIN_DATALOADERS:
            if False:
                i = 10
                return i + 15
            return apply_to_collection(train_dataset, Dataset, dataloader, shuffle=True)

        def val_dataloader() -> EVAL_DATALOADERS:
            if False:
                while True:
                    i = 10
            return apply_to_collection(val_dataset, Dataset, dataloader)

        def test_dataloader() -> EVAL_DATALOADERS:
            if False:
                while True:
                    i = 10
            return apply_to_collection(test_dataset, Dataset, dataloader)

        def predict_dataloader() -> EVAL_DATALOADERS:
            if False:
                while True:
                    i = 10
            return apply_to_collection(predict_dataset, Dataset, dataloader)
        candidate_kwargs = {'batch_size': batch_size, 'num_workers': num_workers}
        accepted_params = inspect.signature(cls.__init__).parameters
        accepts_kwargs = any((param.kind == param.VAR_KEYWORD for param in accepted_params.values()))
        if accepts_kwargs:
            special_kwargs = candidate_kwargs
        else:
            accepted_param_names = set(accepted_params)
            accepted_param_names.discard('self')
            special_kwargs = {k: v for (k, v) in candidate_kwargs.items() if k in accepted_param_names}
        datamodule = cls(**datamodule_kwargs, **special_kwargs)
        if train_dataset is not None:
            datamodule.train_dataloader = train_dataloader
        if val_dataset is not None:
            datamodule.val_dataloader = val_dataloader
        if test_dataset is not None:
            datamodule.test_dataloader = test_dataloader
        if predict_dataset is not None:
            datamodule.predict_dataloader = predict_dataloader
        return datamodule

    def state_dict(self) -> Dict[str, Any]:
        if False:
            return 10
        'Called when saving a checkpoint, implement to generate and save datamodule state.\n\n        Returns:\n            A dictionary containing datamodule state.\n\n        '
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        'Called when loading a checkpoint, implement to reload datamodule state given datamodule state_dict.\n\n        Args:\n            state_dict: the datamodule state returned by ``state_dict``.\n\n        '
        pass

    @_restricted_classmethod
    def load_from_checkpoint(cls, checkpoint_path: Union[_PATH, IO], map_location: _MAP_LOCATION_TYPE=None, hparams_file: Optional[_PATH]=None, **kwargs: Any) -> Self:
        if False:
            i = 10
            return i + 15
        'Primary way of loading a datamodule from a checkpoint. When Lightning saves a checkpoint it stores the\n        arguments passed to ``__init__``  in the checkpoint under ``"datamodule_hyper_parameters"``.\n\n        Any arguments specified through \\*\\*kwargs will override args stored in ``"datamodule_hyper_parameters"``.\n\n        Args:\n            checkpoint_path: Path to checkpoint. This can also be a URL, or file-like object\n            map_location:\n                If your checkpoint saved a GPU model and you now load on CPUs\n                or a different number of GPUs, use this to map to the new setup.\n                The behaviour is the same as in :func:`torch.load`.\n            hparams_file: Optional path to a ``.yaml`` or ``.csv`` file with hierarchical structure\n                as in this example::\n\n                    dataloader:\n                        batch_size: 32\n\n                You most likely won\'t need this since Lightning will always save the hyperparameters\n                to the checkpoint.\n                However, if your checkpoint weights don\'t have the hyperparameters saved,\n                use this method to pass in a ``.yaml`` file with the hparams you\'d like to use.\n                These will be converted into a :class:`~dict` and passed into your\n                :class:`LightningDataModule` for use.\n\n                If your datamodule\'s ``hparams`` argument is :class:`~argparse.Namespace`\n                and ``.yaml`` file has hierarchical structure, you need to refactor your datamodule to treat\n                ``hparams`` as :class:`~dict`.\n            \\**kwargs: Any extra keyword args needed to init the datamodule. Can also be used to override saved\n                hyperparameter values.\n\n        Return:\n            :class:`LightningDataModule` instance with loaded weights and hyperparameters (if available).\n\n        Note:\n            ``load_from_checkpoint`` is a **class** method. You must use your :class:`LightningDataModule`\n            **class** to call it instead of the :class:`LightningDataModule` instance, or a\n            ``TypeError`` will be raised.\n\n        Example::\n\n            # load weights without mapping ...\n            datamodule = MyLightningDataModule.load_from_checkpoint(\'path/to/checkpoint.ckpt\')\n\n            # or load weights and hyperparameters from separate files.\n            datamodule = MyLightningDataModule.load_from_checkpoint(\n                \'path/to/checkpoint.ckpt\',\n                hparams_file=\'/path/to/hparams_file.yaml\'\n            )\n\n            # override some of the params with new values\n            datamodule = MyLightningDataModule.load_from_checkpoint(\n                PATH,\n                batch_size=32,\n                num_workers=10,\n            )\n\n        '
        loaded = _load_from_checkpoint(cls, checkpoint_path, map_location=map_location, hparams_file=hparams_file, strict=None, **kwargs)
        return cast(Self, loaded)