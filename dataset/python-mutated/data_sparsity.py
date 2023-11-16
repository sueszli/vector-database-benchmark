from collections import defaultdict
from copy import deepcopy
import torch
from typing import Any, Optional, Dict
import pytorch_lightning as pl
from ._data_sparstity_utils import _attach_model_to_data_sparsifier, _log_sparsified_level, _get_valid_name

class PostTrainingDataSparsity(pl.callbacks.Callback):
    """Lightning callback that enables post-training sparsity.

    This callback aims to sparsify the model inside lightning module after training.
    **Note that the model is copied and then sparsified, so the existing model is not modified**

    The sparsified model can be used for comparison and can be accessed using
        <callback_obj>.sparsified

    Args:
        data_sparsifier_class (some implemented class of BaseDataSparsifier)
            The data sparsifier object of this class is created when the
            training starts.
            Note: Objects should not be passed in here as they are created
            once the training completes.

        data_sparsifier_args (Dict)
            Dictionary of args to be passed to the data sparsifier.
            Note: data_list arg should be ignored

    Hooks implemented:
        on_fit_end()
            1. copies the model and attaches it to the sparsifier
            2. sparsier step() is called
            3. squashes the mask()
    """

    def __init__(self, data_sparsifier_class, data_sparsifier_args):
        if False:
            print('Hello World!')
        super().__init__()
        self.data_sparsifier_class = data_sparsifier_class
        self.data_sparsifier_args = data_sparsifier_args
        self.data_sparsifier: Any = None
        self.sparsified: Optional[torch.nn.Module] = None

    def on_fit_end(self, trainer, pl_module) -> None:
        if False:
            print('Hello World!')
        self.sparsified = deepcopy(pl_module.model).eval()
        self.data_sparsifier = self.data_sparsifier_class(**self.data_sparsifier_args)
        _attach_model_to_data_sparsifier(self.sparsified, self.data_sparsifier)
        self.data_sparsifier.step()
        self.data_sparsifier.squash_mask()
        _log_sparsified_level(self.sparsified, self.data_sparsifier)

class TrainingAwareDataSparsity(pl.callbacks.Callback):
    """Lightning callback that enables in-training sparsity.

    This callback aims to sparsify the model inside lightning module during training.
    **Note that the model is copied and then sparsified, so the existing model is not modified**

    The sparsified model can be used for comparison and can be accessed using
        <callback_obj>.sparsified

    Args:
        data_sparsifier_class (some implemented class of BaseDataSparsifier)
            The data sparsifier object of this class is created when the
            training starts.
            Note: Objects should not be passed in here as they are created
            when the training starts.

        data_sparsifier_args (Dict)
            Dictionary of args to be passed to the data sparsifier.
            Note: data_list arg should be ignored

        data_scheduler_class (some implemented class of BaseDataScheduler)
            The data scheduler of this class is created when the training starts
            Note: Objects should not be passed in here as they are created
            when the training starts.

        data_scheduler_args(Dict)
            Dictionary of args to be passed to the data scheduler.
            **Note: data_sparsifier arg should be ignored as the recipe
            creates and pass sparsifier object into the class**

    Hooks implemented:
        on_train_start()
            Data sparsifier and scheduler objects are created.
            Pytorch model attached to the sparsifier

        on_train_epoch_start()
            Loads the state_dict of the data sparsifier

        on_train_epoch_end()
            1. Copies the model and attaches it to the sparsifier
            2. sparsifier step() and scheduler step()
            3. Dump state_dict of the current sparsifier

        on_train_end()
            squash mask
    """

    def __init__(self, data_sparsifier_class, data_sparsifier_args, data_scheduler_class, data_scheduler_args):
        if False:
            return 10
        super().__init__()
        self.data_sparsifier_class = data_sparsifier_class
        self.data_sparsifier_args = data_sparsifier_args
        self.data_scheduler_class = data_scheduler_class
        self.data_scheduler_args = data_scheduler_args
        self.data_sparsifier: Any = None
        self.data_scheduler: Any = None
        self.sparsified: Optional[torch.nn.Module] = None
        self.data_sparsifier_state_dict: Any = None

    def on_train_start(self, trainer, pl_module) -> None:
        if False:
            return 10
        self.data_sparsifier = self.data_sparsifier_class(**self.data_sparsifier_args)
        self.sparsified = deepcopy(pl_module.model)
        _attach_model_to_data_sparsifier(self.sparsified, self.data_sparsifier)
        args = deepcopy(self.data_scheduler_args)
        args['data_sparsifier'] = self.data_sparsifier
        self.data_scheduler = self.data_scheduler_class(**args)

    def on_train_epoch_start(self, trainer, pl_module):
        if False:
            return 10
        if self.data_sparsifier_state_dict is None:
            return
        self.data_sparsifier.load_state_dict(self.data_sparsifier_state_dict)

    def __create_config_based_on_state(self, pl_module):
        if False:
            i = 10
            return i + 15
        config: Dict = defaultdict()
        if self.data_sparsifier_state_dict is None:
            return config
        for (name, _) in pl_module.model.named_parameters():
            valid_name = _get_valid_name(name)
            config[valid_name] = self.data_sparsifier.data_groups[valid_name]
        return config

    def on_train_epoch_end(self, trainer, pl_module):
        if False:
            while True:
                i = 10
        self.sparsified = deepcopy(pl_module.model)
        config = self.__create_config_based_on_state(pl_module)
        _attach_model_to_data_sparsifier(self.sparsified, self.data_sparsifier, config=config)
        self.data_sparsifier.step()
        self.data_scheduler.step()
        self.data_sparsifier_state_dict = self.data_sparsifier.state_dict()

    def on_train_end(self, trainer, pl_module):
        if False:
            i = 10
            return i + 15
        self.data_sparsifier.squash_mask()