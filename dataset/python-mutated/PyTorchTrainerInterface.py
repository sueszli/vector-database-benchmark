from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List
import pandas as pd
import torch
from torch import nn

class PyTorchTrainerInterface(ABC):

    @abstractmethod
    def fit(self, data_dictionary: Dict[str, pd.DataFrame], splits: List[str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        :param data_dictionary: the dictionary constructed by DataHandler to hold\n        all the training and test data/labels.\n        :param splits: splits to use in training, splits must contain "train",\n        optional "test" could be added by setting freqai.data_split_parameters.test_size > 0\n        in the config file.\n\n         - Calculates the predicted output for the batch using the PyTorch model.\n         - Calculates the loss between the predicted and actual output using a loss function.\n         - Computes the gradients of the loss with respect to the model\'s parameters using\n           backpropagation.\n         - Updates the model\'s parameters using an optimizer.\n        '

    @abstractmethod
    def save(self, path: Path) -> None:
        if False:
            return 10
        '\n        - Saving any nn.Module state_dict\n        - Saving model_meta_data, this dict should contain any additional data that the\n          user needs to store. e.g class_names for classification models.\n        '

    def load(self, path: Path) -> nn.Module:
        if False:
            i = 10
            return i + 15
        '\n        :param path: path to zip file.\n        :returns: pytorch model.\n        '
        checkpoint = torch.load(path)
        return self.load_from_checkpoint(checkpoint)

    @abstractmethod
    def load_from_checkpoint(self, checkpoint: Dict) -> nn.Module:
        if False:
            print('Hello World!')
        '\n        when using continual_learning, DataDrawer will load the dictionary\n        (containing state dicts and model_meta_data) by calling torch.load(path).\n        you can access this dict from any class that inherits IFreqaiModel by calling\n        get_init_model method.\n        :checkpoint checkpoint: dict containing the model & optimizer state dicts,\n        model_meta_data, etc..\n        '