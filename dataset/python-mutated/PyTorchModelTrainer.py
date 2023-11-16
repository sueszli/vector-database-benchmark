import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset
from freqtrade.freqai.torch.PyTorchDataConvertor import PyTorchDataConvertor
from freqtrade.freqai.torch.PyTorchTrainerInterface import PyTorchTrainerInterface
from .datasets import WindowDataset
logger = logging.getLogger(__name__)

class PyTorchModelTrainer(PyTorchTrainerInterface):

    def __init__(self, model: nn.Module, optimizer: Optimizer, criterion: nn.Module, device: str, data_convertor: PyTorchDataConvertor, model_meta_data: Dict[str, Any]={}, window_size: int=1, tb_logger: Any=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        :param model: The PyTorch model to be trained.\n        :param optimizer: The optimizer to use for training.\n        :param criterion: The loss function to use for training.\n        :param device: The device to use for training (e.g. 'cpu', 'cuda').\n        :param init_model: A dictionary containing the initial model/optimizer\n            state_dict and model_meta_data saved by self.save() method.\n        :param model_meta_data: Additional metadata about the model (optional).\n        :param data_convertor: convertor from pd.DataFrame to torch.tensor.\n        :param n_steps: used to calculate n_epochs. The number of training iterations to run.\n            iteration here refers to the number of times optimizer.step() is called.\n            ignored if n_epochs is set.\n        :param n_epochs: The maximum number batches to use for evaluation.\n        :param batch_size: The size of the batches to use during training.\n        "
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.model_meta_data = model_meta_data
        self.device = device
        self.n_epochs: Optional[int] = kwargs.get('n_epochs', 10)
        self.n_steps: Optional[int] = kwargs.get('n_steps', None)
        if self.n_steps is None and (not self.n_epochs):
            raise Exception('Either `n_steps` or `n_epochs` should be set.')
        self.batch_size: int = kwargs.get('batch_size', 64)
        self.data_convertor = data_convertor
        self.window_size: int = window_size
        self.tb_logger = tb_logger
        self.test_batch_counter = 0

    def fit(self, data_dictionary: Dict[str, pd.DataFrame], splits: List[str]):
        if False:
            i = 10
            return i + 15
        '\n        :param data_dictionary: the dictionary constructed by DataHandler to hold\n        all the training and test data/labels.\n        :param splits: splits to use in training, splits must contain "train",\n        optional "test" could be added by setting freqai.data_split_parameters.test_size > 0\n        in the config file.\n\n         - Calculates the predicted output for the batch using the PyTorch model.\n         - Calculates the loss between the predicted and actual output using a loss function.\n         - Computes the gradients of the loss with respect to the model\'s parameters using\n           backpropagation.\n         - Updates the model\'s parameters using an optimizer.\n        '
        self.model.train()
        data_loaders_dictionary = self.create_data_loaders_dictionary(data_dictionary, splits)
        n_obs = len(data_dictionary['train_features'])
        n_epochs = self.n_epochs or self.calc_n_epochs(n_obs=n_obs)
        batch_counter = 0
        for _ in range(n_epochs):
            for (_, batch_data) in enumerate(data_loaders_dictionary['train']):
                (xb, yb) = batch_data
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                yb_pred = self.model(xb)
                loss = self.criterion(yb_pred, yb)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                self.tb_logger.log_scalar('train_loss', loss.item(), batch_counter)
                batch_counter += 1
            if 'test' in splits:
                self.estimate_loss(data_loaders_dictionary, 'test')

    @torch.no_grad()
    def estimate_loss(self, data_loader_dictionary: Dict[str, DataLoader], split: str) -> None:
        if False:
            while True:
                i = 10
        self.model.eval()
        for (_, batch_data) in enumerate(data_loader_dictionary[split]):
            (xb, yb) = batch_data
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            yb_pred = self.model(xb)
            loss = self.criterion(yb_pred, yb)
            self.tb_logger.log_scalar(f'{split}_loss', loss.item(), self.test_batch_counter)
            self.test_batch_counter += 1
        self.model.train()

    def create_data_loaders_dictionary(self, data_dictionary: Dict[str, pd.DataFrame], splits: List[str]) -> Dict[str, DataLoader]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Converts the input data to PyTorch tensors using a data loader.\n        '
        data_loader_dictionary = {}
        for split in splits:
            x = self.data_convertor.convert_x(data_dictionary[f'{split}_features'], self.device)
            y = self.data_convertor.convert_y(data_dictionary[f'{split}_labels'], self.device)
            dataset = TensorDataset(x, y)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=0)
            data_loader_dictionary[split] = data_loader
        return data_loader_dictionary

    def calc_n_epochs(self, n_obs: int) -> int:
        if False:
            while True:
                i = 10
        '\n        Calculates the number of epochs required to reach the maximum number\n        of iterations specified in the model training parameters.\n\n        the motivation here is that `n_steps` is easier to optimize and keep stable,\n        across different n_obs - the number of data points.\n        '
        assert isinstance(self.n_steps, int), 'Either `n_steps` or `n_epochs` should be set.'
        n_batches = n_obs // self.batch_size
        n_epochs = min(self.n_steps // n_batches, 1)
        if n_epochs <= 10:
            logger.warning(f'Setting low n_epochs: {n_epochs}. Please consider increasing `n_steps` hyper-parameter.')
        return n_epochs

    def save(self, path: Path):
        if False:
            print('Hello World!')
        '\n        - Saving any nn.Module state_dict\n        - Saving model_meta_data, this dict should contain any additional data that the\n          user needs to store. e.g. class_names for classification models.\n        '
        torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'model_meta_data': self.model_meta_data, 'pytrainer': self}, path)

    def load(self, path: Path):
        if False:
            print('Hello World!')
        checkpoint = torch.load(path)
        return self.load_from_checkpoint(checkpoint)

    def load_from_checkpoint(self, checkpoint: Dict):
        if False:
            return 10
        '\n        when using continual_learning, DataDrawer will load the dictionary\n        (containing state dicts and model_meta_data) by calling torch.load(path).\n        you can access this dict from any class that inherits IFreqaiModel by calling\n        get_init_model method.\n        '
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model_meta_data = checkpoint['model_meta_data']
        return self

class PyTorchTransformerTrainer(PyTorchModelTrainer):
    """
    Creating a trainer for the Transformer model.
    """

    def create_data_loaders_dictionary(self, data_dictionary: Dict[str, pd.DataFrame], splits: List[str]) -> Dict[str, DataLoader]:
        if False:
            i = 10
            return i + 15
        '\n        Converts the input data to PyTorch tensors using a data loader.\n        '
        data_loader_dictionary = {}
        for split in splits:
            x = self.data_convertor.convert_x(data_dictionary[f'{split}_features'], self.device)
            y = self.data_convertor.convert_y(data_dictionary[f'{split}_labels'], self.device)
            dataset = WindowDataset(x, y, self.window_size)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=0)
            data_loader_dictionary[split] = data_loader
        return data_loader_dictionary