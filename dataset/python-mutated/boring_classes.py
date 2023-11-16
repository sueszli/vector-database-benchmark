from typing import Any, Dict, Iterator, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset
from lightning.fabric.utilities.types import _TORCH_LRSCHEDULER
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import STEP_OUTPUT

class RandomDictDataset(Dataset):
    """
    .. warning::  This is meant for testing/debugging and is experimental.
    """

    def __init__(self, size: int, length: int):
        if False:
            return 10
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        if False:
            print('Hello World!')
        a = self.data[index]
        b = a + 2
        return {'a': a, 'b': b}

    def __len__(self) -> int:
        if False:
            while True:
                i = 10
        return self.len

class RandomDataset(Dataset):
    """
    .. warning::  This is meant for testing/debugging and is experimental.
    """

    def __init__(self, size: int, length: int):
        if False:
            for i in range(10):
                print('nop')
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index: int) -> Tensor:
        if False:
            return 10
        return self.data[index]

    def __len__(self) -> int:
        if False:
            while True:
                i = 10
        return self.len

class RandomIterableDataset(IterableDataset):
    """
    .. warning::  This is meant for testing/debugging and is experimental.
    """

    def __init__(self, size: int, count: int):
        if False:
            return 10
        self.count = count
        self.size = size

    def __iter__(self) -> Iterator[Tensor]:
        if False:
            i = 10
            return i + 15
        for _ in range(self.count):
            yield torch.randn(self.size)

class RandomIterableDatasetWithLen(IterableDataset):
    """
    .. warning::  This is meant for testing/debugging and is experimental.
    """

    def __init__(self, size: int, count: int):
        if False:
            while True:
                i = 10
        self.count = count
        self.size = size

    def __iter__(self) -> Iterator[Tensor]:
        if False:
            for i in range(10):
                print('nop')
        for _ in range(len(self)):
            yield torch.randn(self.size)

    def __len__(self) -> int:
        if False:
            print('Hello World!')
        return self.count

class BoringModel(LightningModule):
    """Testing PL Module.

    Use as follows:
    - subclass
    - modify the behavior for what you want

    .. warning::  This is meant for testing/debugging and is experimental.

    Example::

        class TestModel(BoringModel):
            def training_step(self, ...):
                ...  # do your own thing

    """

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x: Tensor) -> Tensor:
        if False:
            return 10
        return self.layer(x)

    def loss(self, preds: Tensor, labels: Optional[Tensor]=None) -> Tensor:
        if False:
            return 10
        if labels is None:
            labels = torch.ones_like(preds)
        return torch.nn.functional.mse_loss(preds, labels)

    def step(self, batch: Any) -> Tensor:
        if False:
            print('Hello World!')
        output = self(batch)
        return self.loss(output)

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        if False:
            while True:
                i = 10
        return {'loss': self.step(batch)}

    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        if False:
            return 10
        return {'x': self.step(batch)}

    def test_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        if False:
            while True:
                i = 10
        return {'y': self.step(batch)}

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[_TORCH_LRSCHEDULER]]:
        if False:
            print('Hello World!')
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return ([optimizer], [lr_scheduler])

    def train_dataloader(self) -> DataLoader:
        if False:
            print('Hello World!')
        return DataLoader(RandomDataset(32, 64))

    def val_dataloader(self) -> DataLoader:
        if False:
            return 10
        return DataLoader(RandomDataset(32, 64))

    def test_dataloader(self) -> DataLoader:
        if False:
            while True:
                i = 10
        return DataLoader(RandomDataset(32, 64))

    def predict_dataloader(self) -> DataLoader:
        if False:
            for i in range(10):
                print('nop')
        return DataLoader(RandomDataset(32, 64))

class BoringDataModule(LightningDataModule):
    """
    .. warning::  This is meant for testing/debugging and is experimental.
    """

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.random_full = RandomDataset(32, 64 * 4)

    def setup(self, stage: str) -> None:
        if False:
            i = 10
            return i + 15
        if stage == 'fit':
            self.random_train = Subset(self.random_full, indices=range(64))
        if stage in ('fit', 'validate'):
            self.random_val = Subset(self.random_full, indices=range(64, 64 * 2))
        if stage == 'test':
            self.random_test = Subset(self.random_full, indices=range(64 * 2, 64 * 3))
        if stage == 'predict':
            self.random_predict = Subset(self.random_full, indices=range(64 * 3, 64 * 4))

    def train_dataloader(self) -> DataLoader:
        if False:
            i = 10
            return i + 15
        return DataLoader(self.random_train)

    def val_dataloader(self) -> DataLoader:
        if False:
            return 10
        return DataLoader(self.random_val)

    def test_dataloader(self) -> DataLoader:
        if False:
            while True:
                i = 10
        return DataLoader(self.random_test)

    def predict_dataloader(self) -> DataLoader:
        if False:
            for i in range(10):
                print('nop')
        return DataLoader(self.random_predict)

class ManualOptimBoringModel(BoringModel):
    """
    .. warning::  This is meant for testing/debugging and is experimental.
    """

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.automatic_optimization = False

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        if False:
            print('Hello World!')
        opt = self.optimizers()
        assert isinstance(opt, (Optimizer, LightningOptimizer))
        loss = self.step(batch)
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        return loss

class DemoModel(LightningModule):
    """
    .. warning::  This is meant for testing/debugging and is experimental.
    """

    def __init__(self, out_dim: int=10, learning_rate: float=0.02):
        if False:
            while True:
                i = 10
        super().__init__()
        self.l1 = torch.nn.Linear(32, out_dim)
        self.learning_rate = learning_rate

    def forward(self, x: Tensor) -> Tensor:
        if False:
            while True:
                i = 10
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch: Any, batch_nb: int) -> STEP_OUTPUT:
        if False:
            for i in range(10):
                print('nop')
        x = batch
        x = self(x)
        return x.sum()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if False:
            i = 10
            return i + 15
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

class Net(nn.Module):
    """
    .. warning::  This is meant for testing/debugging and is experimental.
    """

    def __init__(self) -> None:
        if False:
            return 10
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: Tensor) -> Tensor:
        if False:
            for i in range(10):
                print('nop')
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)