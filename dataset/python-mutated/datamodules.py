import torch
from lightning.pytorch.core.datamodule import LightningDataModule
from lightning_utilities.core.imports import RequirementCache
from torch.utils.data import DataLoader
from tests_pytorch.helpers.datasets import MNIST, SklearnDataset, TrialMNIST
_SKLEARN_AVAILABLE = RequirementCache('scikit-learn')

class MNISTDataModule(LightningDataModule):

    def __init__(self, data_dir: str='./', batch_size: int=32, use_trials: bool=False) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.dataset_cls = TrialMNIST if use_trials else MNIST

    def prepare_data(self):
        if False:
            i = 10
            return i + 15
        self.dataset_cls(self.data_dir, train=True, download=True)
        self.dataset_cls(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        if False:
            for i in range(10):
                print('nop')
        if stage == 'fit':
            self.mnist_train = self.dataset_cls(self.data_dir, train=True)
        if stage == 'test':
            self.mnist_test = self.dataset_cls(self.data_dir, train=False)

    def train_dataloader(self):
        if False:
            while True:
                i = 10
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        if False:
            i = 10
            return i + 15
        return DataLoader(self.mnist_test, batch_size=self.batch_size, shuffle=False)

class SklearnDataModule(LightningDataModule):

    def __init__(self, sklearn_dataset, x_type, y_type, batch_size: int=10):
        if False:
            while True:
                i = 10
        if not _SKLEARN_AVAILABLE:
            raise ImportError(str(_SKLEARN_AVAILABLE))
        super().__init__()
        self.batch_size = batch_size
        (self._x, self._y) = sklearn_dataset
        self._split_data()
        self._x_type = x_type
        self._y_type = y_type

    def _split_data(self):
        if False:
            print('Hello World!')
        from sklearn.model_selection import train_test_split
        (self.x_train, self.x_test, self.y_train, self.y_test) = train_test_split(self._x, self._y, test_size=0.2, random_state=42)
        (self.x_train, self.x_valid, self.y_train, self.y_valid) = train_test_split(self.x_train, self.y_train, test_size=0.4, random_state=42)

    def train_dataloader(self):
        if False:
            return 10
        return DataLoader(SklearnDataset(self.x_train, self.y_train, self._x_type, self._y_type), batch_size=self.batch_size)

    def val_dataloader(self):
        if False:
            i = 10
            return i + 15
        return DataLoader(SklearnDataset(self.x_valid, self.y_valid, self._x_type, self._y_type), batch_size=self.batch_size)

    def test_dataloader(self):
        if False:
            print('Hello World!')
        return DataLoader(SklearnDataset(self.x_test, self.y_test, self._x_type, self._y_type), batch_size=self.batch_size)

    def predict_dataloader(self):
        if False:
            while True:
                i = 10
        return DataLoader(SklearnDataset(self.x_test, self.y_test, self._x_type, self._y_type), batch_size=self.batch_size)

    @property
    def sample(self):
        if False:
            i = 10
            return i + 15
        return torch.tensor([self._x[0]], dtype=self._x_type)

class ClassifDataModule(SklearnDataModule):

    def __init__(self, num_features=32, length=800, num_classes=3, batch_size=10, n_clusters_per_class=1, n_informative=2):
        if False:
            return 10
        if not _SKLEARN_AVAILABLE:
            raise ImportError(str(_SKLEARN_AVAILABLE))
        from sklearn.datasets import make_classification
        data = make_classification(n_samples=length, n_features=num_features, n_classes=num_classes, n_clusters_per_class=n_clusters_per_class, n_informative=n_informative, random_state=42)
        super().__init__(data, x_type=torch.float32, y_type=torch.long, batch_size=batch_size)

class RegressDataModule(SklearnDataModule):

    def __init__(self, num_features=16, length=800, batch_size=10):
        if False:
            print('Hello World!')
        if not _SKLEARN_AVAILABLE:
            raise ImportError(str(_SKLEARN_AVAILABLE))
        from sklearn.datasets import make_regression
        (x, y) = make_regression(n_samples=length, n_features=num_features, random_state=42)
        y = [[v] for v in y]
        super().__init__((x, y), x_type=torch.float32, y_type=torch.float32, batch_size=batch_size)