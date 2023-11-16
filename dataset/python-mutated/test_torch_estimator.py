import torch
from torch import nn
import pytest
from unittest import TestCase
from bigdl.orca.torch import TorchModel, TorchLoss
from bigdl.dllib.nncontext import *
from torch.utils.data import TensorDataset, DataLoader
from bigdl.dllib.estimator import *
from bigdl.dllib.keras.optimizers import Adam
from bigdl.dllib.optim.optimizer import MaxEpoch, EveryEpoch
from bigdl.dllib.keras.metrics import Accuracy
from bigdl.dllib.feature.common import FeatureSet

class TestPytorch(TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        ' setup any state tied to the execution of the given method in a\n        class.  setup_method is invoked for every test method of a class.\n        '
        self.sc = init_spark_on_local(4)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        ' teardown any state that was previously setup with a setup_method\n        call.\n        '
        self.sc.stop()

    def test_train_model_with_bn_creator(self):
        if False:
            i = 10
            return i + 15

        class SimpleTorchModel(nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super(SimpleTorchModel, self).__init__()
                self.dense1 = nn.Linear(2, 4)
                self.bn1 = torch.nn.BatchNorm1d(4)
                self.dense2 = nn.Linear(4, 1)

            def forward(self, x):
                if False:
                    return 10
                x = self.dense1(x)
                x = self.bn1(x)
                x = torch.sigmoid(self.dense2(x))
                return x
        self.sc.stop()
        self.sc = init_nncontext()
        torch_model = SimpleTorchModel()
        loss_fn = torch.nn.BCELoss()
        az_model = TorchModel.from_pytorch(torch_model)
        zoo_loss = TorchLoss.from_pytorch(loss_fn)

        def train_dataloader():
            if False:
                i = 10
                return i + 15
            inputs = torch.Tensor([[1, 2], [1, 3], [3, 2], [5, 6], [8, 9], [1, 9]])
            targets = torch.Tensor([[0], [0], [0], [1], [1], [1]])
            return DataLoader(TensorDataset(inputs, targets), batch_size=2)
        train_featureset = FeatureSet.pytorch_dataloader(train_dataloader)
        val_featureset = FeatureSet.pytorch_dataloader(train_dataloader)
        zooOptimizer = Adam()
        estimator = Estimator(az_model, optim_methods=zooOptimizer)
        estimator.train_minibatch(train_featureset, zoo_loss, end_trigger=MaxEpoch(4), checkpoint_trigger=EveryEpoch(), validation_set=val_featureset, validation_method=[Accuracy()])
        trained_model = az_model.to_pytorch()
if __name__ == '__main__':
    pytest.main([__file__])