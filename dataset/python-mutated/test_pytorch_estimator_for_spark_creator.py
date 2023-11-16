import os
from unittest import TestCase
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import Accuracy
from bigdl.orca.learn.trigger import EveryEpoch
from bigdl.orca.learn.optimizers import Adam
resource_path = os.path.join(os.path.split(__file__)[0], '../../resources')

class TestEstimatorForSparkCreator(TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        ' setup any state tied to the execution of the given method in a\n        class.  setup_method is invoked for every test method of a class.\n        '
        self.sc = init_orca_context(cores=4)

    def tearDown(self):
        if False:
            print('Hello World!')
        ' teardown any state that was previously setup with a setup_method\n        call.\n        '
        stop_orca_context()

    def test_bigdl_pytorch_estimator_dataloader_creator(self):
        if False:
            for i in range(10):
                print('nop')

        class SimpleModel(nn.Module):

            def __init__(self, momentum):
                if False:
                    for i in range(10):
                        print('nop')
                super(SimpleModel, self).__init__()
                self.dense1 = nn.Linear(2, 4)
                self.bn1 = torch.nn.BatchNorm1d(4, momentum=momentum)
                self.dense2 = nn.Linear(4, 1)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = self.dense1(x)
                x = self.bn1(x)
                x = torch.sigmoid(self.dense2(x))
                return x

        def model_creator(config):
            if False:
                return 10
            model = SimpleModel(momentum=config.get('momentum', 0.1))
            return model
        estimator = Estimator.from_torch(model=model_creator, loss=nn.BCELoss(), metrics=[Accuracy()], optimizer=Adam(), config={'momentum': 0.9}, backend='bigdl')

        def get_dataloader(config, batch_size):
            if False:
                print('Hello World!')
            inputs = torch.Tensor([[1, 2], [1, 3], [3, 2], [5, 6], [8, 9], [1, 9]])
            targets = torch.Tensor([[0], [0], [0], [1], [1], [1]])
            data_loader = torch.utils.data.DataLoader(TensorDataset(inputs, targets), batch_size=batch_size, num_workers=config.get('threads', 1))
            return data_loader
        estimator.fit(data=get_dataloader, epochs=2, batch_size=2, validation_data=get_dataloader, checkpoint_trigger=EveryEpoch())
        estimator.evaluate(data=get_dataloader, batch_size=2)
        model = estimator.get_model()
        assert isinstance(model, nn.Module)
if __name__ == '__main__':
    pytest.main([__file__])