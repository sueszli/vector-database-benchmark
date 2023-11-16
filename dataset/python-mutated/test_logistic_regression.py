import multiprocessing
import threading
from multiprocessing import Process
import unittest
import numpy as np
import pandas as pd
import os
import torch
from bigdl.ppml.fl.nn.fl_context import init_fl_context
from bigdl.ppml.fl.nn.pytorch.utils import set_one_like_parameter
from bigdl.ppml.fl.nn.fl_server import FLServer
from torch import Tensor, nn
import logging
from bigdl.ppml.fl.estimator import Estimator
from bigdl.ppml.fl.utils import FLTest
from typing import List
resource_path = os.path.join(os.path.dirname(__file__), '../../resources')

def mock_process(data_train, target, client_id, upload_server_model):
    if False:
        i = 10
        return i + 15
    init_fl_context(client_id, target)
    df_train = pd.read_csv(os.path.join(resource_path, data_train))
    if 'Outcome' in df_train:
        df_x = df_train.drop('Outcome', 1)
        df_y = df_train['Outcome']
    else:
        df_x = df_train
        df_y = None
    x = df_x.to_numpy(dtype='float32')
    y = np.expand_dims(df_y.to_numpy(dtype='float32'), axis=1) if df_y is not None else None
    model = LogisticRegressionNetwork1(len(df_x.columns))
    set_one_like_parameter(model)
    loss_fn = nn.BCELoss()
    server_model = LogisticRegressionNetwork2() if upload_server_model else None
    logging.info('Creating FL Pytorch Estimator')
    ppl = Estimator.from_torch(client_model=model, loss_fn=loss_fn, optimizer_cls=torch.optim.SGD, optimizer_args={'lr': 0.001}, server_model=server_model)
    logging.info('Starting training')
    response = ppl.fit(x, y)
    result = ppl.predict(x)
    logging.info(response)
    return ppl

class TestLogisticRegression(FLTest):
    fmt = '%(asctime)s %(levelname)s {%(module)s:%(lineno)d} - %(message)s'
    logging.basicConfig(format=fmt, level=logging.DEBUG)

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            for i in range(10):
                print('nop')
        multiprocessing.set_start_method('spawn')

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        self.fl_server = FLServer(client_num=2)
        self.fl_server.set_port(self.port)
        self.fl_server.build()
        self.fl_server.start()

    def tearDown(self) -> None:
        if False:
            return 10
        self.fl_server.stop()

    def test_two_party_logistic_regression(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        df_train = pd.read_csv(os.path.join(resource_path, 'pima-indians-diabetes.csv'))
        df_x1 = df_train[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness']]
        df_x2 = df_train[['Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
        df_y = df_train['Outcome']
        model = LogisticRegressionNetwork(len(df_x1.columns), len(df_x2.columns))
        set_one_like_parameter(model)
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        x1 = torch.tensor(df_x1.to_numpy(dtype='float32'))
        x2 = torch.tensor(df_x2.to_numpy(dtype='float32'))
        y = torch.tensor(np.expand_dims(df_y.to_numpy(dtype='float32'), axis=1))
        assert len(x1) == len(x2) == len(y)
        batch_size = 4
        pytorch_loss_list = []
        (i, size) = (0, len(x1))
        while i + batch_size < len(x1):
            (X1, X2, Y) = (x1[i:i + batch_size], x2[i:i + batch_size], y[i:i + batch_size])
            pred = model(X1, X2)
            loss = loss_fn(pred, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i += batch_size
            if i % 100 == 0:
                (loss, current) = (loss.item(), i)
                print(f'loss: {loss:>7f}  [{current:>3d}/{size:>3d}]')
                pytorch_loss_list.append(np.array(loss))
        mock_party2 = Process(target=mock_process, args=('diabetes-vfl-2.csv', self.target, 2, False))
        mock_party2.start()
        ppl = mock_process(data_train='diabetes-vfl-1.csv', target=self.target, client_id=1, upload_server_model=True)
        mock_party2.join()
        assert np.allclose(pytorch_loss_list, ppl.loss_history), 'Validation failed, correctness of PPML and native Pytorch not the same'

class LogisticRegressionNetwork1(nn.Module):

    def __init__(self, num_feature) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self.dense = nn.Linear(num_feature, 1)

    def forward(self, x):
        if False:
            while True:
                i = 10
        x = self.dense(x)
        return x

class LogisticRegressionNetwork2(nn.Module):

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: List[Tensor]):
        if False:
            print('Hello World!')
        x = torch.stack(x)
        x = torch.sum(x, dim=0)
        x = self.sigmoid(x)
        return x

class LogisticRegressionNetwork(nn.Module):

    def __init__(self, num_feature1, num_feature2) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self.dense1 = nn.Linear(num_feature1, 1)
        self.dense2 = nn.Linear(num_feature2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        if False:
            i = 10
            return i + 15
        x1 = self.dense1(x1)
        x2 = self.dense2(x2)
        x = torch.stack([x1, x2])
        x = torch.sum(x, dim=0)
        x = self.sigmoid(x)
        return x
if __name__ == '__main__':
    unittest.main()