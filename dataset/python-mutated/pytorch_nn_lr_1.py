import logging
from typing import List
import numpy as np
import pandas as pd
import click
import torch
from torch import Tensor, nn
from bigdl.ppml.fl.estimator import Estimator
from bigdl.ppml.fl.nn.fl_context import init_fl_context
from bigdl.ppml.fl.psi.psi_client import PSI
fmt = '%(asctime)s %(levelname)s {%(module)s:%(lineno)d} - %(message)s'
logging.basicConfig(format=fmt, level=logging.INFO)

class LocalModel(nn.Module):

    def __init__(self, num_feature) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self.dense = nn.Linear(num_feature, 1)

    def forward(self, x):
        if False:
            while True:
                i = 10
        x = self.dense(x)
        return x

class ServerModel(nn.Module):

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: List[Tensor]):
        if False:
            i = 10
            return i + 15
        x = torch.stack(x)
        x = torch.sum(x, dim=0)
        x = self.sigmoid(x)
        return x

@click.command()
@click.option('--load_model', default=False)
@click.option('--data_path', default='./data/diabetes-vfl-1.csv')
def run_client(load_model, data_path):
    if False:
        while True:
            i = 10
    init_fl_context(1)
    df_train = pd.read_csv(data_path)
    df_train['ID'] = df_train['ID'].astype(str)
    psi = PSI()
    intersection = psi.get_intersection(list(df_train['ID']))
    df_train = df_train[df_train['ID'].isin(intersection)]
    df_x = df_train.drop('Outcome', 1)
    df_y = df_train['Outcome']
    x = df_x.to_numpy(dtype='float32')
    y = np.expand_dims(df_y.to_numpy(dtype='float32'), axis=1)
    loss_fn = nn.BCELoss()
    if load_model:
        model = torch.load('/tmp/pytorch_client_model_1.pt')
        ppl = Estimator.from_torch(client_model=model, loss_fn=loss_fn, optimizer_cls=torch.optim.SGD, optimizer_args={'lr': 0.0001}, server_model_path='/tmp/pytorch_server_model', client_model_path='/tmp/pytorch_client_model_1.pt')
        ppl.load_server_model('/tmp/pytorch_server_model')
        response = ppl.fit(x, y, 5)
    else:
        model = LocalModel(len(df_x.columns))
        server_model = ServerModel()
        ppl = Estimator.from_torch(client_model=model, loss_fn=loss_fn, optimizer_cls=torch.optim.SGD, optimizer_args={'lr': 0.0001}, server_model=server_model, server_model_path='/tmp/pytorch_server_model', client_model_path='/tmp/pytorch_client_model_1.pt')
        response = ppl.fit(x, y, 5)
    result = ppl.predict(x)
    print(result[:5])
if __name__ == '__main__':
    run_client()