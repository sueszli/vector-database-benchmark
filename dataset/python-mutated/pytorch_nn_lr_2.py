import logging
import numpy as np
import pandas as pd
import click
import torch
from torch import nn
from bigdl.ppml.fl.estimator import Estimator
from bigdl.ppml.fl.nn.fl_context import init_fl_context
from bigdl.ppml.fl.psi.psi_client import PSI
fmt = '%(asctime)s %(levelname)s {%(module)s:%(lineno)d} - %(message)s'
logging.basicConfig(format=fmt, level=logging.INFO)

class LocalModel(nn.Module):

    def __init__(self, num_feature) -> None:
        if False:
            return 10
        super().__init__()
        self.dense = nn.Linear(num_feature, 1)

    def forward(self, x):
        if False:
            return 10
        x = self.dense(x)
        return x

@click.command()
@click.option('--load_model', default=False)
@click.option('--data_path', default='./data/diabetes-vfl-2.csv')
def run_client(load_model, data_path):
    if False:
        print('Hello World!')
    init_fl_context(2)
    df_train = pd.read_csv(data_path)
    df_train['ID'] = df_train['ID'].astype(str)
    psi = PSI()
    intersection = psi.get_intersection(list(df_train['ID']))
    df_train = df_train[df_train['ID'].isin(intersection)]
    df_x = df_train
    x = df_x.to_numpy(dtype='float32')
    y = None
    loss_fn = nn.BCELoss()
    if load_model:
        model = torch.load('/tmp/pytorch_client_model_2.pt')
        ppl = Estimator.from_torch(client_model=model, loss_fn=loss_fn, optimizer_cls=torch.optim.SGD, optimizer_args={'lr': 0.0001}, client_model_path='/tmp/pytorch_client_model_2.pt')
        response = ppl.fit(x, y, 5)
    else:
        model = LocalModel(len(df_x.columns))
        ppl = Estimator.from_torch(client_model=model, loss_fn=loss_fn, optimizer_cls=torch.optim.SGD, optimizer_args={'lr': 0.0001}, client_model_path='/tmp/pytorch_client_model_2.pt')
        response = ppl.fit(x, y, 5)
    result = ppl.predict(x)
    print(result[:5])
if __name__ == '__main__':
    run_client()