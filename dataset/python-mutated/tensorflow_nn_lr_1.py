import logging
from typing import List
import numpy as np
import pandas as pd
import click
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model, Input
from bigdl.ppml.fl.estimator import Estimator
from bigdl.ppml.fl.nn.fl_context import init_fl_context
from bigdl.ppml.fl.psi.psi_client import PSI
fmt = '%(asctime)s %(levelname)s {%(module)s:%(lineno)d} - %(message)s'
logging.basicConfig(format=fmt, level=logging.INFO)

def build_client_model(feature_num):
    if False:
        print('Hello World!')
    inputs = Input(shape=feature_num)
    outputs = Dense(1)(inputs)
    return Model(inputs=inputs, outputs=outputs, name='vfl_client_model')

def build_server_model():
    if False:
        print('Hello World!')
    input1 = Input(shape=1)
    input2 = Input(shape=1)
    x = tf.stack([input1, input2])
    x = tf.reduce_sum(x, 0)
    outputs = tf.sigmoid(x)
    return Model(inputs=[input1, input2], outputs=outputs, name='vfl_server_model')

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
    df_x = df_x.drop('ID', 1)
    df_y = df_train['Outcome']
    x = df_x.to_numpy(dtype='float32')
    y = np.expand_dims(df_y.to_numpy(dtype='float32'), axis=1)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    if load_model:
        model = tf.keras.models.load_model('/tmp/tensorflow_client_model_1.pt')
        ppl = Estimator.from_keras(client_model=model, loss_fn=loss_fn, optimizer_cls=tf.keras.optimizers.SGD, optimizer_args={'learning_rate': 0.0001}, server_model_path='/tmp/tensorflow_server_model', client_model_path='/tmp/tensorflow_client_model_1.pt')
        ppl.load_server_model('/tmp/tensorflow_server_model')
        response = ppl.fit(x, y, 5)
    else:
        model = build_client_model(4)
        server_model = build_server_model()
        ppl = Estimator.from_keras(client_model=model, loss_fn=loss_fn, optimizer_cls=tf.keras.optimizers.SGD, optimizer_args={'learning_rate': 0.0001}, server_model=server_model, server_model_path='/tmp/tensorflow_server_model', client_model_path='/tmp/tensorflow_client_model_1.pt')
        response = ppl.fit(x, y, 5)
    result = ppl.predict(x)
    print(result[:5])
if __name__ == '__main__':
    run_client()