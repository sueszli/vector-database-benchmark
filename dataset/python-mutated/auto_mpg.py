import math
import bigdl.orca.data.pandas
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.data.transformer import *
from bigdl.orca.learn.tf2.estimator import Estimator
import tensorflow.compat.v1 as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
init_orca_context(memory='4g')
OrcaContext.pandas_read_backend = 'pandas'
path = 'auto-mpg.csv'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
data_shard = bigdl.orca.data.pandas.read_csv(file_path=path, names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True)

def drop_na(df):
    if False:
        while True:
            i = 10
    df = df.dropna()
    return df
data_shard = data_shard.transform_shard(drop_na)

def generate_extra_cols(df):
    if False:
        while True:
            i = 10
    origin = df.pop('Origin')
    df['USA'] = (origin == 1) * 1.0
    df['Europe'] = (origin == 2) * 1.0
    df['Japan'] = (origin == 3) * 1.0
    return df
data_shard = data_shard.transform_shard(generate_extra_cols)
column = data_shard.get_schema()['columns']
scaler = MinMaxScaler(inputCol=list(column[1:]), outputCol='scaled_vec')
data_shard = scaler.fit_transform(data_shard)

def split_train_test(df):
    if False:
        for i in range(10):
            print('nop')
    train_df = df.sample(frac=0.8, random_state=0)
    test_df = df.drop(train_df.index)
    return (train_df, test_df)
(shards_train, shards_val) = data_shard.transform_shard(split_train_test).split()

def build_model(config):
    if False:
        i = 10
        return i + 15
    model = Sequential([Dense(64, activation=tf.nn.relu, input_shape=[9]), Dense(64, activation=tf.nn.relu), Dense(1)])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error', 'mean_squared_error'])
    return model
EPOCHS = 1000
batch_size = 16
train_steps = math.ceil(len(shards_train) / batch_size)
est = Estimator.from_keras(model_creator=build_model)
est.fit(data=shards_train, batch_size=batch_size, epochs=EPOCHS, steps_per_epoch=train_steps, feature_cols=['scaled_vec'], label_cols=['MPG'])
stop_orca_context()