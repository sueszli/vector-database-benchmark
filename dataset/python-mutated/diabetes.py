import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import bigdl.orca.data.pandas
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.tf2.estimator import Estimator
init_orca_context(cluster_mode='local', cores=4, memory='3g')
path = 'pima-indians-diabetes.csv'
data_shard = bigdl.orca.data.pandas.read_csv(path, header=None)
column = list(data_shard.get_schema()['columns'])

def model_creator(config):
    if False:
        i = 10
        return i + 15
    model = Sequential()
    model.add(Dense(12, input_shape=(8,), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
data_shard = data_shard.assembleFeatureLabelCols(featureCols=column[:-1], labelCols=list(column[-1]))
batch_size = 16
train_steps = math.ceil(len(data_shard) / batch_size)
est = Estimator.from_keras(model_creator=model_creator)
est.fit(data=data_shard, batch_size=batch_size, epochs=150, steps_per_epoch=train_steps)
stop_orca_context()