"""
Title: Traffic forecasting using graph neural networks and LSTM
Author: [Arash Khodadadi](https://www.linkedin.com/in/arash-khodadadi-08a02490/)
Date created: 2021/12/28
Last modified: 2021/12/28
Description: This example demonstrates how to do timeseries forecasting over graphs.
Accelerator: GPU
"""
'\n## Introduction\n\nThis example shows how to forecast traffic condition using graph neural networks and LSTM.\nSpecifically, we are interested in predicting the future values of the traffic speed given\na history of the traffic speed for a collection of road segments.\n\nOne popular method to\nsolve this problem is to consider each road segment\'s traffic speed as a separate\ntimeseries and predict the future values of each timeseries\nusing the past values of the same timeseries.\n\nThis method, however, ignores the dependency of the traffic speed of one road segment on\nthe neighboring segments. To be able to take into account the complex interactions between\nthe traffic speed on a collection of neighboring roads, we can define the traffic network\nas a graph and consider the traffic speed as a signal on this graph. In this example,\nwe implement a neural network architecture which can process timeseries data over a graph.\nWe first show how to process the data and create a\n[tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) for\nforecasting over graphs. Then, we implement a model which uses graph convolution and\nLSTM layers to perform forecasting over a graph.\n\nThe data processing and the model architecture are inspired by this paper:\n\nYu, Bing, Haoteng Yin, and Zhanxing Zhu. "Spatio-temporal graph convolutional networks:\na deep learning framework for traffic forecasting." Proceedings of the 27th International\nJoint Conference on Artificial Intelligence, 2018.\n([github](https://github.com/VeritasYin/STGCN_IJCAI-18))\n'
'\n## Setup\n'
import pandas as pd
import numpy as np
import os
import typing
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers
from keras.utils import timeseries_dataset_from_array
'\n## Data preparation\n'
'\n### Data description\n\nWe use a real-world traffic speed dataset named `PeMSD7`. We use the version\ncollected and prepared by [Yu et al., 2018](https://arxiv.org/abs/1709.04875)\nand available\n[here](https://github.com/VeritasYin/STGCN_IJCAI-18/tree/master/dataset).\n\nThe data consists of two files:\n\n- `PeMSD7_W_228.csv` contains the distances between 228\nstations across the District 7 of California.\n- `PeMSD7_V_228.csv` contains traffic\nspeed collected for those stations in the weekdays of May and June of 2012.\n\nThe full description of the dataset can be found in\n[Yu et al., 2018](https://arxiv.org/abs/1709.04875).\n'
'\n### Loading data\n'
url = 'https://github.com/VeritasYin/STGCN_IJCAI-18/raw/master/dataset/PeMSD7_Full.zip'
data_dir = keras.utils.get_file(origin=url, extract=True, archive_format='zip')
data_dir = data_dir.rstrip('PeMSD7_Full.zip')
route_distances = pd.read_csv(os.path.join(data_dir, 'PeMSD7_W_228.csv'), header=None).to_numpy()
speeds_array = pd.read_csv(os.path.join(data_dir, 'PeMSD7_V_228.csv'), header=None).to_numpy()
print(f'route_distances shape={route_distances.shape}')
print(f'speeds_array shape={speeds_array.shape}')
'\n### sub-sampling roads\n\nTo reduce the problem size and make the training faster, we will only\nwork with a sample of 26 roads out of the 228 roads in the dataset.\nWe have chosen the roads by starting from road 0, choosing the 5 closest\nroads to it, and continuing this process until we get 25 roads. You can choose\nany other subset of the roads. We chose the roads in this way to increase the likelihood\nof having roads with correlated speed timeseries.\n`sample_routes` contains the IDs of the selected roads.\n'
sample_routes = [0, 1, 4, 7, 8, 11, 15, 108, 109, 114, 115, 118, 120, 123, 124, 126, 127, 129, 130, 132, 133, 136, 139, 144, 147, 216]
route_distances = route_distances[np.ix_(sample_routes, sample_routes)]
speeds_array = speeds_array[:, sample_routes]
print(f'route_distances shape={route_distances.shape}')
print(f'speeds_array shape={speeds_array.shape}')
'\n### Data visualization\n\nHere are the timeseries of the traffic speed for two of the routes:\n'
plt.figure(figsize=(18, 6))
plt.plot(speeds_array[:, [0, -1]])
plt.legend(['route_0', 'route_25'])
'\nWe can also visualize the correlation between the timeseries in different routes.\n'
plt.figure(figsize=(8, 8))
plt.matshow(np.corrcoef(speeds_array.T), 0)
plt.xlabel('road number')
plt.ylabel('road number')
'\nUsing this correlation heatmap, we can see that for example the speed in\nroutes 4, 5, 6 are highly correlated.\n'
'\n### Splitting and normalizing data\n\nNext, we split the speed values array into train/validation/test sets,\nand normalize the resulting arrays:\n'
(train_size, val_size) = (0.5, 0.2)

def preprocess(data_array: np.ndarray, train_size: float, val_size: float):
    if False:
        for i in range(10):
            print('nop')
    'Splits data into train/val/test sets and normalizes the data.\n\n    Args:\n        data_array: ndarray of shape `(num_time_steps, num_routes)`\n        train_size: A float value between 0.0 and 1.0 that represent the proportion of the dataset\n            to include in the train split.\n        val_size: A float value between 0.0 and 1.0 that represent the proportion of the dataset\n            to include in the validation split.\n\n    Returns:\n        `train_array`, `val_array`, `test_array`\n    '
    num_time_steps = data_array.shape[0]
    (num_train, num_val) = (int(num_time_steps * train_size), int(num_time_steps * val_size))
    train_array = data_array[:num_train]
    (mean, std) = (train_array.mean(axis=0), train_array.std(axis=0))
    train_array = (train_array - mean) / std
    val_array = (data_array[num_train:num_train + num_val] - mean) / std
    test_array = (data_array[num_train + num_val:] - mean) / std
    return (train_array, val_array, test_array)
(train_array, val_array, test_array) = preprocess(speeds_array, train_size, val_size)
print(f'train set size: {train_array.shape}')
print(f'validation set size: {val_array.shape}')
print(f'test set size: {test_array.shape}')
'\n### Creating TensorFlow Datasets\n\nNext, we create the datasets for our forecasting problem. The forecasting problem\ncan be stated as follows: given a sequence of the\nroad speed values at times `t+1, t+2, ..., t+T`, we want to predict the future values of\nthe roads speed for times `t+T+1, ..., t+T+h`. So for each time `t` the inputs to our\nmodel are `T` vectors each of size `N` and the targets are `h` vectors each of size `N`,\nwhere `N` is the number of roads.\n'
'\nWe use the Keras built-in function\n[`timeseries_dataset_from_array()`](https://www.tensorflow.org/api_docs/python/tf/keras/utils/timeseries_dataset_from_array).\nThe function `create_tf_dataset()` below takes as input a `numpy.ndarray` and returns a\n`tf.data.Dataset`. In this function `input_sequence_length=T` and `forecast_horizon=h`.\n\nThe argument `multi_horizon` needs more explanation. Assume `forecast_horizon=3`.\nIf `multi_horizon=True` then the model will make a forecast for time steps\n`t+T+1, t+T+2, t+T+3`. So the target will have shape `(T,3)`. But if\n`multi_horizon=False`, the model will make a forecast only for time step `t+T+3` and\nso the target will have shape `(T, 1)`.\n\nYou may notice that the input tensor in each batch has shape\n`(batch_size, input_sequence_length, num_routes, 1)`. The last dimension is added to\nmake the model more general: at each time step, the input features for each raod may\ncontain multiple timeseries. For instance, one might want to use temperature timeseries\nin addition to historical values of the speed as input features. In this example,\nhowever, the last dimension of the input is always 1.\n\nWe use the last 12 values of the speed in each road to forecast the speed for 3 time\nsteps ahead:\n'
batch_size = 64
input_sequence_length = 12
forecast_horizon = 3
multi_horizon = False

def create_tf_dataset(data_array: np.ndarray, input_sequence_length: int, forecast_horizon: int, batch_size: int=128, shuffle=True, multi_horizon=True):
    if False:
        print('Hello World!')
    'Creates tensorflow dataset from numpy array.\n\n    This function creates a dataset where each element is a tuple `(inputs, targets)`.\n    `inputs` is a Tensor\n    of shape `(batch_size, input_sequence_length, num_routes, 1)` containing\n    the `input_sequence_length` past values of the timeseries for each node.\n    `targets` is a Tensor of shape `(batch_size, forecast_horizon, num_routes)`\n    containing the `forecast_horizon`\n    future values of the timeseries for each node.\n\n    Args:\n        data_array: np.ndarray with shape `(num_time_steps, num_routes)`\n        input_sequence_length: Length of the input sequence (in number of timesteps).\n        forecast_horizon: If `multi_horizon=True`, the target will be the values of the timeseries for 1 to\n            `forecast_horizon` timesteps ahead. If `multi_horizon=False`, the target will be the value of the\n            timeseries `forecast_horizon` steps ahead (only one value).\n        batch_size: Number of timeseries samples in each batch.\n        shuffle: Whether to shuffle output samples, or instead draw them in chronological order.\n        multi_horizon: See `forecast_horizon`.\n\n    Returns:\n        A tf.data.Dataset instance.\n    '
    inputs = timeseries_dataset_from_array(np.expand_dims(data_array[:-forecast_horizon], axis=-1), None, sequence_length=input_sequence_length, shuffle=False, batch_size=batch_size)
    target_offset = input_sequence_length if multi_horizon else input_sequence_length + forecast_horizon - 1
    target_seq_length = forecast_horizon if multi_horizon else 1
    targets = timeseries_dataset_from_array(data_array[target_offset:], None, sequence_length=target_seq_length, shuffle=False, batch_size=batch_size)
    dataset = tf.data.Dataset.zip((inputs, targets))
    if shuffle:
        dataset = dataset.shuffle(100)
    return dataset.prefetch(16).cache()
(train_dataset, val_dataset) = (create_tf_dataset(data_array, input_sequence_length, forecast_horizon, batch_size) for data_array in [train_array, val_array])
test_dataset = create_tf_dataset(test_array, input_sequence_length, forecast_horizon, batch_size=test_array.shape[0], shuffle=False, multi_horizon=multi_horizon)
'\n### Roads Graph\n\nAs mentioned before, we assume that the road segments form a graph.\nThe `PeMSD7` dataset has the road segments distance. The next step\nis to create the graph adjacency matrix from these distances. Following\n[Yu et al., 2018](https://arxiv.org/abs/1709.04875) (equation 10) we assume there\nis an edge between two nodes in the graph if the distance between the corresponding roads\nis less than a threshold.\n'

def compute_adjacency_matrix(route_distances: np.ndarray, sigma2: float, epsilon: float):
    if False:
        print('Hello World!')
    'Computes the adjacency matrix from distances matrix.\n\n    It uses the formula in https://github.com/VeritasYin/STGCN_IJCAI-18#data-preprocessing to\n    compute an adjacency matrix from the distance matrix.\n    The implementation follows that paper.\n\n    Args:\n        route_distances: np.ndarray of shape `(num_routes, num_routes)`. Entry `i,j` of this array is the\n            distance between roads `i,j`.\n        sigma2: Determines the width of the Gaussian kernel applied to the square distances matrix.\n        epsilon: A threshold specifying if there is an edge between two nodes. Specifically, `A[i,j]=1`\n            if `np.exp(-w2[i,j] / sigma2) >= epsilon` and `A[i,j]=0` otherwise, where `A` is the adjacency\n            matrix and `w2=route_distances * route_distances`\n\n    Returns:\n        A boolean graph adjacency matrix.\n    '
    num_routes = route_distances.shape[0]
    route_distances = route_distances / 10000.0
    (w2, w_mask) = (route_distances * route_distances, np.ones([num_routes, num_routes]) - np.identity(num_routes))
    return (np.exp(-w2 / sigma2) >= epsilon) * w_mask
'\nThe function `compute_adjacency_matrix()` returns a boolean adjacency matrix\nwhere 1 means there is an edge between two nodes. We use the following class\nto store the information about the graph.\n'

class GraphInfo:

    def __init__(self, edges: typing.Tuple[list, list], num_nodes: int):
        if False:
            i = 10
            return i + 15
        self.edges = edges
        self.num_nodes = num_nodes
sigma2 = 0.1
epsilon = 0.5
adjacency_matrix = compute_adjacency_matrix(route_distances, sigma2, epsilon)
(node_indices, neighbor_indices) = np.where(adjacency_matrix == 1)
graph = GraphInfo(edges=(node_indices.tolist(), neighbor_indices.tolist()), num_nodes=adjacency_matrix.shape[0])
print(f'number of nodes: {graph.num_nodes}, number of edges: {len(graph.edges[0])}')
'\n## Network architecture\n\nOur model for forecasting over the graph consists of a graph convolution\nlayer and a LSTM layer.\n'
"\n### Graph convolution layer\n\nOur implementation of the graph convolution layer resembles the implementation\nin [this Keras example](https://keras.io/examples/graph/gnn_citations/). Note that\nin that example input to the layer is a 2D tensor of shape `(num_nodes,in_feat)`\nbut in our example the input to the layer is a 4D tensor of shape\n`(num_nodes, batch_size, input_seq_length, in_feat)`. The graph convolution layer\nperforms the following steps:\n\n- The nodes' representations are computed in `self.compute_nodes_representation()`\nby multiplying the input features by `self.weight`\n- The aggregated neighbors' messages are computed in `self.compute_aggregated_messages()`\nby first aggregating the neighbors' representations and then multiplying the results by\n`self.weight`\n- The final output of the layer is computed in `self.update()` by combining the nodes\nrepresentations and the neighbors' aggregated messages\n"

class GraphConv(layers.Layer):

    def __init__(self, in_feat, out_feat, graph_info: GraphInfo, aggregation_type='mean', combination_type='concat', activation: typing.Optional[str]=None, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.graph_info = graph_info
        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.weight = tf.Variable(initial_value=keras.initializers.GlorotUniform()(shape=(in_feat, out_feat), dtype='float32'), trainable=True)
        self.activation = layers.Activation(activation)

    def aggregate(self, neighbour_representations: tf.Tensor):
        if False:
            print('Hello World!')
        aggregation_func = {'sum': tf.math.unsorted_segment_sum, 'mean': tf.math.unsorted_segment_mean, 'max': tf.math.unsorted_segment_max}.get(self.aggregation_type)
        if aggregation_func:
            return aggregation_func(neighbour_representations, self.graph_info.edges[0], num_segments=self.graph_info.num_nodes)
        raise ValueError(f'Invalid aggregation type: {self.aggregation_type}')

    def compute_nodes_representation(self, features: tf.Tensor):
        if False:
            print('Hello World!')
        "Computes each node's representation.\n\n        The nodes' representations are obtained by multiplying the features tensor with\n        `self.weight`. Note that\n        `self.weight` has shape `(in_feat, out_feat)`.\n\n        Args:\n            features: Tensor of shape `(num_nodes, batch_size, input_seq_len, in_feat)`\n\n        Returns:\n            A tensor of shape `(num_nodes, batch_size, input_seq_len, out_feat)`\n        "
        return tf.matmul(features, self.weight)

    def compute_aggregated_messages(self, features: tf.Tensor):
        if False:
            while True:
                i = 10
        neighbour_representations = tf.gather(features, self.graph_info.edges[1])
        aggregated_messages = self.aggregate(neighbour_representations)
        return tf.matmul(aggregated_messages, self.weight)

    def update(self, nodes_representation: tf.Tensor, aggregated_messages: tf.Tensor):
        if False:
            print('Hello World!')
        if self.combination_type == 'concat':
            h = tf.concat([nodes_representation, aggregated_messages], axis=-1)
        elif self.combination_type == 'add':
            h = nodes_representation + aggregated_messages
        else:
            raise ValueError(f'Invalid combination type: {self.combination_type}.')
        return self.activation(h)

    def call(self, features: tf.Tensor):
        if False:
            i = 10
            return i + 15
        'Forward pass.\n\n        Args:\n            features: tensor of shape `(num_nodes, batch_size, input_seq_len, in_feat)`\n\n        Returns:\n            A tensor of shape `(num_nodes, batch_size, input_seq_len, out_feat)`\n        '
        nodes_representation = self.compute_nodes_representation(features)
        aggregated_messages = self.compute_aggregated_messages(features)
        return self.update(nodes_representation, aggregated_messages)
"\n### LSTM plus graph convolution\n\nBy applying the graph convolution layer to the input tensor, we get another tensor\ncontaining the nodes' representations over time (another 4D tensor). For each time\nstep, a node's representation is informed by the information from its neighbors.\n\nTo make good forecasts, however, we need not only information from the neighbors\nbut also we need to process the information over time. To this end, we can pass each\nnode's tensor through a recurrent layer. The `LSTMGC` layer below, first applies\na graph convolution layer to the inputs and then passes the results through a\n`LSTM` layer.\n"

class LSTMGC(layers.Layer):
    """Layer comprising a convolution layer followed by LSTM and dense layers."""

    def __init__(self, in_feat, out_feat, lstm_units: int, input_seq_len: int, output_seq_len: int, graph_info: GraphInfo, graph_conv_params: typing.Optional[dict]=None, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        if graph_conv_params is None:
            graph_conv_params = {'aggregation_type': 'mean', 'combination_type': 'concat', 'activation': None}
        self.graph_conv = GraphConv(in_feat, out_feat, graph_info, **graph_conv_params)
        self.lstm = layers.LSTM(lstm_units, activation='relu')
        self.dense = layers.Dense(output_seq_len)
        (self.input_seq_len, self.output_seq_len) = (input_seq_len, output_seq_len)

    def call(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        'Forward pass.\n\n        Args:\n            inputs: tf.Tensor of shape `(batch_size, input_seq_len, num_nodes, in_feat)`\n\n        Returns:\n            A tensor of shape `(batch_size, output_seq_len, num_nodes)`.\n        '
        inputs = tf.transpose(inputs, [2, 0, 1, 3])
        gcn_out = self.graph_conv(inputs)
        shape = tf.shape(gcn_out)
        (num_nodes, batch_size, input_seq_len, out_feat) = (shape[0], shape[1], shape[2], shape[3])
        gcn_out = tf.reshape(gcn_out, (batch_size * num_nodes, input_seq_len, out_feat))
        lstm_out = self.lstm(gcn_out)
        dense_output = self.dense(lstm_out)
        output = tf.reshape(dense_output, (num_nodes, batch_size, self.output_seq_len))
        return tf.transpose(output, [1, 2, 0])
'\n## Model training\n'
in_feat = 1
batch_size = 64
epochs = 20
input_sequence_length = 12
forecast_horizon = 3
multi_horizon = False
out_feat = 10
lstm_units = 64
graph_conv_params = {'aggregation_type': 'mean', 'combination_type': 'concat', 'activation': None}
st_gcn = LSTMGC(in_feat, out_feat, lstm_units, input_sequence_length, forecast_horizon, graph, graph_conv_params)
inputs = layers.Input((input_sequence_length, graph.num_nodes, in_feat))
outputs = st_gcn(inputs)
model = keras.models.Model(inputs, outputs)
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.0002), loss=keras.losses.MeanSquaredError())
model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=[keras.callbacks.EarlyStopping(patience=10)])
'\n## Making forecasts on test set\n\nNow we can use the trained model to make forecasts for the test set. Below, we\ncompute the MAE of the model and compare it to the MAE of naive forecasts.\nThe naive forecasts are the last value of the speed for each node.\n'
(x_test, y) = next(test_dataset.as_numpy_iterator())
y_pred = model.predict(x_test)
plt.figure(figsize=(18, 6))
plt.plot(y[:, 0, 0])
plt.plot(y_pred[:, 0, 0])
plt.legend(['actual', 'forecast'])
(naive_mse, model_mse) = (np.square(x_test[:, -1, :, 0] - y[:, 0, :]).mean(), np.square(y_pred[:, 0, :] - y[:, 0, :]).mean())
print(f'naive MAE: {naive_mse}, model MAE: {model_mse}')
"\nOf course, the goal here is to demonstrate the method,\nnot to achieve the best performance. To improve the\nmodel's accuracy, all model hyperparameters should be tuned carefully. In addition,\nseveral of the `LSTMGC` blocks can be stacked to increase the representation power\nof the model.\n"