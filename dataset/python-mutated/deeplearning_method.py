__author__ = 'liudong'
__date__ = '2018/5/29 下午7:40'
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def load_data(train_path, test_path):
    if False:
        i = 10
        return i + 15
    "\n    加载数据的方法\n    :param train_path: path for the train set file\n    :param test_path: path for the test set file\n    :return: a 'pandas' array for each set\n    "
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    print('number of training examples = ' + str(train_data.shape[0]))
    print('number of test examples = ' + str(test_data.shape[0]))
    print('train shape: ' + str(train_data.shape))
    print('test shape: ' + str(test_data.shape))
    return (train_data, test_data)

def pre_process_data(df):
    if False:
        i = 10
        return i + 15
    '\n    Perform a number of pre process functions on the data set\n    :param df: pandas data frame\n    :return: processed data frame\n    '
    df = pd.get_dummies(df)
    return df

def mini_batches(train_set, train_labels, mini_batch_size):
    if False:
        while True:
            i = 10
    '\n    Generate mini batches from the data set (data and labels)\n    :param train_set: data set with the examples\n    :param train_labels: data set with the labels\n    :param mini_batch_size: mini batch size\n    :return: mini batches\n    '
    set_size = train_set.shape[0]
    batches = []
    num_complete_minibatches = set_size // mini_batch_size
    for k in range(0, num_complete_minibatches):
        mini_batch_x = train_set[k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_y = train_labels[k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch = (mini_batch_x, mini_batch_y)
        batches.append(mini_batch)
    if set_size % mini_batch_size != 0:
        mini_batch_x = train_set[set_size - set_size % mini_batch_size:]
        mini_batch_y = train_labels[set_size - set_size % mini_batch_size:]
        mini_batch = (mini_batch_x, mini_batch_y)
        batches.append(mini_batch)
    return batches

def create_placeholders(input_size, output_size):
    if False:
        print('Hello World!')
    '\n    Creates the placeholders for the tensorflow session.\n    :param input_size: scalar, input size\n    :param output_size: scalar, output size\n    :return: X  placeholder for the data input, of shape [None, input_size] and dtype "float"\n    :return: Y placeholder for the input labels, of shape [None, output_size] and dtype "float"\n    '
    x = tf.placeholder(shape=(None, input_size), dtype=tf.float32, name='X')
    y = tf.placeholder(shape=(None, output_size), dtype=tf.float32, name='Y')
    return (x, y)

def forward_propagation(x, parameters, keep_prob=1.0, hidden_activation='relu'):
    if False:
        while True:
            i = 10
    '\n    Implement forward propagation with dropout for the [LINEAR->RELU]*(L-1)->LINEAR-> computation\n    :param x: data, pandas array of shape (input size, number of examples)\n    :param parameters: output of initialize_parameters()\n    :param keep_prob: probability to keep each node of the layer\n    :param hidden_activation: activation function of the hidden layers\n    :return: last LINEAR value\n    '
    a_dropout = x
    n_layers = len(parameters) // 2
    for l in range(1, n_layers):
        a_prev = a_dropout
        a_dropout = linear_activation_forward(a_prev, parameters['w%s' % l], parameters['b%s' % l], hidden_activation)
        if keep_prob < 1.0:
            a_dropout = tf.nn.dropout(a_dropout, keep_prob)
    al = tf.matmul(a_dropout, parameters['w%s' % n_layers]) + parameters['b%s' % n_layers]
    return al

def linear_activation_forward(a_prev, w, b, activation):
    if False:
        while True:
            i = 10
    '\n    Implement the forward propagation for the LINEAR->ACTIVATION layer\n    :param a_prev: activations from previous layer (or input data): (size of previous layer, number of examples)\n    :param w: weights matrix: numpy array of shape (size of current layer, size of previous layer)\n    :param b: bias vector, numpy array of shape (size of the current layer, 1)\n    :param activation: the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"\n    :return: the output of the activation function, also called the post-activation value\n    '
    a = None
    if activation == 'sigmoid':
        z = tf.matmul(a_prev, w) + b
        a = tf.nn.sigmoid(z)
    elif activation == 'relu':
        z = tf.matmul(a_prev, w) + b
        a = tf.nn.relu(z)
    elif activation == 'leaky relu':
        z = tf.matmul(a_prev, w) + b
        a = tf.nn.leaky_relu(z)
    return a

def initialize_parameters(layer_dims):
    if False:
        while True:
            i = 10
    '\n    :param layer_dims: python array (list) containing the dimensions of each layer in our network\n    :return: python dictionary containing your parameters "w1", "b1", ..., "wn", "bn":\n                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])\n                    bl -- bias vector of shape (layer_dims[l], 1)\n    '
    parameters = {}
    n_layers = len(layer_dims)
    for l in range(1, n_layers):
        parameters['w' + str(l)] = tf.get_variable('w' + str(l), [layer_dims[l - 1], layer_dims[l]], initializer=tf.contrib.layers.xavier_initializer())
        parameters['b' + str(l)] = tf.get_variable('b' + str(l), [layer_dims[l]], initializer=tf.zeros_initializer())
    return parameters

def compute_cost(z3, y):
    if False:
        print('Hello World!')
    '\n    :param z3: output of forward propagation (output of the last LINEAR unit)\n    :param y: "true" labels vector placeholder, same shape as Z3\n    :return: Tensor of the cost function (RMSE as it is a regression)\n    '
    cost = tf.sqrt(tf.reduce_mean(tf.square(y - z3)))
    return cost

def predict(data, parameters):
    if False:
        return 10
    '\n    make a prediction based on a data set and parameters\n    :param data: based data set\n    :param parameters: based parameters\n    :return: array of predictions\n    '
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        dataset = tf.cast(tf.constant(data), tf.float32)
        fw_prop_result = forward_propagation(dataset, parameters)
        prediction = fw_prop_result.eval()
    return prediction

def rmse(predictions, labels):
    if False:
        print('Hello World!')
    '\n    calculate cost between two data sets\n    :param predictions: data set of predictions\n    :param labels: data set of labels (real values)\n    :return: percentage of correct predictions\n    '
    prediction_size = predictions.shape[0]
    prediction_cost = np.sqrt(np.sum(np.square(labels - predictions)) / prediction_size)
    return prediction_cost

def rmsle(predictions, labels):
    if False:
        while True:
            i = 10
    '\n    calculate cost between two data sets\n    :param predictions: data set of predictions\n    :param labels: data set of labels (real values)\n    :return: percentage of correct predictions\n    '
    prediction_size = predictions.shape[0]
    prediction_cost = np.sqrt(np.sum(np.square(np.log(predictions + 1) - np.log(labels + 1))) / prediction_size)
    return prediction_cost

def l2_regularizer(cost, l2_beta, parameters, n_layers):
    if False:
        i = 10
        return i + 15
    '\n    Function to apply l2 regularization to the model\n    :param cost: usual cost of the model\n    :param l2_beta: beta value used for the normalization\n    :param parameters: parameters from the model (used to get weights values)\n    :param n_layers: number of layers of the model\n    :return: cost updated\n    '
    regularizer = 0
    for i in range(1, n_layers):
        regularizer += tf.nn.l2_loss(parameters['w%s' % i])
    cost = tf.reduce_mean(cost + l2_beta * regularizer)
    return cost

def build_submission_name(layers_dims, num_epochs, lr_decay, learning_rate, l2_beta, keep_prob, minibatch_size, num_examples):
    if False:
        while True:
            i = 10
    '\n    builds a string (submission file name), based on the model parameters\n    :param layers_dims: model layers dimensions\n    :param num_epochs: model number of epochs\n    :param lr_decay: model learning rate decay\n    :param learning_rate: model learning rate\n    :param l2_beta: beta used on l2 normalization\n    :param keep_prob: keep probability used on dropout normalization\n    :param minibatch_size: model mini batch size (0 to do not use mini batches)\n    :param num_examples: number of model examples (training data)\n    :return: built string\n    '
    submission_name = 'ly{}-epoch{}.csv'.format(layers_dims, num_epochs)
    if lr_decay != 0:
        submission_name = 'lrdc{}-'.format(lr_decay) + submission_name
    else:
        submission_name = 'lr{}-'.format(learning_rate) + submission_name
    if l2_beta > 0:
        submission_name = 'l2{}-'.format(l2_beta) + submission_name
    if keep_prob < 1:
        submission_name = 'dk{}-'.format(keep_prob) + submission_name
    if minibatch_size != num_examples:
        submission_name = 'mb{}-'.format(minibatch_size) + submission_name
    return submission_name

def plot_model_cost(train_costs, validation_costs, submission_name):
    if False:
        print('Hello World!')
    '\n    :param train_costs: array with the costs from the model training\n    :param validation_costs: array with the costs from the model validation\n    :param submission_name: name of the submission (used for the plot title)\n    :return:\n    '
    plt.plot(np.squeeze(train_costs), label='Train cost')
    plt.plot(np.squeeze(validation_costs), label='Validation cost')
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title('Model: ' + submission_name)
    plt.legend()
    plt.show()
    plt.close()

def model(train_set, train_labels, validation_set, validation_labels, layers_dims, learning_rate=0.01, num_epochs=1001, print_cost=True, plot_cost=True, l2_beta=0.0, keep_prob=1.0, hidden_activation='relu', return_best=False, minibatch_size=0, lr_decay=0):
    if False:
        for i in range(10):
            print('nop')
    '\n    :param train_set: training set\n    :param train_labels: training labels\n    :param validation_set: validation set\n    :param validation_labels: validation labels\n    :param layers_dims: array with the layer for the model\n    :param learning_rate: learning rate of the optimization\n    :param num_epochs: number of epochs of the optimization loop\n    :param print_cost: True to print the cost every 500 epochs\n    :param plot_cost: True to plot the train and validation cost\n    :param l2_beta: beta parameter for the l2 regularization\n    :param keep_prob: probability to keep each node of each hidden layer (dropout)\n    :param hidden_activation: activation function to be used on the hidden layers\n    :param return_best: True to return the highest params from all epochs\n    :param minibatch_size: size of th mini batch\n    :param lr_decay: if != 0, sets de learning rate decay on each epoch\n    :return parameters: parameters learnt by the model. They can then be used to predict.\n    :return submission_name: name for the trained model\n    '
    ops.reset_default_graph()
    input_size = layers_dims[0]
    output_size = layers_dims[-1]
    num_examples = train_set.shape[0]
    n_layers = len(layers_dims)
    train_costs = []
    validation_costs = []
    best_iteration = [float('inf'), 0]
    best_params = None
    if minibatch_size == 0 or minibatch_size > num_examples:
        minibatch_size = num_examples
    num_minibatches = num_examples // minibatch_size
    if num_minibatches == 0:
        num_minibatches = 1
    submission_name = build_submission_name(layers_dims, num_epochs, lr_decay, learning_rate, l2_beta, keep_prob, minibatch_size, num_examples)
    (x, y) = create_placeholders(input_size, output_size)
    tf_valid_dataset = tf.cast(tf.constant(validation_set), tf.float32)
    parameters = initialize_parameters(layers_dims)
    fw_output_train = forward_propagation(x, parameters, keep_prob, hidden_activation)
    train_cost = compute_cost(fw_output_train, y)
    fw_output_valid = forward_propagation(tf_valid_dataset, parameters, keep_prob, hidden_activation)
    validation_cost = compute_cost(fw_output_valid, validation_labels)
    if l2_beta > 0:
        train_cost = l2_regularizer(train_cost, l2_beta, parameters, n_layers)
        validation_cost = l2_regularizer(validation_cost, l2_beta, parameters, n_layers)
    if lr_decay != 0:
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.inverse_time_decay(learning_rate, global_step=global_step, decay_rate=lr_decay, decay_steps=1)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(train_cost, global_step=global_step)
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(train_cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            train_epoch_cost = 0.0
            validation_epoch_cost = 0.0
            minibatches = mini_batches(train_set, train_labels, minibatch_size)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                feed_dict = {x: minibatch_X, y: minibatch_Y}
                (_, minibatch_train_cost, minibatch_validation_cost) = sess.run([optimizer, train_cost, validation_cost], feed_dict=feed_dict)
                train_epoch_cost += minibatch_train_cost / num_minibatches
                validation_epoch_cost += minibatch_validation_cost / num_minibatches
            if print_cost is True and epoch % 500 == 0:
                print('Train cost after epoch %i: %f' % (epoch, train_epoch_cost))
                print('Validation cost after epoch %i: %f' % (epoch, validation_epoch_cost))
            if plot_cost is True and epoch % 10 == 0:
                train_costs.append(train_epoch_cost)
                validation_costs.append(validation_epoch_cost)
            if return_best is True and validation_epoch_cost < best_iteration[0]:
                best_iteration[0] = validation_epoch_cost
                best_iteration[1] = epoch
                best_params = sess.run(parameters)
        if return_best is True:
            parameters = best_params
        else:
            parameters = sess.run(parameters)
        print('Parameters have been trained, getting metrics...')
        train_rmse = rmse(predict(train_set, parameters), train_labels)
        validation_rmse = rmse(predict(validation_set, parameters), validation_labels)
        train_rmsle = rmsle(predict(train_set, parameters), train_labels)
        validation_rmsle = rmsle(predict(validation_set, parameters), validation_labels)
        print('Train rmse: {:.4f}'.format(train_rmse))
        print('Validation rmse: {:.4f}'.format(validation_rmse))
        print('Train rmsle: {:.4f}'.format(train_rmsle))
        print('Validation rmsle: {:.4f}'.format(validation_rmsle))
        submission_name = 'tr_cost-{:.2f}-vd_cost{:.2f}-'.format(train_rmse, validation_rmse) + submission_name
        if return_best is True:
            print('Lowest rmse: {:.2f} at epoch {}'.format(best_iteration[0], best_iteration[1]))
        if plot_cost is True:
            plot_model_cost(train_costs, validation_costs, submission_name)
        return (parameters, submission_name)
TRAIN_PATH = '/Users/liudong/Desktop/house_price/train.csv'
TEST_PATH = '/Users/liudong/Desktop/house_price/test.csv'
(train, test) = load_data(TRAIN_PATH, TEST_PATH)
train_raw_labels = train['SalePrice'].to_frame().as_matrix()
train_pre = pre_process_data(train)
test_pre = pre_process_data(test)
train_pre = train_pre.drop(['Id', 'SalePrice'], axis=1)
test_pre = test_pre.drop(['Id'], axis=1)
(train_pre, test_pre) = train_pre.align(test_pre, join='outer', axis=1)
train_pre.replace(to_replace=np.nan, value=0, inplace=True)
test_pre.replace(to_replace=np.nan, value=0, inplace=True)
train_pre = train_pre.as_matrix().astype(np.float)
test_pre = test_pre.as_matrix().astype(np.float)
standard_scaler = preprocessing.StandardScaler()
train_pre = standard_scaler.fit_transform(train_pre)
test_pre = standard_scaler.fit_transform(test_pre)
(X_train, X_valid, Y_train, Y_valid) = train_test_split(train_pre, train_raw_labels, test_size=0.3, random_state=1)
input_size = train_pre.shape[1]
output_size = 1
num_epochs = 10000
learning_rate = 0.01
layers_dims = [input_size, 500, 500, output_size]
(parameters, submission_name) = model(X_train, Y_train, X_valid, Y_valid, layers_dims, num_epochs=num_epochs, learning_rate=learning_rate, print_cost=True, plot_cost=True, l2_beta=10, keep_prob=0.7, minibatch_size=0, return_best=True)
print(submission_name)
prediction = list(map(lambda val: float(val), predict(test_pre, parameters)))
result = pd.DataFrame()
result['Id'] = test.Id.values
result['SalePrice'] = prediction
result.to_csv('/Users/liudong/Desktop/house_price/result1.csv', index=False)
print('##########结束训练##########')