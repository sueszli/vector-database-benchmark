"""
visit https://mofanpy.com/tutorials/ for more!

Build two networks.
1. Without batch normalization
2. With batch normalization

Run tests on these two networks.
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
ACTIVATION = tf.nn.relu
N_LAYERS = 7
N_HIDDEN_UNITS = 30

def fix_seed(seed=1):
    if False:
        for i in range(10):
            print('nop')
    np.random.seed(seed)
    tf.set_random_seed(seed)

def plot_his(inputs, inputs_norm):
    if False:
        print('Hello World!')
    for (j, all_inputs) in enumerate([inputs, inputs_norm]):
        for (i, input) in enumerate(all_inputs):
            plt.subplot(2, len(all_inputs), j * len(all_inputs) + (i + 1))
            plt.cla()
            if i == 0:
                the_range = (-7, 10)
            else:
                the_range = (-1, 1)
            plt.hist(input.ravel(), bins=15, range=the_range, color='#FF5733')
            plt.yticks(())
            if j == 1:
                plt.xticks(the_range)
            else:
                plt.xticks(())
            ax = plt.gca()
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
        plt.title('%s normalizing' % ('Without' if j == 0 else 'With'))
    plt.draw()
    plt.pause(0.01)

def built_net(xs, ys, norm):
    if False:
        for i in range(10):
            print('nop')

    def add_layer(inputs, in_size, out_size, activation_function=None, norm=False):
        if False:
            return 10
        Weights = tf.Variable(tf.random_normal([in_size, out_size], mean=0.0, stddev=1.0))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if norm:
            (fc_mean, fc_var) = tf.nn.moments(Wx_plus_b, axes=[0])
            scale = tf.Variable(tf.ones([out_size]))
            shift = tf.Variable(tf.zeros([out_size]))
            epsilon = 0.001
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                if False:
                    while True:
                        i = 10
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return (tf.identity(fc_mean), tf.identity(fc_var))
            (mean, var) = mean_var_with_update()
            Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, mean, var, shift, scale, epsilon)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs
    fix_seed(1)
    if norm:
        (fc_mean, fc_var) = tf.nn.moments(xs, axes=[0])
        scale = tf.Variable(tf.ones([1]))
        shift = tf.Variable(tf.zeros([1]))
        epsilon = 0.001
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            if False:
                while True:
                    i = 10
            ema_apply_op = ema.apply([fc_mean, fc_var])
            with tf.control_dependencies([ema_apply_op]):
                return (tf.identity(fc_mean), tf.identity(fc_var))
        (mean, var) = mean_var_with_update()
        xs = tf.nn.batch_normalization(xs, mean, var, shift, scale, epsilon)
    layers_inputs = [xs]
    for l_n in range(N_LAYERS):
        layer_input = layers_inputs[l_n]
        in_size = layers_inputs[l_n].get_shape()[1].value
        output = add_layer(layer_input, in_size, N_HIDDEN_UNITS, ACTIVATION, norm)
        layers_inputs.append(output)
    prediction = add_layer(layers_inputs[-1], 30, 1, activation_function=None)
    cost = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
    return [train_op, cost, layers_inputs]
fix_seed(1)
x_data = np.linspace(-7, 10, 2500)[:, np.newaxis]
np.random.shuffle(x_data)
noise = np.random.normal(0, 8, x_data.shape)
y_data = np.square(x_data) - 5 + noise
plt.scatter(x_data, y_data)
plt.show()
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
(train_op, cost, layers_inputs) = built_net(xs, ys, norm=False)
(train_op_norm, cost_norm, layers_inputs_norm) = built_net(xs, ys, norm=True)
sess = tf.Session()
if int(tf.__version__.split('.')[1]) < 12 and int(tf.__version__.split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)
cost_his = []
cost_his_norm = []
record_step = 5
plt.ion()
plt.figure(figsize=(7, 3))
for i in range(250):
    if i % 50 == 0:
        (all_inputs, all_inputs_norm) = sess.run([layers_inputs, layers_inputs_norm], feed_dict={xs: x_data, ys: y_data})
        plot_his(all_inputs, all_inputs_norm)
    sess.run([train_op, train_op_norm], feed_dict={xs: x_data[i * 10:i * 10 + 10], ys: y_data[i * 10:i * 10 + 10]})
    if i % record_step == 0:
        cost_his.append(sess.run(cost, feed_dict={xs: x_data, ys: y_data}))
        cost_his_norm.append(sess.run(cost_norm, feed_dict={xs: x_data, ys: y_data}))
plt.ioff()
plt.figure()
plt.plot(np.arange(len(cost_his)) * record_step, np.array(cost_his), label='no BN')
plt.plot(np.arange(len(cost_his)) * record_step, np.array(cost_his_norm), label='BN')
plt.legend()
plt.show()