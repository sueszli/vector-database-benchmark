from __future__ import print_function
import tensorflow as tf

class TrainConfig:
    batch_size = 20
    time_steps = 20
    input_size = 10
    output_size = 2
    cell_size = 11
    learning_rate = 0.01

class TestConfig(TrainConfig):
    time_steps = 1

class RNN(object):

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        self._batch_size = config.batch_size
        self._time_steps = config.time_steps
        self._input_size = config.input_size
        self._output_size = config.output_size
        self._cell_size = config.cell_size
        self._lr = config.learning_rate
        self._built_RNN()

    def _built_RNN(self):
        if False:
            while True:
                i = 10
        with tf.variable_scope('inputs'):
            self._xs = tf.placeholder(tf.float32, [self._batch_size, self._time_steps, self._input_size], name='xs')
            self._ys = tf.placeholder(tf.float32, [self._batch_size, self._time_steps, self._output_size], name='ys')
        with tf.name_scope('RNN'):
            with tf.variable_scope('input_layer'):
                l_in_x = tf.reshape(self._xs, [-1, self._input_size], name='2_2D')
                Wi = self._weight_variable([self._input_size, self._cell_size])
                print(Wi.name)
                bi = self._bias_variable([self._cell_size])
                with tf.name_scope('Wx_plus_b'):
                    l_in_y = tf.matmul(l_in_x, Wi) + bi
                l_in_y = tf.reshape(l_in_y, [-1, self._time_steps, self._cell_size], name='2_3D')
            with tf.variable_scope('cell'):
                cell = tf.contrib.rnn.BasicLSTMCell(self._cell_size)
                with tf.name_scope('initial_state'):
                    self._cell_initial_state = cell.zero_state(self._batch_size, dtype=tf.float32)
                self.cell_outputs = []
                cell_state = self._cell_initial_state
                for t in range(self._time_steps):
                    if t > 0:
                        tf.get_variable_scope().reuse_variables()
                    (cell_output, cell_state) = cell(l_in_y[:, t, :], cell_state)
                    self.cell_outputs.append(cell_output)
                self._cell_final_state = cell_state
            with tf.variable_scope('output_layer'):
                cell_outputs_reshaped = tf.reshape(tf.concat(self.cell_outputs, 1), [-1, self._cell_size])
                Wo = self._weight_variable((self._cell_size, self._output_size))
                bo = self._bias_variable((self._output_size,))
                product = tf.matmul(cell_outputs_reshaped, Wo) + bo
                self._pred = tf.nn.relu(product)
        with tf.name_scope('cost'):
            _pred = tf.reshape(self._pred, [self._batch_size, self._time_steps, self._output_size])
            mse = self.ms_error(_pred, self._ys)
            mse_ave_across_batch = tf.reduce_mean(mse, 0)
            mse_sum_across_time = tf.reduce_sum(mse_ave_across_batch, 0)
            self._cost = mse_sum_across_time
            self._cost_ave_time = self._cost / self._time_steps
        with tf.variable_scope('trian'):
            self._lr = tf.convert_to_tensor(self._lr)
            self.train_op = tf.train.AdamOptimizer(self._lr).minimize(self._cost)

    @staticmethod
    def ms_error(y_target, y_pre):
        if False:
            while True:
                i = 10
        return tf.square(tf.subtract(y_target, y_pre))

    @staticmethod
    def _weight_variable(shape, name='weights'):
        if False:
            print('Hello World!')
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.5)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    @staticmethod
    def _bias_variable(shape, name='biases'):
        if False:
            while True:
                i = 10
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)
if __name__ == '__main__':
    train_config = TrainConfig()
    test_config = TestConfig()
    with tf.variable_scope('train_rnn'):
        train_rnn1 = RNN(train_config)
    with tf.variable_scope('test_rnn'):
        test_rnn1 = RNN(test_config)
    with tf.variable_scope('rnn') as scope:
        sess = tf.Session()
        train_rnn2 = RNN(train_config)
        scope.reuse_variables()
        test_rnn2 = RNN(test_config)
        if int(tf.__version__.split('.')[1]) < 12 and int(tf.__version__.split('.')[0]) < 1:
            init = tf.initialize_all_variables()
        else:
            init = tf.global_variables_initializer()
        sess.run(init)