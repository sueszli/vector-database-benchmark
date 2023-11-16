"""Tests for lstm_object_detection.lstm.lstm_cells."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from lstm_object_detection.lstm import lstm_cells

class BottleneckConvLstmCellsTest(tf.test.TestCase):

    def test_run_lstm_cell(self):
        if False:
            while True:
                i = 10
        filter_size = [3, 3]
        output_size = [10, 10]
        num_units = 15
        state_name = 'lstm_state'
        batch_size = 4
        dtype = tf.float32
        learned_state = False
        inputs = tf.zeros([4, 10, 10, 3], dtype=tf.float32)
        cell = lstm_cells.BottleneckConvLSTMCell(filter_size=filter_size, output_size=output_size, num_units=num_units)
        init_state = cell.init_state(state_name, batch_size, dtype, learned_state)
        (output, state_tuple) = cell(inputs, init_state)
        self.assertAllEqual([4, 10, 10, 15], output.shape.as_list())
        self.assertAllEqual([4, 10, 10, 15], state_tuple[0].shape.as_list())
        self.assertAllEqual([4, 10, 10, 15], state_tuple[1].shape.as_list())

    def test_run_lstm_cell_with_flattened_state(self):
        if False:
            while True:
                i = 10
        filter_size = [3, 3]
        output_dim = 10
        output_size = [output_dim] * 2
        num_units = 15
        state_name = 'lstm_state'
        batch_size = 4
        dtype = tf.float32
        learned_state = False
        inputs = tf.zeros([batch_size, output_dim, output_dim, 3], dtype=tf.float32)
        cell = lstm_cells.BottleneckConvLSTMCell(filter_size=filter_size, output_size=output_size, num_units=num_units, flatten_state=True)
        init_state = cell.init_state(state_name, batch_size, dtype, learned_state)
        (output, state_tuple) = cell(inputs, init_state)
        self.assertAllEqual([4, 10, 10, 15], output.shape.as_list())
        self.assertAllEqual([4, 1500], state_tuple[0].shape.as_list())
        self.assertAllEqual([4, 1500], state_tuple[1].shape.as_list())

    def test_run_lstm_cell_with_output_bottleneck(self):
        if False:
            for i in range(10):
                print('nop')
        filter_size = [3, 3]
        output_dim = 10
        output_size = [output_dim] * 2
        num_units = 15
        state_name = 'lstm_state'
        batch_size = 4
        dtype = tf.float32
        learned_state = False
        inputs = tf.zeros([batch_size, output_dim, output_dim, 3], dtype=tf.float32)
        cell = lstm_cells.BottleneckConvLSTMCell(filter_size=filter_size, output_size=output_size, num_units=num_units, output_bottleneck=True)
        init_state = cell.init_state(state_name, batch_size, dtype, learned_state)
        (output, state_tuple) = cell(inputs, init_state)
        self.assertAllEqual([4, 10, 10, 30], output.shape.as_list())
        self.assertAllEqual([4, 10, 10, 15], state_tuple[0].shape.as_list())
        self.assertAllEqual([4, 10, 10, 15], state_tuple[1].shape.as_list())

    def test_get_init_state(self):
        if False:
            print('Hello World!')
        filter_size = [3, 3]
        output_dim = 10
        output_size = [output_dim] * 2
        num_units = 15
        state_name = 'lstm_state'
        batch_size = 4
        dtype = tf.float32
        learned_state = False
        cell = lstm_cells.BottleneckConvLSTMCell(filter_size=filter_size, output_size=output_size, num_units=num_units)
        (init_c, init_h) = cell.init_state(state_name, batch_size, dtype, learned_state)
        self.assertEqual(tf.float32, init_c.dtype)
        self.assertEqual(tf.float32, init_h.dtype)
        with self.test_session() as sess:
            (init_c_res, init_h_res) = sess.run([init_c, init_h])
            self.assertAllClose(np.zeros((4, 10, 10, 15)), init_c_res)
            self.assertAllClose(np.zeros((4, 10, 10, 15)), init_h_res)

    def test_get_init_learned_state(self):
        if False:
            while True:
                i = 10
        filter_size = [3, 3]
        output_size = [10, 10]
        num_units = 15
        state_name = 'lstm_state'
        batch_size = 4
        dtype = tf.float32
        learned_state = True
        cell = lstm_cells.BottleneckConvLSTMCell(filter_size=filter_size, output_size=output_size, num_units=num_units)
        (init_c, init_h) = cell.init_state(state_name, batch_size, dtype, learned_state)
        self.assertEqual(tf.float32, init_c.dtype)
        self.assertEqual(tf.float32, init_h.dtype)
        self.assertAllEqual([4, 10, 10, 15], init_c.shape.as_list())
        self.assertAllEqual([4, 10, 10, 15], init_h.shape.as_list())

    def test_unroll(self):
        if False:
            while True:
                i = 10
        filter_size = [3, 3]
        output_size = [10, 10]
        num_units = 15
        state_name = 'lstm_state'
        batch_size = 4
        dtype = tf.float32
        unroll = 10
        learned_state = False
        inputs = tf.zeros([4, 10, 10, 3], dtype=tf.float32)
        cell = lstm_cells.BottleneckConvLSTMCell(filter_size=filter_size, output_size=output_size, num_units=num_units)
        state = cell.init_state(state_name, batch_size, dtype, learned_state)
        for step in range(unroll):
            (output, state) = cell(inputs, state)
        self.assertAllEqual([4, 10, 10, 15], output.shape.as_list())
        self.assertAllEqual([4, 10, 10, 15], state[0].shape.as_list())
        self.assertAllEqual([4, 10, 10, 15], state[1].shape.as_list())

    def test_prebottleneck(self):
        if False:
            while True:
                i = 10
        filter_size = [3, 3]
        output_size = [10, 10]
        num_units = 15
        state_name = 'lstm_state'
        batch_size = 4
        dtype = tf.float32
        unroll = 10
        learned_state = False
        inputs_large = tf.zeros([4, 10, 10, 5], dtype=tf.float32)
        inputs_small = tf.zeros([4, 10, 10, 3], dtype=tf.float32)
        cell = lstm_cells.BottleneckConvLSTMCell(filter_size=filter_size, output_size=output_size, num_units=num_units, pre_bottleneck=True)
        state = cell.init_state(state_name, batch_size, dtype, learned_state)
        for step in range(unroll):
            if step % 2 == 0:
                inputs = cell.pre_bottleneck(inputs_large, state[1], 0)
            else:
                inputs = cell.pre_bottleneck(inputs_small, state[1], 1)
            (output, state) = cell(inputs, state)
        self.assertAllEqual([4, 10, 10, 15], output.shape.as_list())
        self.assertAllEqual([4, 10, 10, 15], state[0].shape.as_list())
        self.assertAllEqual([4, 10, 10, 15], state[1].shape.as_list())

    def test_flatten_state(self):
        if False:
            while True:
                i = 10
        filter_size = [3, 3]
        output_size = [10, 10]
        num_units = 15
        state_name = 'lstm_state'
        batch_size = 4
        dtype = tf.float32
        unroll = 10
        learned_state = False
        inputs_large = tf.zeros([4, 10, 10, 5], dtype=tf.float32)
        inputs_small = tf.zeros([4, 10, 10, 3], dtype=tf.float32)
        cell = lstm_cells.BottleneckConvLSTMCell(filter_size=filter_size, output_size=output_size, num_units=num_units, pre_bottleneck=True, flatten_state=True)
        state = cell.init_state(state_name, batch_size, dtype, learned_state)
        for step in range(unroll):
            if step % 2 == 0:
                inputs = cell.pre_bottleneck(inputs_large, state[1], 0)
            else:
                inputs = cell.pre_bottleneck(inputs_small, state[1], 1)
            (output, state) = cell(inputs, state)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            (output_result, state_result) = sess.run([output, state])
            self.assertAllEqual((4, 10, 10, 15), output_result.shape)
            self.assertAllEqual((4, 10 * 10 * 15), state_result[0].shape)
            self.assertAllEqual((4, 10 * 10 * 15), state_result[1].shape)

class GroupedConvLstmCellsTest(tf.test.TestCase):

    def test_run_lstm_cell(self):
        if False:
            print('Hello World!')
        filter_size = [3, 3]
        output_size = [10, 10]
        num_units = 16
        state_name = 'lstm_state'
        batch_size = 4
        dtype = tf.float32
        learned_state = False
        inputs = tf.zeros([4, 10, 10, 3], dtype=tf.float32)
        cell = lstm_cells.GroupedConvLSTMCell(filter_size=filter_size, output_size=output_size, num_units=num_units, is_training=True)
        init_state = cell.init_state(state_name, batch_size, dtype, learned_state)
        (output, state_tuple) = cell(inputs, init_state)
        self.assertAllEqual([4, 10, 10, 16], output.shape.as_list())
        self.assertAllEqual([4, 10, 10, 16], state_tuple[0].shape.as_list())
        self.assertAllEqual([4, 10, 10, 16], state_tuple[1].shape.as_list())

    def test_run_lstm_cell_with_output_bottleneck(self):
        if False:
            return 10
        filter_size = [3, 3]
        output_dim = 10
        output_size = [output_dim] * 2
        num_units = 16
        state_name = 'lstm_state'
        batch_size = 4
        dtype = tf.float32
        learned_state = False
        inputs = tf.zeros([batch_size, output_dim, output_dim, 3], dtype=tf.float32)
        cell = lstm_cells.GroupedConvLSTMCell(filter_size=filter_size, output_size=output_size, num_units=num_units, is_training=True, output_bottleneck=True)
        init_state = cell.init_state(state_name, batch_size, dtype, learned_state)
        (output, state_tuple) = cell(inputs, init_state)
        self.assertAllEqual([4, 10, 10, 32], output.shape.as_list())
        self.assertAllEqual([4, 10, 10, 16], state_tuple[0].shape.as_list())
        self.assertAllEqual([4, 10, 10, 16], state_tuple[1].shape.as_list())

    def test_get_init_state(self):
        if False:
            while True:
                i = 10
        filter_size = [3, 3]
        output_dim = 10
        output_size = [output_dim] * 2
        num_units = 16
        state_name = 'lstm_state'
        batch_size = 4
        dtype = tf.float32
        learned_state = False
        cell = lstm_cells.GroupedConvLSTMCell(filter_size=filter_size, output_size=output_size, num_units=num_units, is_training=True)
        (init_c, init_h) = cell.init_state(state_name, batch_size, dtype, learned_state)
        self.assertEqual(tf.float32, init_c.dtype)
        self.assertEqual(tf.float32, init_h.dtype)
        with self.test_session() as sess:
            (init_c_res, init_h_res) = sess.run([init_c, init_h])
            self.assertAllClose(np.zeros((4, 10, 10, 16)), init_c_res)
            self.assertAllClose(np.zeros((4, 10, 10, 16)), init_h_res)

    def test_get_init_learned_state(self):
        if False:
            for i in range(10):
                print('nop')
        filter_size = [3, 3]
        output_size = [10, 10]
        num_units = 16
        state_name = 'lstm_state'
        batch_size = 4
        dtype = tf.float32
        learned_state = True
        cell = lstm_cells.GroupedConvLSTMCell(filter_size=filter_size, output_size=output_size, num_units=num_units, is_training=True)
        (init_c, init_h) = cell.init_state(state_name, batch_size, dtype, learned_state)
        self.assertEqual(tf.float32, init_c.dtype)
        self.assertEqual(tf.float32, init_h.dtype)
        self.assertAllEqual([4, 10, 10, 16], init_c.shape.as_list())
        self.assertAllEqual([4, 10, 10, 16], init_h.shape.as_list())

    def test_unroll(self):
        if False:
            while True:
                i = 10
        filter_size = [3, 3]
        output_size = [10, 10]
        num_units = 16
        state_name = 'lstm_state'
        batch_size = 4
        dtype = tf.float32
        unroll = 10
        learned_state = False
        inputs = tf.zeros([4, 10, 10, 3], dtype=tf.float32)
        cell = lstm_cells.GroupedConvLSTMCell(filter_size=filter_size, output_size=output_size, num_units=num_units, is_training=True)
        state = cell.init_state(state_name, batch_size, dtype, learned_state)
        for step in range(unroll):
            (output, state) = cell(inputs, state)
        self.assertAllEqual([4, 10, 10, 16], output.shape.as_list())
        self.assertAllEqual([4, 10, 10, 16], state[0].shape.as_list())
        self.assertAllEqual([4, 10, 10, 16], state[1].shape.as_list())

    def test_prebottleneck(self):
        if False:
            print('Hello World!')
        filter_size = [3, 3]
        output_size = [10, 10]
        num_units = 16
        state_name = 'lstm_state'
        batch_size = 4
        dtype = tf.float32
        unroll = 10
        learned_state = False
        inputs_large = tf.zeros([4, 10, 10, 5], dtype=tf.float32)
        inputs_small = tf.zeros([4, 10, 10, 3], dtype=tf.float32)
        cell = lstm_cells.GroupedConvLSTMCell(filter_size=filter_size, output_size=output_size, num_units=num_units, is_training=True, pre_bottleneck=True)
        state = cell.init_state(state_name, batch_size, dtype, learned_state)
        for step in range(unroll):
            if step % 2 == 0:
                inputs = cell.pre_bottleneck(inputs_large, state[1], 0)
            else:
                inputs = cell.pre_bottleneck(inputs_small, state[1], 1)
            (output, state) = cell(inputs, state)
        self.assertAllEqual([4, 10, 10, 16], output.shape.as_list())
        self.assertAllEqual([4, 10, 10, 16], state[0].shape.as_list())
        self.assertAllEqual([4, 10, 10, 16], state[1].shape.as_list())

    def test_flatten_state(self):
        if False:
            for i in range(10):
                print('nop')
        filter_size = [3, 3]
        output_size = [10, 10]
        num_units = 16
        state_name = 'lstm_state'
        batch_size = 4
        dtype = tf.float32
        unroll = 10
        learned_state = False
        inputs_large = tf.zeros([4, 10, 10, 5], dtype=tf.float32)
        inputs_small = tf.zeros([4, 10, 10, 3], dtype=tf.float32)
        cell = lstm_cells.GroupedConvLSTMCell(filter_size=filter_size, output_size=output_size, num_units=num_units, is_training=True, pre_bottleneck=True, flatten_state=True)
        state = cell.init_state(state_name, batch_size, dtype, learned_state)
        for step in range(unroll):
            if step % 2 == 0:
                inputs = cell.pre_bottleneck(inputs_large, state[1], 0)
            else:
                inputs = cell.pre_bottleneck(inputs_small, state[1], 1)
            (output, state) = cell(inputs, state)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            (output_result, state_result) = sess.run([output, state])
            self.assertAllEqual((4, 10, 10, 16), output_result.shape)
            self.assertAllEqual((4, 10 * 10 * 16), state_result[0].shape)
            self.assertAllEqual((4, 10 * 10 * 16), state_result[1].shape)
if __name__ == '__main__':
    tf.test.main()