"""Tests for object_detection.utils.variables_helper."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
from object_detection.utils import variables_helper

class FilterVariablesTest(tf.test.TestCase):

    def _create_variables(self):
        if False:
            return 10
        return [tf.Variable(1.0, name='FeatureExtractor/InceptionV3/weights'), tf.Variable(1.0, name='FeatureExtractor/InceptionV3/biases'), tf.Variable(1.0, name='StackProposalGenerator/weights'), tf.Variable(1.0, name='StackProposalGenerator/biases')]

    def test_return_all_variables_when_empty_regex(self):
        if False:
            i = 10
            return i + 15
        variables = self._create_variables()
        out_variables = variables_helper.filter_variables(variables, [''])
        self.assertItemsEqual(out_variables, variables)

    def test_return_variables_which_do_not_match_single_regex(self):
        if False:
            for i in range(10):
                print('nop')
        variables = self._create_variables()
        out_variables = variables_helper.filter_variables(variables, ['FeatureExtractor/.*'])
        self.assertItemsEqual(out_variables, variables[2:])

    def test_return_variables_which_do_not_match_any_regex_in_list(self):
        if False:
            return 10
        variables = self._create_variables()
        out_variables = variables_helper.filter_variables(variables, ['FeatureExtractor.*biases', 'StackProposalGenerator.*biases'])
        self.assertItemsEqual(out_variables, [variables[0], variables[2]])

    def test_return_variables_matching_empty_regex_list(self):
        if False:
            i = 10
            return i + 15
        variables = self._create_variables()
        out_variables = variables_helper.filter_variables(variables, [''], invert=True)
        self.assertItemsEqual(out_variables, [])

    def test_return_variables_matching_some_regex_in_list(self):
        if False:
            i = 10
            return i + 15
        variables = self._create_variables()
        out_variables = variables_helper.filter_variables(variables, ['FeatureExtractor.*biases', 'StackProposalGenerator.*biases'], invert=True)
        self.assertItemsEqual(out_variables, [variables[1], variables[3]])

class MultiplyGradientsMatchingRegexTest(tf.test.TestCase):

    def _create_grads_and_vars(self):
        if False:
            while True:
                i = 10
        return [(tf.constant(1.0), tf.Variable(1.0, name='FeatureExtractor/InceptionV3/weights')), (tf.constant(2.0), tf.Variable(2.0, name='FeatureExtractor/InceptionV3/biases')), (tf.constant(3.0), tf.Variable(3.0, name='StackProposalGenerator/weights')), (tf.constant(4.0), tf.Variable(4.0, name='StackProposalGenerator/biases'))]

    def test_multiply_all_feature_extractor_variables(self):
        if False:
            i = 10
            return i + 15
        grads_and_vars = self._create_grads_and_vars()
        regex_list = ['FeatureExtractor/.*']
        multiplier = 0.0
        grads_and_vars = variables_helper.multiply_gradients_matching_regex(grads_and_vars, regex_list, multiplier)
        exp_output = [(0.0, 1.0), (0.0, 2.0), (3.0, 3.0), (4.0, 4.0)]
        init_op = tf.global_variables_initializer()
        with self.test_session() as sess:
            sess.run(init_op)
            output = sess.run(grads_and_vars)
            self.assertItemsEqual(output, exp_output)

    def test_multiply_all_bias_variables(self):
        if False:
            for i in range(10):
                print('nop')
        grads_and_vars = self._create_grads_and_vars()
        regex_list = ['.*/biases']
        multiplier = 0.0
        grads_and_vars = variables_helper.multiply_gradients_matching_regex(grads_and_vars, regex_list, multiplier)
        exp_output = [(1.0, 1.0), (0.0, 2.0), (3.0, 3.0), (0.0, 4.0)]
        init_op = tf.global_variables_initializer()
        with self.test_session() as sess:
            sess.run(init_op)
            output = sess.run(grads_and_vars)
            self.assertItemsEqual(output, exp_output)

class FreezeGradientsMatchingRegexTest(tf.test.TestCase):

    def _create_grads_and_vars(self):
        if False:
            while True:
                i = 10
        return [(tf.constant(1.0), tf.Variable(1.0, name='FeatureExtractor/InceptionV3/weights')), (tf.constant(2.0), tf.Variable(2.0, name='FeatureExtractor/InceptionV3/biases')), (tf.constant(3.0), tf.Variable(3.0, name='StackProposalGenerator/weights')), (tf.constant(4.0), tf.Variable(4.0, name='StackProposalGenerator/biases'))]

    def test_freeze_all_feature_extractor_variables(self):
        if False:
            i = 10
            return i + 15
        grads_and_vars = self._create_grads_and_vars()
        regex_list = ['FeatureExtractor/.*']
        grads_and_vars = variables_helper.freeze_gradients_matching_regex(grads_and_vars, regex_list)
        exp_output = [(3.0, 3.0), (4.0, 4.0)]
        init_op = tf.global_variables_initializer()
        with self.test_session() as sess:
            sess.run(init_op)
            output = sess.run(grads_and_vars)
            self.assertItemsEqual(output, exp_output)

class GetVariablesAvailableInCheckpointTest(tf.test.TestCase):

    def test_return_all_variables_from_checkpoint(self):
        if False:
            while True:
                i = 10
        with tf.Graph().as_default():
            variables = [tf.Variable(1.0, name='weights'), tf.Variable(1.0, name='biases')]
            checkpoint_path = os.path.join(self.get_temp_dir(), 'model.ckpt')
            init_op = tf.global_variables_initializer()
            saver = tf.train.Saver(variables)
            with self.test_session() as sess:
                sess.run(init_op)
                saver.save(sess, checkpoint_path)
            out_variables = variables_helper.get_variables_available_in_checkpoint(variables, checkpoint_path)
        self.assertItemsEqual(out_variables, variables)

    def test_return_all_variables_from_checkpoint_with_partition(self):
        if False:
            for i in range(10):
                print('nop')
        with tf.Graph().as_default():
            partitioner = tf.fixed_size_partitioner(2)
            variables = [tf.get_variable(name='weights', shape=(2, 2), partitioner=partitioner), tf.Variable([1.0, 2.0], name='biases')]
            checkpoint_path = os.path.join(self.get_temp_dir(), 'model.ckpt')
            init_op = tf.global_variables_initializer()
            saver = tf.train.Saver(variables)
            with self.test_session() as sess:
                sess.run(init_op)
                saver.save(sess, checkpoint_path)
            out_variables = variables_helper.get_variables_available_in_checkpoint(variables, checkpoint_path)
        self.assertItemsEqual(out_variables, variables)

    def test_return_variables_available_in_checkpoint(self):
        if False:
            while True:
                i = 10
        checkpoint_path = os.path.join(self.get_temp_dir(), 'model.ckpt')
        with tf.Graph().as_default():
            weight_variable = tf.Variable(1.0, name='weights')
            global_step = tf.train.get_or_create_global_step()
            graph1_variables = [weight_variable, global_step]
            init_op = tf.global_variables_initializer()
            saver = tf.train.Saver(graph1_variables)
            with self.test_session() as sess:
                sess.run(init_op)
                saver.save(sess, checkpoint_path)
        with tf.Graph().as_default():
            graph2_variables = graph1_variables + [tf.Variable(1.0, name='biases')]
            out_variables = variables_helper.get_variables_available_in_checkpoint(graph2_variables, checkpoint_path, include_global_step=False)
        self.assertItemsEqual(out_variables, [weight_variable])

    def test_return_variables_available_an_checkpoint_with_dict_inputs(self):
        if False:
            i = 10
            return i + 15
        checkpoint_path = os.path.join(self.get_temp_dir(), 'model.ckpt')
        with tf.Graph().as_default():
            graph1_variables = [tf.Variable(1.0, name='ckpt_weights')]
            init_op = tf.global_variables_initializer()
            saver = tf.train.Saver(graph1_variables)
            with self.test_session() as sess:
                sess.run(init_op)
                saver.save(sess, checkpoint_path)
        with tf.Graph().as_default():
            graph2_variables_dict = {'ckpt_weights': tf.Variable(1.0, name='weights'), 'ckpt_biases': tf.Variable(1.0, name='biases')}
            out_variables = variables_helper.get_variables_available_in_checkpoint(graph2_variables_dict, checkpoint_path)
        self.assertTrue(isinstance(out_variables, dict))
        self.assertItemsEqual(list(out_variables.keys()), ['ckpt_weights'])
        self.assertTrue(out_variables['ckpt_weights'].op.name == 'weights')

    def test_return_variables_with_correct_sizes(self):
        if False:
            i = 10
            return i + 15
        checkpoint_path = os.path.join(self.get_temp_dir(), 'model.ckpt')
        with tf.Graph().as_default():
            bias_variable = tf.Variable(3.0, name='biases')
            global_step = tf.train.get_or_create_global_step()
            graph1_variables = [tf.Variable([[1.0, 2.0], [3.0, 4.0]], name='weights'), bias_variable, global_step]
            init_op = tf.global_variables_initializer()
            saver = tf.train.Saver(graph1_variables)
            with self.test_session() as sess:
                sess.run(init_op)
                saver.save(sess, checkpoint_path)
        with tf.Graph().as_default():
            graph2_variables = [tf.Variable([1.0, 2.0], name='weights'), bias_variable, global_step]
        out_variables = variables_helper.get_variables_available_in_checkpoint(graph2_variables, checkpoint_path, include_global_step=True)
        self.assertItemsEqual(out_variables, [bias_variable, global_step])
if __name__ == '__main__':
    tf.test.main()