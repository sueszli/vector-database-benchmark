"""Basic test of all registered models."""
import tensorflow as tf
import all_models
from entropy_coder.model import model_factory

class AllModelsTest(tf.test.TestCase):

    def testBuildModelForTraining(self):
        if False:
            return 10
        factory = model_factory.GetModelRegistry()
        model_names = factory.GetAvailableModels()
        for m in model_names:
            tf.reset_default_graph()
            global_step = tf.Variable(tf.zeros([], dtype=tf.int64), trainable=False, name='global_step')
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
            batch_size = 3
            height = 40
            width = 20
            depth = 5
            binary_codes = tf.placeholder(dtype=tf.float32, shape=[batch_size, height, width, depth])
            print('Creating model: {}'.format(m))
            model = factory.CreateModel(m)
            model.Initialize(global_step, optimizer, model.GetConfigStringForUnitTest())
            self.assertTrue(model.loss is None, 'model: {}'.format(m))
            self.assertTrue(model.train_op is None, 'model: {}'.format(m))
            self.assertTrue(model.average_code_length is None, 'model: {}'.format(m))
            model.BuildGraph(binary_codes)
            self.assertTrue(model.loss is not None, 'model: {}'.format(m))
            self.assertTrue(model.average_code_length is not None, 'model: {}'.format(m))
            if model.train_op is None:
                print('Model {} is not trainable'.format(m))
if __name__ == '__main__':
    tf.test.main()