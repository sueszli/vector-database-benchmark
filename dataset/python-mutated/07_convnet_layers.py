""" Using convolutional net on MNIST dataset of handwritten digits
MNIST dataset: http://yann.lecun.com/exdb/mnist/
CS 20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Chip Huyen (chiphuyen@cs.stanford.edu)
Lecture 07
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import tensorflow as tf
import utils

class ConvNet(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.lr = 0.001
        self.batch_size = 128
        self.keep_prob = tf.constant(0.75)
        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.n_classes = 10
        self.skip_step = 20
        self.n_test = 10000
        self.training = False

    def get_data(self):
        if False:
            return 10
        with tf.name_scope('data'):
            (train_data, test_data) = utils.get_mnist_dataset(self.batch_size)
            iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
            (img, self.label) = iterator.get_next()
            self.img = tf.reshape(img, shape=[-1, 28, 28, 1])
            self.train_init = iterator.make_initializer(train_data)
            self.test_init = iterator.make_initializer(test_data)

    def inference(self):
        if False:
            for i in range(10):
                print('nop')
        conv1 = tf.layers.conv2d(inputs=self.img, filters=32, kernel_size=[5, 5], padding='SAME', activation=tf.nn.relu, name='conv1')
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name='pool1')
        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding='SAME', activation=tf.nn.relu, name='conv2')
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name='pool2')
        feature_dim = pool2.shape[1] * pool2.shape[2] * pool2.shape[3]
        pool2 = tf.reshape(pool2, [-1, feature_dim])
        fc = tf.layers.dense(pool2, 1024, activation=tf.nn.relu, name='fc')
        dropout = tf.layers.dropout(fc, self.keep_prob, training=self.training, name='dropout')
        self.logits = tf.layers.dense(dropout, self.n_classes, name='logits')

    def loss(self):
        if False:
            return 10
        '\n        define loss function\n        use softmax cross entropy with logits as the loss function\n        compute mean cross entropy, softmax is applied internally\n        '
        with tf.name_scope('loss'):
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits)
            self.loss = tf.reduce_mean(entropy, name='loss')

    def optimize(self):
        if False:
            print('Hello World!')
        '\n        Define training op\n        using Adam Gradient Descent to minimize cost\n        '
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.gstep)

    def summary(self):
        if False:
            while True:
                i = 10
        '\n        Create summaries to write on TensorBoard\n        '
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.histogram('histogram loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def eval(self):
        if False:
            print('Hello World!')
        '\n        Count the number of right predictions in a batch\n        '
        with tf.name_scope('predict'):
            preds = tf.nn.softmax(self.logits)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    def build(self):
        if False:
            return 10
        '\n        Build the computation graph\n        '
        self.get_data()
        self.inference()
        self.loss()
        self.optimize()
        self.eval()
        self.summary()

    def train_one_epoch(self, sess, saver, init, writer, epoch, step):
        if False:
            while True:
                i = 10
        start_time = time.time()
        sess.run(init)
        self.training = True
        total_loss = 0
        n_batches = 0
        try:
            while True:
                (_, l, summaries) = sess.run([self.opt, self.loss, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                if (step + 1) % self.skip_step == 0:
                    print('Loss at step {0}: {1}'.format(step, l))
                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        saver.save(sess, 'checkpoints/convnet_layers/mnist-convnet', step)
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss / n_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))
        return step

    def eval_once(self, sess, init, writer, epoch, step):
        if False:
            while True:
                i = 10
        start_time = time.time()
        sess.run(init)
        self.training = False
        total_correct_preds = 0
        try:
            while True:
                (accuracy_batch, summaries) = sess.run([self.accuracy, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                total_correct_preds += accuracy_batch
        except tf.errors.OutOfRangeError:
            pass
        print('Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds / self.n_test))
        print('Took: {0} seconds'.format(time.time() - start_time))

    def train(self, n_epochs):
        if False:
            return 10
        '\n        The train function alternates between training one epoch and evaluating\n        '
        utils.safe_mkdir('checkpoints')
        utils.safe_mkdir('checkpoints/convnet_layers')
        writer = tf.summary.FileWriter('./graphs/convnet_layers', tf.get_default_graph())
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/convnet_layers/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            step = self.gstep.eval()
            for epoch in range(n_epochs):
                step = self.train_one_epoch(sess, saver, self.train_init, writer, epoch, step)
                self.eval_once(sess, self.test_init, writer, epoch, step)
        writer.close()
if __name__ == '__main__':
    model = ConvNet()
    model.build()
    model.train(n_epochs=15)