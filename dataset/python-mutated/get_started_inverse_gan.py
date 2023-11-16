from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
import tensorflow as tf
from art.estimators.classification import TensorFlowClassifier
from art.defences.preprocessor.inverse_gan import InverseGAN
from art.estimators.encoding.tensorflow import TensorFlowEncoder
from art.estimators.generation.tensorflow import TensorFlowGenerator
from art.utils import load_mnist
from art.attacks.evasion import FastGradientMethod
from examples.inverse_gan_author_utils import EncoderReconstructor, GeneratorReconstructor
logging.root.setLevel(logging.NOTSET)
logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

def create_ts1_art_mnist_classifier(min_pixel_value, max_pixel_value):
    if False:
        return 10
    input_ph = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    labels_ph = tf.placeholder(tf.int32, shape=[None, 10])
    x = tf.layers.conv2d(input_ph, filters=4, kernel_size=5, activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(x, 2, 2)
    x = tf.layers.conv2d(x, filters=10, kernel_size=5, activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(x, 2, 2)
    x = tf.contrib.layers.flatten(x)
    x = tf.layers.dense(x, 100, activation=tf.nn.relu)
    logits = tf.layers.dense(x, 10)
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels_ph))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train = optimizer.minimize(loss)
    sess.run(tf.global_variables_initializer())
    classifier = TensorFlowClassifier(clip_values=(min_pixel_value, max_pixel_value), input_ph=input_ph, output=logits, labels_ph=labels_ph, train=train, loss=loss, learning=None, sess=sess, preprocessing_defences=[])
    return classifier

def create_ts1_encoder_model(batch_size):
    if False:
        while True:
            i = 10
    encoder_reconstructor = EncoderReconstructor(batch_size)
    (unmodified_z_tensor, images_tensor) = encoder_reconstructor.generate_z_extrapolated_k()
    encoder = TensorFlowEncoder(input_ph=images_tensor, model=unmodified_z_tensor, sess=sess)
    return encoder

def create_ts1_generator_model(batch_size):
    if False:
        for i in range(10):
            print('nop')
    generator = GeneratorReconstructor(batch_size)
    generator.sess.run(generator.init_opt)
    generator = TensorFlowGenerator(input_ph=generator.z_general_placeholder, model=generator.z_hats_recs, sess=generator.sess)
    return generator

def get_accuracy(y_pred, y):
    if False:
        for i in range(10):
            print('nop')
    accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)) / len(y)
    return round(accuracy * 100, 2)

def main():
    if False:
        for i in range(10):
            print('nop')
    logging.info('Loading a Dataset')
    ((_, _), (x_test_original, y_test_original), min_pixel_value, max_pixel_value) = load_mnist()
    batch_size = 1000
    (x_test, y_test) = (x_test_original[:batch_size], y_test_original[:batch_size])
    logging.info('Creating a TS1 Mnist Classifier')
    classifier = create_ts1_art_mnist_classifier(min_pixel_value, max_pixel_value)
    classifier.fit(x_test, y_test, batch_size=batch_size, nb_epochs=3)
    logging.info('Evaluate the ART classifier on non adversarial examples')
    predictions = classifier.predict(x_test)
    accuracy_non_adv = get_accuracy(predictions, y_test)
    logging.info('Generate adversarial examples')
    attack = FastGradientMethod(classifier, eps=0.2)
    x_test_adv = attack.generate(x=x_test)
    logging.info('Evaluate the classifier on the adversarial examples')
    predictions = classifier.predict(x_test_adv)
    accuracy_adv = get_accuracy(predictions, y_test)
    logging.info('Create DefenceGAN')
    encoder = create_ts1_encoder_model(batch_size)
    generator = create_ts1_generator_model(batch_size)
    inverse_gan = InverseGAN(sess=generator._sess, gan=generator, inverse_gan=encoder)
    logging.info('Generating Defended Samples')
    x_test_defended = inverse_gan(x_test_adv, maxiter=1)
    logging.info('Evaluate the classifier on the defended examples')
    predictions = classifier.predict(x_test_defended)
    accuracy_defended = get_accuracy(predictions, y_test)
    logger.info('Accuracy on non adversarial examples: {}%'.format(accuracy_non_adv))
    logger.info('Accuracy on adversarial examples: {}%'.format(accuracy_adv))
    logger.info('Accuracy on defended examples: {}%'.format(accuracy_defended))
if __name__ == '__main__':
    main()