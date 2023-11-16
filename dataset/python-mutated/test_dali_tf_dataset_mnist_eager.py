import tensorflow as tf
from nose.tools import with_setup
import test_dali_tf_dataset_mnist as mnist
from test_utils_tensorflow import skip_for_incompatible_tf, available_gpus
from nose_utils import raises
from nose import SkipTest
from distutils.version import LooseVersion
tf.compat.v1.enable_eager_execution()

def test_keras_single_gpu():
    if False:
        print('Hello World!')
    mnist.run_keras_single_device('gpu', 0)

def test_keras_single_other_gpu():
    if False:
        print('Hello World!')
    mnist.run_keras_single_device('gpu', 1)

def test_keras_single_cpu():
    if False:
        return 10
    mnist.run_keras_single_device('cpu', 0)

@with_setup(skip_for_incompatible_tf)
@raises(Exception, 'TF device and DALI device mismatch')
def test_keras_wrong_placement_gpu():
    if False:
        i = 10
        return i + 15
    with tf.device('cpu:0'):
        model = mnist.keras_model()
        train_dataset = mnist.get_dataset('gpu', 0)
        model.fit(train_dataset, epochs=mnist.EPOCHS, steps_per_epoch=mnist.ITERATIONS)

@with_setup(skip_for_incompatible_tf)
@raises(Exception, 'TF device and DALI device mismatch')
def test_keras_wrong_placement_cpu():
    if False:
        return 10
    with tf.device('gpu:0'):
        model = mnist.keras_model()
        train_dataset = mnist.get_dataset('cpu', 0)
        model.fit(train_dataset, epochs=mnist.EPOCHS, steps_per_epoch=mnist.ITERATIONS)

@with_setup(skip_for_incompatible_tf)
def test_keras_multi_gpu_mirrored_strategy():
    if False:
        return 10
    if LooseVersion(tf.__version__) >= LooseVersion('2.12.0'):
        raise SkipTest('This test is not supported for TensorFlow 2.12')
    strategy = tf.distribute.MirroredStrategy(devices=available_gpus())
    with strategy.scope():
        model = mnist.keras_model()
    train_dataset = mnist.get_dataset_multi_gpu(strategy)
    model.fit(train_dataset, epochs=mnist.EPOCHS, steps_per_epoch=mnist.ITERATIONS)
    assert model.evaluate(train_dataset, steps=mnist.ITERATIONS)[1] > mnist.TARGET

@with_setup(mnist.clear_checkpoints, mnist.clear_checkpoints)
def test_estimators_single_gpu():
    if False:
        print('Hello World!')
    mnist.run_estimators_single_device('gpu', 0)

@with_setup(mnist.clear_checkpoints, mnist.clear_checkpoints)
def test_estimators_single_other_gpu():
    if False:
        print('Hello World!')
    mnist.run_estimators_single_device('gpu', 1)

@with_setup(mnist.clear_checkpoints, mnist.clear_checkpoints)
def test_estimators_single_cpu():
    if False:
        return 10
    mnist.run_estimators_single_device('cpu', 0)