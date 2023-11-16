import tensorflow as tf
import torch
from deepchecks.vision.datasets.classification import mnist_tensorflow as mnist_tf
from deepchecks.vision.datasets.classification import mnist_torch as mnist
from deepchecks.vision.datasets.detection import coco_tensorflow, coco_torch
from deepchecks.vision.suites.default_suites import full_suite
from tests.common import get_expected_results_length, validate_suite_result

def test_full_suite_mnist_torch(device):
    if False:
        for i in range(10):
            print('nop')
    if torch.cuda.is_available():
        mnist_train_gpu = mnist.load_dataset(train=True, object_type='VisionData', n_samples=200, device=device, pin_memory=False)
        mnist_test_gpu = mnist.load_dataset(train=False, object_type='VisionData', n_samples=200, device=device, pin_memory=False)
        mnist_iterator_train_gpu = mnist.load_dataset(train=True, use_iterable_dataset=True, object_type='VisionData', n_samples=200, device=device, pin_memory=False)
        mnist_iterator_test_gpu = mnist.load_dataset(train=False, use_iterable_dataset=True, object_type='VisionData', n_samples=200, device=device, pin_memory=False)
        suite = full_suite(imaginery_kwarg='just to make sure all checks have kwargs in the init')
        arguments = (dict(train_dataset=mnist_train_gpu, test_dataset=mnist_test_gpu, max_samples=200), dict(train_dataset=mnist_iterator_train_gpu, test_dataset=mnist_iterator_test_gpu, max_samples=200))
        for args in arguments:
            result = suite.run(**args)
            length = get_expected_results_length(suite, args)
            validate_suite_result(result, length)

def test_full_suite_mnist_tensorflow():
    if False:
        print('Hello World!')
    if len(tf.config.list_physical_devices('GPU')) > 0:
        with tf.device('/device:GPU:0'):
            suite = full_suite(imaginery_kwarg='just to make sure all checks have kwargs in the init')
            mnist_visiondata_train = mnist_tf.load_dataset(train=True)
            mnist_visiondata_test = mnist_tf.load_dataset(train=False)
            arguments = (dict(train_dataset=mnist_visiondata_train, test_dataset=mnist_visiondata_test),)
            for args in arguments:
                result = suite.run(**args)
                length = get_expected_results_length(suite, args)
                validate_suite_result(result, length)

def test_full_suite_coco_torch(device):
    if False:
        return 10
    if torch.cuda.is_available():
        coco_train_gpu = coco_torch.load_dataset(train=True, object_type='VisionData', n_samples=200, device=device, pin_memory=False)
        coco_test_gpu = coco_torch.load_dataset(train=False, object_type='VisionData', n_samples=200, device=device, pin_memory=False)
        suite = full_suite(imaginery_kwarg='just to make sure all checks have kwargs in the init')
        arguments = (dict(train_dataset=coco_train_gpu, test_dataset=coco_test_gpu),)
        for args in arguments:
            result = suite.run(**args)
            length = get_expected_results_length(suite, args)
            validate_suite_result(result, length)

def test_full_suite_coco_tensorflow():
    if False:
        return 10
    if len(tf.config.list_physical_devices('GPU')) > 0:
        with tf.device('/device:GPU:0'):
            coco_train_gpu = coco_tensorflow.load_dataset(train=True, object_type='VisionData', n_samples=200)
            coco_test_gpu = coco_tensorflow.load_dataset(train=False, object_type='VisionData', n_samples=200)
            suite = full_suite(imaginery_kwarg='just to make sure all checks have kwargs in the init')
            arguments = (dict(train_dataset=coco_train_gpu, test_dataset=coco_test_gpu),)
            for args in arguments:
                result = suite.run(**args)
                length = get_expected_results_length(suite, args)
                validate_suite_result(result, length)