"""A client that talks to tensorflow_model_server loaded with mnist model.

The client downloads test images of mnist data set, queries the service with
such test images to get predictions, and calculates the inference error rate.

Typical usage example:

    mnist_client.py --num_tests=100 --server=localhost:9000
"""
from __future__ import print_function
import sys
import threading
import grpc
import numpy
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import mnist_input_data
tf.compat.v1.app.flags.DEFINE_integer('concurrency', 1, 'maximum number of concurrent inference requests')
tf.compat.v1.app.flags.DEFINE_integer('num_tests', 100, 'Number of test images')
tf.compat.v1.app.flags.DEFINE_string('server', '', 'PredictionService host:port')
tf.compat.v1.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory. ')
FLAGS = tf.compat.v1.app.flags.FLAGS

class _ResultCounter(object):
    """Counter for the prediction results."""

    def __init__(self, num_tests, concurrency):
        if False:
            for i in range(10):
                print('nop')
        self._num_tests = num_tests
        self._concurrency = concurrency
        self._error = 0
        self._done = 0
        self._active = 0
        self._condition = threading.Condition()

    def inc_error(self):
        if False:
            print('Hello World!')
        with self._condition:
            self._error += 1

    def inc_done(self):
        if False:
            return 10
        with self._condition:
            self._done += 1
            self._condition.notify()

    def dec_active(self):
        if False:
            for i in range(10):
                print('nop')
        with self._condition:
            self._active -= 1
            self._condition.notify()

    def get_error_rate(self):
        if False:
            i = 10
            return i + 15
        with self._condition:
            while self._done != self._num_tests:
                self._condition.wait()
            return self._error / float(self._num_tests)

    def throttle(self):
        if False:
            for i in range(10):
                print('nop')
        with self._condition:
            while self._active == self._concurrency:
                self._condition.wait()
            self._active += 1

def _create_rpc_callback(label, result_counter):
    if False:
        print('Hello World!')
    'Creates RPC callback function.\n\n  Args:\n    label: The correct label for the predicted example.\n    result_counter: Counter for the prediction result.\n  Returns:\n    The callback function.\n  '

    def _callback(result_future):
        if False:
            while True:
                i = 10
        'Callback function.\n\n    Calculates the statistics for the prediction result.\n\n    Args:\n      result_future: Result future of the RPC.\n    '
        exception = result_future.exception()
        if exception:
            result_counter.inc_error()
            print(exception)
        else:
            sys.stdout.write('.')
            sys.stdout.flush()
            response = numpy.array(result_future.result().outputs['scores'].float_val)
            prediction = numpy.argmax(response)
            if label != prediction:
                result_counter.inc_error()
        result_counter.inc_done()
        result_counter.dec_active()
    return _callback

def do_inference(hostport, work_dir, concurrency, num_tests):
    if False:
        i = 10
        return i + 15
    'Tests PredictionService with concurrent requests.\n\n  Args:\n    hostport: Host:port address of the PredictionService.\n    work_dir: The full path of working directory for test data set.\n    concurrency: Maximum number of concurrent requests.\n    num_tests: Number of test images to use.\n\n  Returns:\n    The classification error rate.\n\n  Raises:\n    IOError: An error occurred processing test data set.\n  '
    test_data_set = mnist_input_data.read_data_sets(work_dir).test
    channel = grpc.insecure_channel(hostport)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    result_counter = _ResultCounter(num_tests, concurrency)
    for _ in range(num_tests):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'mnist'
        request.model_spec.signature_name = 'predict_images'
        (image, label) = test_data_set.next_batch(1)
        request.inputs['images'].CopyFrom(tf.make_tensor_proto(image[0], shape=[1, image[0].size]))
        result_counter.throttle()
        result_future = stub.Predict.future(request, 5.0)
        result_future.add_done_callback(_create_rpc_callback(label[0], result_counter))
    return result_counter.get_error_rate()

def main(_):
    if False:
        print('Hello World!')
    if FLAGS.num_tests > 10000:
        print('num_tests should not be greater than 10k')
        return
    if not FLAGS.server:
        print('please specify server host:port')
        return
    error_rate = do_inference(FLAGS.server, FLAGS.work_dir, FLAGS.concurrency, FLAGS.num_tests)
    print('\nInference error rate: %s%%' % (error_rate * 100))
if __name__ == '__main__':
    tf.compat.v1.app.run()