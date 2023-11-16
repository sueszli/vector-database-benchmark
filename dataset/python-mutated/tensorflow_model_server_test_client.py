"""Manual test client for tensorflow_model_server."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import grpc
import tensorflow as tf
from tensorflow.core.framework import types_pb2
from tensorflow.python.platform import flags
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
tf.compat.v1.app.flags.DEFINE_string('server', 'localhost:8500', 'inception_inference service host:port')
FLAGS = tf.compat.v1.app.flags.FLAGS

def main(_):
    if False:
        return 10
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'default'
    request.inputs['x'].dtype = types_pb2.DT_FLOAT
    request.inputs['x'].float_val.append(2.0)
    request.output_filter.append('y')
    channel = grpc.insecure_channel(FLAGS.server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    print(stub.Predict(request, 5.0))
if __name__ == '__main__':
    tf.compat.v1.app.run()