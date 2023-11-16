"""Send JPEG image to tensorflow_model_server loaded with ResNet model.

"""
from __future__ import print_function
import io
import grpc
import numpy as np
from PIL import Image
import requests
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
IMAGE_URL = 'https://tensorflow.org/images/blogs/serving/cat.jpg'
tf.compat.v1.app.flags.DEFINE_string('server', 'localhost:8500', 'PredictionService host:port')
tf.compat.v1.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.compat.v1.app.flags.FLAGS
MODEL_ACCEPT_JPG = False

def main(_):
    if False:
        i = 10
        return i + 15
    if FLAGS.image:
        with open(FLAGS.image, 'rb') as f:
            data = f.read()
    else:
        dl_request = requests.get(IMAGE_URL, stream=True)
        dl_request.raise_for_status()
        data = dl_request.content
    if not MODEL_ACCEPT_JPG:
        data = Image.open(io.BytesIO(dl_request.content))
        data = np.array(data) / 255.0
        data = np.expand_dims(data, 0)
        data = data.astype(np.float32)
    channel = grpc.insecure_channel(FLAGS.server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'resnet'
    request.model_spec.signature_name = 'serving_default'
    request.inputs['input_1'].CopyFrom(tf.make_tensor_proto(data))
    result = stub.Predict(request, 10.0)
    result = result.outputs['activation_49'].float_val
    print('Prediction class: {}'.format(np.argmax(result)))
if __name__ == '__main__':
    tf.compat.v1.app.run()