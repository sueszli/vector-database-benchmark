"""A client that performs inferences on a ResNet model using the REST API.

The client downloads a test image of a cat, queries the server over the REST API
with the test image repeatedly and measures how long it takes to respond.

The client expects a TensorFlow Serving ModelServer running a ResNet SavedModel
from:

https://github.com/tensorflow/models/tree/master/official/vision/image_classification/resnet#pretrained-models

Typical usage example:

    resnet_client.py
"""
from __future__ import print_function
import base64
import io
import json
import numpy as np
from PIL import Image
import requests
SERVER_URL = 'http://localhost:8501/v1/models/resnet:predict'
IMAGE_URL = 'https://tensorflow.org/images/blogs/serving/cat.jpg'
MODEL_ACCEPT_JPG = False

def main():
    if False:
        i = 10
        return i + 15
    dl_request = requests.get(IMAGE_URL, stream=True)
    dl_request.raise_for_status()
    if MODEL_ACCEPT_JPG:
        jpeg_bytes = base64.b64encode(dl_request.content).decode('utf-8')
        predict_request = '{"instances" : [{"b64": "%s"}]}' % jpeg_bytes
    else:
        jpeg_rgb = Image.open(io.BytesIO(dl_request.content))
        jpeg_rgb = np.expand_dims(np.array(jpeg_rgb) / 255.0, 0).tolist()
        predict_request = json.dumps({'instances': jpeg_rgb})
    for _ in range(3):
        response = requests.post(SERVER_URL, data=predict_request)
        response.raise_for_status()
    total_time = 0
    num_requests = 10
    for _ in range(num_requests):
        response = requests.post(SERVER_URL, data=predict_request)
        response.raise_for_status()
        total_time += response.elapsed.total_seconds()
        prediction = response.json()['predictions'][0]
    print('Prediction class: {}, avg latency: {} ms'.format(np.argmax(prediction), total_time * 1000 / num_requests))
if __name__ == '__main__':
    main()