"""This is an example to call REST API from TFServing docker containers.

Examples:
    python automl_vision_edge_container_predict.py \\
    --image_file_path=./test.jpg --image_key=1 --port_number=8501

"""
import argparse
import base64
import cv2
import io
import json
import requests

def preprocess_image(image_file_path, max_width, max_height):
    if False:
        while True:
            i = 10
    'Preprocesses input images for AutoML Vision Edge models.\n\n    Args:\n        image_file_path: Path to a local image for the prediction request.\n        max_width: The max width for preprocessed images. The max width is 640\n            (1024) for AutoML Vision Image Classfication (Object Detection)\n            models.\n        max_height: The max width for preprocessed images. The max height is\n            480 (1024) for AutoML Vision Image Classfication (Object\n            Detetion) models.\n    Returns:\n        The preprocessed encoded image bytes.\n    '
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
    im = cv2.imread(image_file_path)
    [height, width, _] = im.shape
    if height > max_height or width > max_width:
        ratio = max(height / float(max_width), width / float(max_height))
        new_height = int(height / ratio + 0.5)
        new_width = int(width / ratio + 0.5)
        resized_im = cv2.resize(im, (new_width, new_height), interpolation=cv2.INTER_AREA)
        (_, processed_image) = cv2.imencode('.jpg', resized_im, encode_param)
    else:
        (_, processed_image) = cv2.imencode('.jpg', im, encode_param)
    return base64.b64encode(processed_image).decode('utf-8')

def container_predict(image_file_path, image_key, port_number=8501):
    if False:
        while True:
            i = 10
    'Sends a prediction request to TFServing docker container REST API.\n\n    Args:\n        image_file_path: Path to a local image for the prediction request.\n        image_key: Your chosen string key to identify the given image.\n        port_number: The port number on your device to accept REST API calls.\n    Returns:\n        The response of the prediction request.\n    '
    encoded_image = preprocess_image(image_file_path=image_file_path, max_width=640, max_height=480)
    instances = {'instances': [{'image_bytes': {'b64': str(encoded_image)}, 'key': image_key}]}
    url = 'http://localhost:{}/v1/models/default:predict'.format(port_number)
    response = requests.post(url, data=json.dumps(instances))
    print(response.json())
    return response.json()

def main():
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_file_path', type=str)
    parser.add_argument('--image_key', type=str, default='1')
    parser.add_argument('--port_number', type=int, default=8501)
    args = parser.parse_args()
    container_predict(args.image_file_path, args.image_key, args.port_number)
if __name__ == '__main__':
    main()