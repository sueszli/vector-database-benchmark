"""Tests for automl_vision_edge_container_predict.

The test will automatically start a container with a sample saved_model.pb,
send a request with one image, verify the response and delete the started
container.

If you want to try the test, please install
[gsutil tools](https://cloud.google.com/storage/docs/gsutil_install) and
[Docker CE](https://docs.docker.com/install/) first.

Examples:
sudo python -m pytest automl_vision_edge_container_predict_test.py
"""
import os
import subprocess
import tempfile
import time
import pytest
import automl_vision_edge_container_predict as predict
IMAGE_FILE_PATH = os.path.join(os.path.dirname(__file__), 'test.jpg')
CPU_DOCKER_GCS_PATH = '{}'.format('gcr.io/cloud-devrel-public-resources/gcloud-container-1.14.0:latest')
SAMPLE_SAVED_MODEL = '{}'.format('gs://cloud-samples-data/vision/edge_container_predict/saved_model.pb')
NAME = 'AutomlVisionEdgeContainerPredictTest'
PORT_NUMBER = 8505

@pytest.fixture
def edge_container_predict_server_port():
    if False:
        while True:
            i = 10
    subprocess.check_output(['docker', 'pull', CPU_DOCKER_GCS_PATH], env={'DOCKER_API_VERSION': '1.38'})
    if os.environ.get('TRAMPOLINE_VERSION'):
        model_path = tempfile.TemporaryDirectory()
    else:
        model_path = tempfile.TemporaryDirectory(dir=os.path.dirname(__file__))
    print('Using model_path: {}'.format(model_path))
    subprocess.check_output(['gsutil', '-m', 'cp', SAMPLE_SAVED_MODEL, model_path.name])
    subprocess.Popen(['docker', 'run', '--rm', '--name', NAME, '-v', model_path.name + ':/tmp/mounted_model/0001', '-p', str(PORT_NUMBER) + ':8501', '-t', CPU_DOCKER_GCS_PATH], env={'DOCKER_API_VERSION': '1.38'})
    time.sleep(10)
    yield PORT_NUMBER
    subprocess.check_output(['docker', 'stop', NAME], env={'DOCKER_API_VERSION': '1.38'})
    subprocess.check_output(['docker', 'rmi', CPU_DOCKER_GCS_PATH], env={'DOCKER_API_VERSION': '1.38'})
    model_path.cleanup()

@Retry()
def test_edge_container_predict(capsys, edge_container_predict_server_port):
    if False:
        while True:
            i = 10
    image_key = '1'
    response = predict.container_predict(IMAGE_FILE_PATH, image_key, PORT_NUMBER)
    assert 'predictions' in response
    assert 'key' in response['predictions'][0]
    assert image_key == response['predictions'][0]['key']