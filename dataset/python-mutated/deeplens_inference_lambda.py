from threading import Thread, Event
import os
import json
import numpy as np
import awscam
import cv2
import greengrasssdk

def lambda_handler(event, context):
    if False:
        print('Hello World!')
    'Empty entry point to the Lambda function invoked from the edge.'
    return

class LocalDisplay(Thread):
    """Class for facilitating the local display of inference results
    (as images). The class is designed to run on its own thread. In
    particular the class dumps the inference results into a FIFO
    located in the tmp directory (which lambda has access to). The
    results can be rendered using mplayer by typing:
    mplayer -demuxer lavf -lavfdopts format=mjpeg:probesize=32 /tmp/results.mjpeg
    """

    def __init__(self, resolution):
        if False:
            while True:
                i = 10
        'resolution - Desired resolution of the project stream'
        super(LocalDisplay, self).__init__()
        RESOLUTION = {'1080p': (1920, 1080), '720p': (1280, 720), '480p': (858, 480)}
        if resolution not in RESOLUTION:
            raise Exception('Invalid resolution')
        self.resolution = RESOLUTION[resolution]
        self.frame = cv2.imencode('.jpg', 255 * np.ones([640, 480, 3]))[1]
        self.stop_request = Event()

    def run(self):
        if False:
            return 10
        'Overridden method that continually dumps images to the desired\n        FIFO file.\n        '
        result_path = '/tmp/results.mjpeg'
        if not os.path.exists(result_path):
            os.mkfifo(result_path)
        with open(result_path, 'w') as fifo_file:
            while not self.stop_request.isSet():
                try:
                    fifo_file.write(self.frame.tobytes())
                except IOError:
                    continue

    def set_frame_data(self, frame):
        if False:
            return 10
        'Method updates the image data. This currently encodes the\n        numpy array to jpg but can be modified to support other encodings.\n        frame - Numpy array containing the image data of the next frame\n                in the project stream.\n        '
        (ret, jpeg) = cv2.imencode('.jpg', cv2.resize(frame, self.resolution))
        if not ret:
            raise Exception('Failed to set frame data')
        self.frame = jpeg

    def join(self):
        if False:
            return 10
        self.stop_request.set()

def infinite_infer_run():
    if False:
        for i in range(10):
            print('nop')
    'Run the DeepLens inference loop frame by frame'
    try:
        model_type = 'classification'
        output_map = {0: 'dog', 1: 'cat'}
        client = greengrasssdk.client('iot-data')
        iot_topic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])
        local_display = LocalDisplay('480p')
        local_display.start()
        model_path = '/opt/awscam/artifacts/mxnet_resnet18-catsvsdogs_FP32_FUSED.xml'
        client.publish(topic=iot_topic, payload='Loading action cat-dog model')
        model = awscam.Model(model_path, {'GPU': 1})
        client.publish(topic=iot_topic, payload='Cat-Dog model loaded')
        num_top_k = 2
        input_height = 224
        input_width = 224
        while True:
            ...
    except Exception as ex:
        client.publish(topic=iot_topic, payload='Error in cat-dog lambda: {}'.format(ex))
        (ret, frame) = awscam.getLastFrame()
        if not ret:
            raise Exception('Failed to get frame from the stream')
        frame_resize = cv2.resize(frame, (input_height, input_width))
        parsed_inference_results = model.parseResult(model_type, model.doInference(frame_resize))
        top_k = parsed_inference_results[model_type][0:num_top_k]
        cv2.putText(frame, output_map[top_k[0]['label']], (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 165, 20), 8)
        local_display.set_frame_data(frame)
        cloud_output = {}
        for obj in top_k:
            cloud_output[output_map[obj['label']]] = obj['prob']
        client.publish(topic=iot_topic, payload=json.dumps(cloud_output))
' A sample lambda for cat-dog detection'
from threading import Thread, Event
import os
import json
import numpy as np
import awscam
import cv2
import greengrasssdk

def lambda_handler(event, context):
    if False:
        for i in range(10):
            print('nop')
    'Empty entry point to the Lambda function invoked from the edge.'
    return

class LocalDisplay(Thread):
    """Class for facilitating the local display of inference results
    (as images). The class is designed to run on its own thread. In
    particular the class dumps the inference results into a FIFO
    located in the tmp directory (which lambda has access to). The
    results can be rendered using mplayer by typing:
    mplayer -demuxer lavf -lavfdopts format=mjpeg:probesize=32 /tmp/results.mjpeg
    """

    def __init__(self, resolution):
        if False:
            return 10
        'resolution - Desired resolution of the project stream'
        super(LocalDisplay, self).__init__()
        RESOLUTION = {'1080p': (1920, 1080), '720p': (1280, 720), '480p': (858, 480)}
        if resolution not in RESOLUTION:
            raise Exception('Invalid resolution')
        self.resolution = RESOLUTION[resolution]
        self.frame = cv2.imencode('.jpg', 255 * np.ones([640, 480, 3]))[1]
        self.stop_request = Event()

    def run(self):
        if False:
            return 10
        'Overridden method that continually dumps images to the desired\n        FIFO file.\n        '
        result_path = '/tmp/results.mjpeg'
        if not os.path.exists(result_path):
            os.mkfifo(result_path)
        with open(result_path, 'w') as fifo_file:
            while not self.stop_request.isSet():
                try:
                    fifo_file.write(self.frame.tobytes())
                except IOError:
                    continue

    def set_frame_data(self, frame):
        if False:
            while True:
                i = 10
        'Method updates the image data. This currently encodes the\n        numpy array to jpg but can be modified to support other encodings.\n        frame - Numpy array containing the image data of the next frame\n                in the project stream.\n        '
        (ret, jpeg) = cv2.imencode('.jpg', cv2.resize(frame, self.resolution))
        if not ret:
            raise Exception('Failed to set frame data')
        self.frame = jpeg

    def join(self):
        if False:
            i = 10
            return i + 15
        self.stop_request.set()

def infinite_infer_run():
    if False:
        print('Hello World!')
    'Run the DeepLens inference loop frame by frame'
    try:
        model_type = 'classification'
        output_map = {0: 'dog', 1: 'cat'}
        client = greengrasssdk.client('iot-data')
        iot_topic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])
        local_display = LocalDisplay('480p')
        local_display.start()
        model_path = '/opt/awscam/artifacts/mxnet_resnet18-catsvsdogs_FP32_FUSED.xml'
        client.publish(topic=iot_topic, payload='Loading action cat-dog model')
        model = awscam.Model(model_path, {'GPU': 1})
        client.publish(topic=iot_topic, payload='Cat-Dog model loaded')
        num_top_k = 2
        input_height = 224
        input_width = 224
        while True:
            (ret, frame) = awscam.getLastFrame()
            if not ret:
                raise Exception('Failed to get frame from the stream')
            frame_resize = cv2.resize(frame, (input_height, input_width))
            parsed_inference_results = model.parseResult(model_type, model.doInference(frame_resize))
            top_k = parsed_inference_results[model_type][0:num_top_k]
            cv2.putText(frame, output_map[top_k[0]['label']], (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 165, 20), 8)
            local_display.set_frame_data(frame)
            cloud_output = {}
            for obj in top_k:
                cloud_output[output_map[obj['label']]] = obj['prob']
            client.publish(topic=iot_topic, payload=json.dumps(cloud_output))
    except Exception as ex:
        client.publish(topic=iot_topic, payload='Error in cat-dog lambda: {}'.format(ex))
infinite_infer_run()