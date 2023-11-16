import os
import greengrasssdk
from threading import Timer
import time
import awscam
import cv2
from threading import Thread
client = greengrasssdk.client('iot-data')
iotTopic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])
(_, frame) = awscam.getLastFrame()
(_, jpeg) = cv2.imencode('.jpg', frame)
Write_To_FIFO = True

class FIFO_Thread(Thread):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        'Constructor.'
        Thread.__init__(self)

    def run(self):
        if False:
            return 10
        fifo_path = '/tmp/results.mjpeg'
        if not os.path.exists(fifo_path):
            os.mkfifo(fifo_path)
        f = open(fifo_path, 'w')
        client.publish(topic=iotTopic, payload='Opened Pipe')
        while Write_To_FIFO:
            try:
                f.write(jpeg.tobytes())
            except IOError as e:
                continue

def greengrass_infinite_infer_run():
    if False:
        print('Hello World!')
    try:
        modelPath = '/opt/awscam/artifacts/mxnet_deploy_ssd_resnet50_300_FP16_FUSED.xml'
        modelType = 'ssd'
        input_width = 300
        input_height = 300
        max_threshold = 0.25
        outMap = {1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'dining table', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}
        results_thread = FIFO_Thread()
        results_thread.start()
        client.publish(topic=iotTopic, payload='Object detection starts now')
        mcfg = {'GPU': 1}
        model = awscam.Model(modelPath, mcfg)
        client.publish(topic=iotTopic, payload='Model loaded')
        (ret, frame) = awscam.getLastFrame()
        if ret == False:
            raise Exception('Failed to get frame from the stream')
        yscale = float(frame.shape[0] / input_height)
        xscale = float(frame.shape[1] / input_width)
        doInfer = True
        while doInfer:
            (ret, frame) = awscam.getLastFrame()
            if ret == False:
                raise Exception('Failed to get frame from the stream')
            frameResize = cv2.resize(frame, (input_width, input_height))
            inferOutput = model.doInference(frameResize)
            parsed_results = model.parseResult(modelType, inferOutput)['ssd']
            label = '{'
            for obj in parsed_results:
                if obj['prob'] > max_threshold:
                    xmin = int(xscale * obj['xmin']) + int(obj['xmin'] - input_width / 2 + input_width / 2)
                    ymin = int(yscale * obj['ymin'])
                    xmax = int(xscale * obj['xmax']) + int(obj['xmax'] - input_width / 2 + input_width / 2)
                    ymax = int(yscale * obj['ymax'])
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 165, 20), 4)
                    label += '"{}": {:.2f},'.format(outMap[obj['label']], obj['prob'])
                    label_show = '{}:    {:.2f}%'.format(outMap[obj['label']], obj['prob'] * 100)
                    cv2.putText(frame, label_show, (xmin, ymin - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 20), 4)
            label += '"null": 0.0'
            label += '}'
            client.publish(topic=iotTopic, payload=label)
            global jpeg
            (ret, jpeg) = cv2.imencode('.jpg', frame)
    except Exception as e:
        msg = 'Test failed: ' + str(e)
        client.publish(topic=iotTopic, payload=msg)
    Timer(15, greengrass_infinite_infer_run).start()
greengrass_infinite_infer_run()

def function_handler(event, context):
    if False:
        print('Hello World!')
    return