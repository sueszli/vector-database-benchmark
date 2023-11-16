import threading
import time
import cv2
import rospy
from flask import Flask, Response
from std_msgs.msg import Bool
from src.sensors import Camera as cam
SEND_FPS = 20
FRAME_TIME = 1.0 / SEND_FPS
last_send_loop = 0.0
DETECTION_MAX_FPS = 2
DETECTION_FRAME_TIME = 1.0 / DETECTION_MAX_FPS
last_detection_loop = 0.0
HOST_IP = '192.168.137.113'
HOST_PORT = 60500
FRAME_WRITE_LOCATION = '/home/andrei/Desktop/mUAV/catkin_ws/src/drone/data/live'
app = Flask(__name__)
threading.Thread(target=lambda : rospy.init_node('FlaskServer', disable_signals=True)).start()
SAVE_FRAME = True

def writeFramesToDisk(pub):
    if False:
        for i in range(10):
            print('nop')
    global last_detection_loop, SAVE_FRAME
    while True:
        if not SAVE_FRAME or time.time() - last_detection_loop < DETECTION_FRAME_TIME:
            continue
        last_detection_loop = time.time()
        detection_started = rospy.get_param('/run/detection_started')
        if detection_started:
            if cam.FRAME is not None:
                frame = cam.FRAME.copy()
                cv2.imwrite(FRAME_WRITE_LOCATION + '/frame.jpg', frame)
                pub.publish(Bool(True))
                SAVE_FRAME = False

def handleFrameRequested(data):
    if False:
        for i in range(10):
            print('nop')
    global SAVE_FRAME
    SAVE_FRAME = True

def encodeFrame():
    if False:
        return 10
    global last_send_loop
    while True:
        if time.time() - last_send_loop < FRAME_TIME:
            continue
        last_send_loop = time.time()
        frame = cam.FRAME.copy()
        resized = cv2.resize(frame, (410, 308), interpolation=cv2.INTER_NEAREST)
        (_, encoded) = cv2.imencode('.jpg', resized)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + bytearray(encoded) + b'\r\n')

@app.route('/live_feed')
def streamFrames():
    if False:
        return 10
    return Response(encodeFrame(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':
    try:
        frame_share_pub = rospy.Publisher('FrameSaved', Bool, queue_size=1)
        rospy.Subscriber('FrameRequested', Bool, handleFrameRequested)
        camera_thread = threading.Thread(target=lambda : cam.captureFrames())
        camera_thread.daemon = True
        camera_thread.start()
        frame_thread = threading.Thread(target=writeFramesToDisk, args=[frame_share_pub])
        frame_thread.daemon = True
        frame_thread.start()
        app.run(host=HOST_IP, port=HOST_PORT, use_reloader=False, threaded=True)
        cam.video_capture.release()
        camera_thread.join(1)
    except rospy.ROSInterruptException as error:
        pass