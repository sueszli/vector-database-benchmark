import cv2
import math
import time
import numpy as np
import threading
from datetime import datetime
import csv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rospy
from drone.msg import Altitude as AltitudeMsg
from drone.msg import Attitude as AttitudeMsg
from drone.msg import GPSinfo as GPSMsg
from drone.msg import ComputeMsg
from std_msgs.msg import Float32
from std_msgs.msg import Bool
from src.compute import Detector as detector
FRAME = None
FRAME_OVERLAY_NP = None
SONAR = 0.0
REL_ALT = 0.0
ABS_ALT = 0.0
H = 0.0
AREA_DIST_VERT = 0.0
AREA_DIST_HORIZ = 0.0
CAM_V_FOV = 48.8
SONAR_MAX = 300
SONAR_MIN = 2
PIXEL_SIZE = 0.0
CAM_AREA = 0.0
MAX_AREA = 10000
MIN_AREA = 1
LAT_LNG = (0.0, 0.0)
SAT = 0
BATTERY = 0
AIR_TEMP = 0
CSV_HEADER = ['time', 'stamp', 'people', 'warning', 'density', 'latitude', 'longitude', 'sat', 'altitude[m]', 'height[m]', 'temp[°C]', 'angle[deg]', 'area[m2]', 'battery[%]']
CSV_FILENAME = 'data.csv'
OUT_PATH = '/home/andrei/Desktop/mUAV/catkin_ws/src/drone/data/out/'
CSV_PATH = ''
TODAY_FOLDER_PATH = OUT_PATH + str(datetime.now().strftime('%d%h%y'))
WIDTH_TILES = 4
HEIGHT_TILES = 3
DETECTION_STARTED = False
FRAME_READY = False
DETECTION_RING = 1
PREV_OVERLAY = None
PREV_DETECTIONS = []
OVERLAY_INDEX = 0
BEST_OVERLAY = None
BEST_DETECTIONS = []
DISTANCING_THRESHOLD = 150
compute_pub = None
height_ring = 3
heights = [0] * height_ring
height_index = 0

def sonar_callback(data):
    if False:
        return 10
    global SONAR
    SONAR = data.data
    resolveHeight()

def alt_callback(data):
    if False:
        i = 10
        return i + 15
    global REL_ALT, ABS_ALT, AIR_TEMP
    REL_ALT = round(data.relative, 1)
    ABS_ALT = round(data.absolute, 1)
    AIR_TEMP = round(data.temp, 1)
    resolveHeight()

def gps_callback(data):
    if False:
        return 10
    global LAT_LNG, SAT
    LAT_LNG = (round(data.lat, 5), round(data.lng, 5))
    SAT = data.fix

def attitude_callback(data):
    if False:
        return 10
    global BATTERY
    BATTERY = data.percentage

def resolveHeight():
    if False:
        print('Hello World!')
    global H, heights, height_index
    if SONAR_MIN <= SONAR <= SONAR_MAX:
        h = SONAR / 100
    else:
        h = REL_ALT
    heights[height_index] = h
    height_index += 1
    if height_index == height_ring:
        height_index = 0
        H = round(np.average(heights), 1)
    if not DETECTION_STARTED and compute_pub is not None:
        compute_pub.publish(ComputeMsg(0, 0, 0, CAM_AREA, H))

def computeVisibleCamArea(cam_angle):
    if False:
        i = 10
        return i + 15
    global AREA_DIST_VERT, AREA_DIST_HORIZ
    AREA_DIST_HORIZ = 1.2 * H
    theta = cam_angle - CAM_V_FOV / 2
    alpha = 90.0 - theta - CAM_V_FOV
    coeff = 1.0 / math.tan(math.radians(alpha)) - math.sin(math.radians(theta))
    if coeff < 0:
        return MIN_AREA
    AREA_DIST_VERT = coeff * H
    computePixelSize(frame_height=1232, frame_width=1632)
    area = round(AREA_DIST_VERT * AREA_DIST_HORIZ, 2)
    area *= 1.15
    return min(max(area, MIN_AREA), MAX_AREA)

def computePixelSize(frame_height, frame_width):
    if False:
        i = 10
        return i + 15
    global PIXEL_SIZE
    PIXEL_SIZE = (AREA_DIST_VERT / frame_height + AREA_DIST_HORIZ / frame_width) / 2 * 100

def frame_saved_callback(data):
    if False:
        print('Hello World!')
    global FRAME_READY, FRAME
    FRAME = cv2.imread('/home/andrei/Desktop/mUAV/catkin_ws/src/drone/data/live/frame.jpg')
    FRAME_READY = True

def process_data(current_overlay, current_detections):
    if False:
        i = 10
        return i + 15
    global PREV_DETECTIONS, PREV_OVERLAY, BEST_DETECTIONS, BEST_OVERLAY, OVERLAY_INDEX
    current_people = len(current_detections)
    if current_people > 0:
        OVERLAY_INDEX += 1
        if current_people > len(BEST_DETECTIONS):
            BEST_DETECTIONS = current_detections
            BEST_OVERLAY = current_overlay
        PREV_DETECTIONS = current_detections
        PREV_OVERLAY = current_overlay
        if OVERLAY_INDEX == DETECTION_RING:
            OVERLAY_INDEX = 0
            people_count = len(BEST_DETECTIONS)
            overlay = BEST_OVERLAY
            detections = BEST_DETECTIONS
            time_of_day = str(datetime.now().strftime('%H:%M:%S'))
            timestamp = round(time.time())
            density = round(people_count / CAM_AREA * 10, 2)
            (overlay, neck_breathers) = check_social_distancing(detections, overlay)
            rospy.loginfo('{}: {} people, {}/10m2 density, {} too close'.format(rospy.get_caller_id(), people_count, density, neck_breathers))
            compute_pub.publish(ComputeMsg(people_count, neck_breathers, density, CAM_AREA, H))
            if CSV_PATH is not None:
                with open(CSV_PATH, 'a', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    row = [time_of_day, timestamp, people_count, neck_breathers, density, LAT_LNG[0], LAT_LNG[1], SAT, ABS_ALT, H, AIR_TEMP, rospy.get_param('/physical/camera_angle'), round(CAM_AREA, 1), BATTERY]
                    writer.writerow(row)
            cv2.imwrite(TODAY_FOLDER_PATH + '/{}_{}_{}.jpg'.format(timestamp, people_count, neck_breathers), overlay)
            BEST_OVERLAY = None
            BEST_DETECTIONS = []
    else:
        rospy.loginfo('{}: No people detected!'.format(rospy.get_caller_id()))
        compute_pub.publish(ComputeMsg(0, 0, 0, CAM_AREA, H))

def video_mock_test():
    if False:
        for i in range(10):
            print('nop')
    global PREV_DETECTIONS, PREV_OVERLAY, BEST_DETECTIONS, BEST_OVERLAY, OVERLAY_INDEX, H
    video = cv2.VideoCapture('/home/andrei/Desktop/mUAV/catkin_ws/src/drone/data/videos/fountain.mp4')
    while video.isOpened():
        (ret, frame) = video.read()
        if not ret:
            break
        resized = cv2.resize(frame, (1080, 1920), interpolation=cv2.INTER_NEAREST)
        (overlay, detections) = detector.run_inference(resized, 3, 5)
        current_people = len(detections)
        if current_people > 0:
            OVERLAY_INDEX += 1
            if current_people > len(BEST_DETECTIONS):
                BEST_DETECTIONS = detections
                BEST_OVERLAY = overlay
            PREV_DETECTIONS = detections
            PREV_OVERLAY = overlay
            if OVERLAY_INDEX == DETECTION_RING:
                OVERLAY_INDEX = 0
                people_count = len(BEST_DETECTIONS)
                overlay = BEST_OVERLAY
                detections = BEST_DETECTIONS
                time_of_day = str(datetime.now().strftime('%H:%M:%S'))
                timestamp = round(time.time())
                H = 10.0
                area = computeVisibleCamArea(45)
                density = round(people_count / area * 10, 2)
                (overlay, neck_breathers) = check_social_distancing(detections, overlay)
                rospy.loginfo('{}: {} people, {}/10m2 density, {} too close'.format(rospy.get_caller_id(), people_count, density, neck_breathers))
                if CSV_PATH is not None:
                    with open(CSV_PATH, 'a', newline='') as csv_file:
                        writer = csv.writer(csv_file)
                        row = [time_of_day, timestamp, people_count, neck_breathers, density, LAT_LNG[0], LAT_LNG[1], SAT, ABS_ALT, 10, AIR_TEMP, 45, area, BATTERY]
                        writer.writerow(row)
                cv2.imwrite('/home/andrei/Desktop/mUAV/catkin_ws/src/drone/data/out/random/{}_{}.jpg'.format(timestamp, people_count), overlay)
    else:
        rospy.loginfo('{}: No people detected!'.format(rospy.get_caller_id()))
    video.release()
    rospy.loginfo('{}: Video processing done!'.format(rospy.get_caller_id()))

def get_physical_distance(detection_box_1, detection_box_2):
    if False:
        for i in range(10):
            print('nop')
    b1X = detection_box_1.Center[0]
    b1Y = detection_box_1.Center[1]
    b2X = detection_box_2.Center[0]
    b2Y = detection_box_2.Center[1]
    pixel_distance = math.sqrt((b2X - b1X) ** 2 + (b2Y - b1Y) ** 2)
    physical_distance = pixel_distance * PIXEL_SIZE
    return physical_distance

def check_social_distancing(detections, overlay):
    if False:
        print('Hello World!')
    highlighted = []
    for idx1 in range(len(detections) - 1):
        for idx2 in range(idx1 + 1, len(detections)):
            physical_distance = get_physical_distance(detections[idx1], detections[idx2])
            if physical_distance < DISTANCING_THRESHOLD:
                cv2.rectangle(overlay, (int(detections[idx1].Left), int(detections[idx1].Top)), (int(detections[idx1].Right), int(detections[idx1].Bottom)), color=(0, 0, 255), thickness=2)
                cv2.rectangle(overlay, (int(detections[idx2].Left), int(detections[idx2].Top)), (int(detections[idx2].Right), int(detections[idx2].Bottom)), color=(0, 0, 255), thickness=2)
                if idx1 not in highlighted:
                    highlighted.append(idx1)
                if idx2 not in highlighted:
                    highlighted.append(idx2)
    neck_breathers = len(highlighted)
    return (overlay, neck_breathers)

def generate_statistics():
    if False:
        i = 10
        return i + 15
    x = []
    y = []
    z = []
    with open('/home/andrei/Desktop/mUAV/catkin_ws/src/drone/data/out/csv/15Mar21.csv', 'r') as csvfile:
        plots = csv.DictReader(csvfile, delimiter=',')
        initial_time = 9999999999
        initial = True
        for row in plots:
            timestamp = int(row['stamp     '])
            people = int(row['people'])
            density = float(row['density'])
            altitude = float(row['altitude[m]'])
            height = float(row['height[m]'])
            temp = float(row['temp[°C]'])
            area = float(row['area[m2]'])
            battery = int(row['battery[%]'])
            if initial:
                initial_time = timestamp
                initial = False
            x.append((timestamp - initial_time) / 60)
            y.append(density)
            z.append(people)
        (fig, axis) = plt.subplots(2, sharex='all')
        axis[0].plot(x, y, label='Density')
        axis[0].set_title('Density')
        axis[0].set(ylabel='People/10m2')
        axis[1].plot(x, z, label='Total people')
        axis[1].set_title('People')
        plt.xlabel('Time [min]')
        plt.savefig('/home/andrei/Desktop/mUAV/catkin_ws/src/drone/data/out/csv/GRAPH2.png')

def detectionThread():
    if False:
        i = 10
        return i + 15
    global FRAME_OVERLAY_NP, FRAME_READY, OVERLAY_INDEX
    frame_pub = rospy.Publisher('FrameRequested', Bool, queue_size=1)
    while True:
        if not DETECTION_STARTED or not FRAME_READY:
            continue
        if FRAME is not None:
            frame = FRAME.copy()
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            cv2.imwrite(TODAY_FOLDER_PATH + '/{}.jpg'.format(round(time.time())), frame)
            (FRAME_OVERLAY_NP, detections) = detector.run_inference(frame, WIDTH_TILES, HEIGHT_TILES)
            cv2.imwrite(TODAY_FOLDER_PATH + '/{}_{}.jpg'.format(round(time.time()), len(detections)), FRAME_OVERLAY_NP)
            process_data(FRAME_OVERLAY_NP.copy(), detections)
            FRAME_READY = False
            frame_pub.publish(Bool(True))

def main():
    if False:
        print('Hello World!')
    global DETECTION_STARTED, CAM_AREA, CSV_FILENAME, CSV_PATH, compute_pub
    rospy.init_node('ComputingNode')
    rospy.Subscriber('SonarReading', Float32, sonar_callback)
    rospy.Subscriber('Altitude', AltitudeMsg, alt_callback)
    rospy.Subscriber('FrameSaved', Bool, frame_saved_callback)
    rospy.Subscriber('GPS', GPSMsg, gps_callback)
    rospy.Subscriber('CraftAttitude', AttitudeMsg, attitude_callback)
    compute_pub = rospy.Publisher('Compute', ComputeMsg, queue_size=1)
    detector.load_net('ssd-mobilenet-v2', threshold=0.35)
    CSV_PATH = TODAY_FOLDER_PATH + '/' + CSV_FILENAME
    if not os.path.exists(TODAY_FOLDER_PATH):
        os.makedirs(TODAY_FOLDER_PATH)
        with open(CSV_PATH, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(CSV_HEADER)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        cam_angle = rospy.get_param('/physical/camera_angle')
        DETECTION_STARTED = rospy.get_param('/run/detection_started')
        CAM_AREA = computeVisibleCamArea(cam_angle)
        rate.sleep()
if __name__ == '__main__':
    try:
        detector_thread = threading.Thread(target=detectionThread)
        detector_thread.daemon = True
        detector_thread.start()
        main()
    except rospy.ROSInterruptException as error:
        pass