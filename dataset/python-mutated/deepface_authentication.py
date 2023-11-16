import os
from matplotlib import pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from deepface import DeepFace
from deepface.basemodels import Facenet
import cv2
import time
import numpy as np
import pandas as pd
import math

def face_confidence(face_distance, face_match_threshold=0.6):
    if False:
        i = 10
        return i + 15
    return round(10 - face_distance, 2) * 10

class FaceRecognition:
    cnt = 0
    color_bgr = {'green': (0, 255, 0), 'red': (0, 0, 255), 'blue': (255, 0, 0)}
    new_frame_time = 0
    models = {'default': 'Facenet'}
    detectors = {'default': 'opencv'}
    metrics = {'default': 'euclidean'}
    db_path = 'training_data/face'

    def take_picture(self, name):
        if False:
            return 10
        flag = True
        while flag:
            if not os.path.exists('training_data/face/' + name):
                os.mkdir('training_data/face/' + name)
                flag = False
            else:
                print('Name already exists. I will automatically remove')
                files_in_dir = os.listdir(f'training_data/face/{name}')
                for f in files_in_dir:
                    os.remove(f'training_data/face/{name}/{f}')
                os.rmdir(f'training_data/face/{name}')
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        img_counter = 0
        cv2.namedWindow('Take the picture', cv2.WINDOW_NORMAL)
        cam.set(3, 640)
        cam.set(4, 480)
        prev_frame_time = 0
        while True:
            (ret, frame) = cam.read()
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            if not ret:
                print('Failed to grab frame')
                break
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
            net = cv2.dnn.readNetFromCaffe('models/face/deploy.prototxt.txt', 'models/face/res10_300x300_ssd_iter_140000.caffemodel')
            net.setInput(blob)
            detections = net.forward()
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (startX, startY, endX, endY) = box.astype('int')
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, 'FPS: {:.2f}'.format(fps), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, str(img_counter) + 'Pic', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Take the picture', frame)
            k = cv2.waitKey(1)
            if k == ord('q'):
                print('Closing collecting data ................')
                break
            elif k % 256 == 32:
                img_name = 'training_data/face/{}/{}.png'.format(name, img_counter)
                cv2.imwrite(img_name, frame)
                print('{} written!'.format(img_name))
                img_name = 'training_data/face/{}/{}.png'.format(name, img_counter)
                cv2.imwrite(img_name, frame)
                print('{} written!'.format(img_name[19:]))
                img_counter += 1
                if img_counter == 10:
                    break
        cam.release()
        cv2.destroyAllWindows()

    def train_model(self):
        if False:
            i = 10
            return i + 15
        if os.path.exists('training_data/face/representations_facenet.pkl'):
            os.remove('training_data/face/representations_facenet.pkl')
        DeepFace.find(img_path='faces/test/Nam.png', db_path='training_data/face', model_name=self.models['default'], distance_metric=self.metrics['default'], detector_backend=self.detectors['default'], enforce_detection=False)
        print('Training completed')

    def run_recognition(self):
        if False:
            return 10
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        prev_frame_time = 0
        count_detect_true = 0
        print('run recognition')
        while True:
            (ret, frame) = cap.read()
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            name_detected = 'Unknown'
            accuracy = 0
            cv2.putText(frame, 'FPS: {:.2f}'.format(fps), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
            net = cv2.dnn.readNetFromCaffe('models/face/deploy.prototxt.txt', 'models/face/res10_300x300_ssd_iter_140000.caffemodel')
            net.setInput(blob)
            detections = net.forward()
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (startX, startY, endX, endY) = box.astype('int')
                    response = DeepFace.find(img_path=frame, db_path=self.db_path, model_name=self.models['default'], distance_metric=self.metrics['default'], silent=True, enforce_detection=False, detector_backend=self.detectors['default'])
                    print('response', response)
                    accuracy = 0
                    df = pd.DataFrame(response[0])
                    print('df.shape', df.shape[0])
                    if df.shape[0] > 0:
                        accuracy = face_confidence(df['Facenet_euclidean'][0])
                        path_name_image = df['identity'][0]
                        (dirpath, filename) = os.path.split(path_name_image)
                        parts = dirpath.split('\\')
                        name_detected = parts[-1]
                        if name_detected == 'Unknown' or accuracy < 50:
                            print('case 1')
                            cv2.rectangle(frame, (startX, startY), (endX, endY), self.color_bgr['red'], 2)
                            cv2.putText(frame, 'Unknown', (startX, endY + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.color_bgr['red'], 2)
                        elif name_detected != 'Unknown' and accuracy > 50:
                            print('case 2')
                            cv2.rectangle(frame, (startX, startY), (endX, endY), self.color_bgr['green'], 2)
                            cv2.putText(frame, name_detected, (startX, endY + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.color_bgr['green'], 2)
                    elif name_detected == 'Unknown' or accuracy < 50:
                        print('case 3')
                        cv2.rectangle(frame, (startX, startY), (endX, endY), self.color_bgr['red'], 2)
                        cv2.putText(frame, 'Unknown', (startX, endY + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.color_bgr['red'], 2)
                    else:
                        print('case 4')
                        cv2.rectangle(frame, (startX, startY), (endX, endY), self.color_bgr['green'], 2)
                    print('NAME_Dectect', name_detected, accuracy)
                    if accuracy > 50:
                        count_detect_true += 1
                    else:
                        count_detect_true = 0
            cv2.imshow('Face', frame)
            if count_detect_true > 5:
                print(name_detected)
                return (True, name_detected)
            if cv2.waitKey(1) & 255 == ord('q'):
                print('Closing program ................')
                break
        cv2.destroyAllWindows()
        cap.release()