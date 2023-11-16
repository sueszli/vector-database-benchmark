import os
import cv2
from deepface.detectors import FaceDetector

def build_model():
    if False:
        print('Hello World!')
    detector = {}
    detector['face_detector'] = build_cascade('haarcascade')
    detector['eye_detector'] = build_cascade('haarcascade_eye')
    return detector

def build_cascade(model_name='haarcascade'):
    if False:
        for i in range(10):
            print('nop')
    opencv_path = get_opencv_path()
    if model_name == 'haarcascade':
        face_detector_path = opencv_path + 'haarcascade_frontalface_default.xml'
        if os.path.isfile(face_detector_path) != True:
            raise ValueError('Confirm that opencv is installed on your environment! Expected path ', face_detector_path, ' violated.')
        detector = cv2.CascadeClassifier(face_detector_path)
    elif model_name == 'haarcascade_eye':
        eye_detector_path = opencv_path + 'haarcascade_eye.xml'
        if os.path.isfile(eye_detector_path) != True:
            raise ValueError('Confirm that opencv is installed on your environment! Expected path ', eye_detector_path, ' violated.')
        detector = cv2.CascadeClassifier(eye_detector_path)
    else:
        raise ValueError(f'unimplemented model_name for build_cascade - {model_name}')
    return detector

def detect_face(detector, img, align=True):
    if False:
        i = 10
        return i + 15
    resp = []
    detected_face = None
    img_region = [0, 0, img.shape[1], img.shape[0]]
    faces = []
    try:
        (faces, _, scores) = detector['face_detector'].detectMultiScale3(img, 1.1, 10, outputRejectLevels=True)
    except:
        pass
    if len(faces) > 0:
        for ((x, y, w, h), confidence) in zip(faces, scores):
            detected_face = img[int(y):int(y + h), int(x):int(x + w)]
            if align:
                detected_face = align_face(detector['eye_detector'], detected_face)
            img_region = [x, y, w, h]
            resp.append((detected_face, img_region, confidence))
    return resp

def align_face(eye_detector, img):
    if False:
        for i in range(10):
            print('nop')
    detected_face_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_detector.detectMultiScale(detected_face_gray, 1.1, 10)
    eyes = sorted(eyes, key=lambda v: abs(v[2] * v[3]), reverse=True)
    if len(eyes) >= 2:
        eye_1 = eyes[0]
        eye_2 = eyes[1]
        if eye_1[0] < eye_2[0]:
            left_eye = eye_1
            right_eye = eye_2
        else:
            left_eye = eye_2
            right_eye = eye_1
        left_eye = (int(left_eye[0] + left_eye[2] / 2), int(left_eye[1] + left_eye[3] / 2))
        right_eye = (int(right_eye[0] + right_eye[2] / 2), int(right_eye[1] + right_eye[3] / 2))
        img = FaceDetector.alignment_procedure(img, left_eye, right_eye)
    return img

def get_opencv_path():
    if False:
        for i in range(10):
            print('nop')
    opencv_home = cv2.__file__
    folders = opencv_home.split(os.path.sep)[0:-1]
    path = folders[0]
    for folder in folders[1:]:
        path = path + '/' + folder
    return path + '/data/'