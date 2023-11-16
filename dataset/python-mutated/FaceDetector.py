import math
from PIL import Image
import numpy as np
from deepface.commons import distance
from deepface.detectors import OpenCvWrapper, SsdWrapper, DlibWrapper, MtcnnWrapper, RetinaFaceWrapper, MediapipeWrapper, YoloWrapper, YunetWrapper, FastMtcnnWrapper

def build_model(detector_backend):
    if False:
        while True:
            i = 10
    global face_detector_obj
    backends = {'opencv': OpenCvWrapper.build_model, 'ssd': SsdWrapper.build_model, 'dlib': DlibWrapper.build_model, 'mtcnn': MtcnnWrapper.build_model, 'retinaface': RetinaFaceWrapper.build_model, 'mediapipe': MediapipeWrapper.build_model, 'yolov8': YoloWrapper.build_model, 'yunet': YunetWrapper.build_model, 'fastmtcnn': FastMtcnnWrapper.build_model}
    if not 'face_detector_obj' in globals():
        face_detector_obj = {}
    built_models = list(face_detector_obj.keys())
    if detector_backend not in built_models:
        face_detector = backends.get(detector_backend)
        if face_detector:
            face_detector = face_detector()
            face_detector_obj[detector_backend] = face_detector
        else:
            raise ValueError('invalid detector_backend passed - ' + detector_backend)
    return face_detector_obj[detector_backend]

def detect_face(face_detector, detector_backend, img, align=True):
    if False:
        for i in range(10):
            print('nop')
    obj = detect_faces(face_detector, detector_backend, img, align)
    if len(obj) > 0:
        (face, region, confidence) = obj[0]
    else:
        face = None
        region = [0, 0, img.shape[1], img.shape[0]]
        confidence = 0
    return (face, region, confidence)

def detect_faces(face_detector, detector_backend, img, align=True):
    if False:
        print('Hello World!')
    backends = {'opencv': OpenCvWrapper.detect_face, 'ssd': SsdWrapper.detect_face, 'dlib': DlibWrapper.detect_face, 'mtcnn': MtcnnWrapper.detect_face, 'retinaface': RetinaFaceWrapper.detect_face, 'mediapipe': MediapipeWrapper.detect_face, 'yolov8': YoloWrapper.detect_face, 'yunet': YunetWrapper.detect_face, 'fastmtcnn': FastMtcnnWrapper.detect_face}
    detect_face_fn = backends.get(detector_backend)
    if detect_face_fn:
        obj = detect_face_fn(face_detector, img, align)
        return obj
    else:
        raise ValueError('invalid detector_backend passed - ' + detector_backend)

def alignment_procedure(img, left_eye, right_eye):
    if False:
        for i in range(10):
            print('nop')
    (left_eye_x, left_eye_y) = left_eye
    (right_eye_x, right_eye_y) = right_eye
    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1
    a = distance.findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
    b = distance.findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
    c = distance.findEuclideanDistance(np.array(right_eye), np.array(left_eye))
    if b != 0 and c != 0:
        cos_a = (b * b + c * c - a * a) / (2 * b * c)
        angle = np.arccos(cos_a)
        angle = angle * 180 / math.pi
        if direction == -1:
            angle = 90 - angle
        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))
    return img