from deepface.detectors import FaceDetector
PATH = '/.deepface/weights/yolov8n-face.pt'
WEIGHT_URL = 'https://drive.google.com/uc?id=1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb'
LANDMARKS_CONFIDENCE_THRESHOLD = 0.5

def build_model():
    if False:
        i = 10
        return i + 15
    'Build YOLO (yolov8n-face) model'
    import gdown
    import os
    from ultralytics import YOLO
    from deepface.commons.functions import get_deepface_home
    weight_path = f'{get_deepface_home()}{PATH}'
    if not os.path.isfile(weight_path):
        gdown.download(WEIGHT_URL, weight_path, quiet=False)
        print(f'Downloaded YOLO model {os.path.basename(weight_path)}')
    return YOLO(weight_path)

def detect_face(face_detector, img, align=False):
    if False:
        return 10
    resp = []
    results = face_detector.predict(img, verbose=False, show=False, conf=0.25)[0]
    for result in results:
        (x, y, w, h) = result.boxes.xywh.tolist()[0]
        confidence = result.boxes.conf.tolist()[0]
        (x, y, w, h) = (int(x - w / 2), int(y - h / 2), int(w), int(h))
        detected_face = img[y:y + h, x:x + w].copy()
        if align:
            left_eye = (result.keypoints.xy[0][0], result.keypoints.conf[0][0])
            right_eye = (result.keypoints.xy[0][1], result.keypoints.conf[0][1])
            if left_eye[1] > LANDMARKS_CONFIDENCE_THRESHOLD and right_eye[1] > LANDMARKS_CONFIDENCE_THRESHOLD:
                detected_face = FaceDetector.alignment_procedure(detected_face, left_eye[0].cpu(), right_eye[0].cpu())
        resp.append((detected_face, [x, y, w, h], confidence))
    return resp