from deepface.detectors import FaceDetector

def build_model():
    if False:
        while True:
            i = 10
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)
    return face_detection

def detect_face(face_detector, img, align=True):
    if False:
        return 10
    resp = []
    img_width = img.shape[1]
    img_height = img.shape[0]
    results = face_detector.process(img)
    if results.detections is None:
        return resp
    for detection in results.detections:
        (confidence,) = detection.score
        bounding_box = detection.location_data.relative_bounding_box
        landmarks = detection.location_data.relative_keypoints
        x = int(bounding_box.xmin * img_width)
        w = int(bounding_box.width * img_width)
        y = int(bounding_box.ymin * img_height)
        h = int(bounding_box.height * img_height)
        left_eye = (int(landmarks[0].x * img_width), int(landmarks[0].y * img_height))
        right_eye = (int(landmarks[1].x * img_width), int(landmarks[1].y * img_height))
        if x > 0 and y > 0:
            detected_face = img[y:y + h, x:x + w]
            img_region = [x, y, w, h]
            if align:
                detected_face = FaceDetector.alignment_procedure(detected_face, left_eye, right_eye)
            resp.append((detected_face, img_region, confidence))
    return resp