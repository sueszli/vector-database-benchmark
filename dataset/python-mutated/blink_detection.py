import face_recognition
import cv2
import time
from scipy.spatial import distance as dist
EYES_CLOSED_SECONDS = 5

def main():
    if False:
        print('Hello World!')
    closed_count = 0
    video_capture = cv2.VideoCapture(0)
    (ret, frame) = video_capture.read(0)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)
    process = True
    while True:
        (ret, frame) = video_capture.read(0)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        if process:
            face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)
            for face_landmark in face_landmarks_list:
                left_eye = face_landmark['left_eye']
                right_eye = face_landmark['right_eye']
                color = (255, 0, 0)
                thickness = 2
                cv2.rectangle(small_frame, left_eye[0], right_eye[-1], color, thickness)
                cv2.imshow('Video', small_frame)
                ear_left = get_ear(left_eye)
                ear_right = get_ear(right_eye)
                closed = ear_left < 0.2 and ear_right < 0.2
                if closed:
                    closed_count += 1
                else:
                    closed_count = 0
                if closed_count >= EYES_CLOSED_SECONDS:
                    asleep = True
                    while asleep:
                        print('EYES CLOSED')
                        if cv2.waitKey(1) == 32:
                            asleep = False
                            print('EYES OPENED')
                    closed_count = 0
        process = not process
        key = cv2.waitKey(1) & 255
        if key == ord('q'):
            break

def get_ear(eye):
    if False:
        for i in range(10):
            print('nop')
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
if __name__ == '__main__':
    main()