import sys
import dlib

def show_jittered_images(window, jittered_images):
    if False:
        i = 10
        return i + 15
    '\n        Shows the specified jittered images one by one\n    '
    for img in jittered_images:
        window.set_image(img)
        dlib.hit_enter_to_continue()
if len(sys.argv) != 2:
    print('Call this program like this:\n   ./face_jitter.py shape_predictor_5_face_landmarks.dat\nYou can download a trained facial shape predictor from:\n    http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2\n')
    exit()
predictor_path = sys.argv[1]
face_file_path = '../examples/faces/Tom_Cruise_avp_2014_4.jpg'
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
img = dlib.load_rgb_image(face_file_path)
dets = detector(img)
num_faces = len(dets)
faces = dlib.full_object_detections()
for detection in dets:
    faces.append(sp(img, detection))
image = dlib.get_face_chip(img, faces[0], size=320)
window = dlib.image_window()
window.set_image(image)
dlib.hit_enter_to_continue()
jittered_images = dlib.jitter_image(image, num_jitters=5)
show_jittered_images(window, jittered_images)
jittered_images = dlib.jitter_image(image, num_jitters=5, disturb_colors=True)
show_jittered_images(window, jittered_images)