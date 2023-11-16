import face_recognition
import cv2
from multiprocessing import Process, Manager, cpu_count, set_start_method
import time
import numpy
import threading
import platform

def next_id(current_id, worker_num):
    if False:
        for i in range(10):
            print('nop')
    if current_id == worker_num:
        return 1
    else:
        return current_id + 1

def prev_id(current_id, worker_num):
    if False:
        while True:
            i = 10
    if current_id == 1:
        return worker_num
    else:
        return current_id - 1

def capture(read_frame_list, Global, worker_num):
    if False:
        print('Hello World!')
    video_capture = cv2.VideoCapture(0)
    print('Width: %d, Height: %d, FPS: %d' % (video_capture.get(3), video_capture.get(4), video_capture.get(5)))
    while not Global.is_exit:
        if Global.buff_num != next_id(Global.read_num, worker_num):
            (ret, frame) = video_capture.read()
            read_frame_list[Global.buff_num] = frame
            Global.buff_num = next_id(Global.buff_num, worker_num)
        else:
            time.sleep(0.01)
    video_capture.release()

def process(worker_id, read_frame_list, write_frame_list, Global, worker_num):
    if False:
        print('Hello World!')
    known_face_encodings = Global.known_face_encodings
    known_face_names = Global.known_face_names
    while not Global.is_exit:
        while Global.read_num != worker_id or Global.read_num != prev_id(Global.buff_num, worker_num):
            if Global.is_exit:
                break
            time.sleep(0.01)
        time.sleep(Global.frame_delay)
        frame_process = read_frame_list[worker_id]
        Global.read_num = next_id(Global.read_num, worker_num)
        rgb_frame = frame_process[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        for ((top, right, bottom, left), face_encoding) in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = 'Unknown'
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            cv2.rectangle(frame_process, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame_process, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame_process, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        while Global.write_num != worker_id:
            time.sleep(0.01)
        write_frame_list[worker_id] = frame_process
        Global.write_num = next_id(Global.write_num, worker_num)
if __name__ == '__main__':
    if platform.system() == 'Darwin':
        set_start_method('forkserver')
    Global = Manager().Namespace()
    Global.buff_num = 1
    Global.read_num = 1
    Global.write_num = 1
    Global.frame_delay = 0
    Global.is_exit = False
    read_frame_list = Manager().dict()
    write_frame_list = Manager().dict()
    if cpu_count() > 2:
        worker_num = cpu_count() - 1
    else:
        worker_num = 2
    p = []
    p.append(threading.Thread(target=capture, args=(read_frame_list, Global, worker_num)))
    p[0].start()
    obama_image = face_recognition.load_image_file('obama.jpg')
    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
    biden_image = face_recognition.load_image_file('biden.jpg')
    biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
    Global.known_face_encodings = [obama_face_encoding, biden_face_encoding]
    Global.known_face_names = ['Barack Obama', 'Joe Biden']
    for worker_id in range(1, worker_num + 1):
        p.append(Process(target=process, args=(worker_id, read_frame_list, write_frame_list, Global, worker_num)))
        p[worker_id].start()
    last_num = 1
    fps_list = []
    tmp_time = time.time()
    while not Global.is_exit:
        while Global.write_num != last_num:
            last_num = int(Global.write_num)
            delay = time.time() - tmp_time
            tmp_time = time.time()
            fps_list.append(delay)
            if len(fps_list) > 5 * worker_num:
                fps_list.pop(0)
            fps = len(fps_list) / numpy.sum(fps_list)
            print('fps: %.2f' % fps)
            if fps < 6:
                Global.frame_delay = 1 / fps * 0.75
            elif fps < 20:
                Global.frame_delay = 1 / fps * 0.5
            elif fps < 30:
                Global.frame_delay = 1 / fps * 0.25
            else:
                Global.frame_delay = 0
            cv2.imshow('Video', write_frame_list[prev_id(Global.write_num, worker_num)])
        if cv2.waitKey(1) & 255 == ord('q'):
            Global.is_exit = True
            break
        time.sleep(0.01)
    cv2.destroyAllWindows()