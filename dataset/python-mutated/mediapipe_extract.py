import cv2
import os, sys, gc
import time
import numpy as np
import mediapipe as mp
from tqdm.auto import tqdm
import multiprocessing
from joblib import Parallel, delayed
from natsort import natsorted
from glob import glob
import math
import pickle
mp_holistic = mp.solutions.holistic
N_FACE_LANDMARKS = 468
N_BODY_LANDMARKS = 33
N_HAND_LANDMARKS = 21

class Counter(object):

    def __init__(self, initval=0):
        if False:
            for i in range(10):
                print('nop')
        self.val = multiprocessing.RawValue('i', initval)
        self.lock = multiprocessing.Lock()

    def increment(self):
        if False:
            while True:
                i = 10
        with self.lock:
            self.val.value += 1

    @property
    def value(self):
        if False:
            i = 10
            return i + 15
        return self.val.value

def process_body_landmarks(component, n_points):
    if False:
        while True:
            i = 10
    kps = np.zeros((n_points, 3))
    conf = np.zeros(n_points)
    if component is not None:
        landmarks = component.landmark
        kps = np.array([[p.x, p.y, p.z] for p in landmarks])
        conf = np.array([p.visibility for p in landmarks])
    return (kps, conf)

def process_other_landmarks(component, n_points):
    if False:
        i = 10
        return i + 15
    kps = np.zeros((n_points, 3))
    conf = np.zeros(n_points)
    if component is not None:
        landmarks = component.landmark
        kps = np.array([[p.x, p.y, p.z] for p in landmarks])
        conf = np.ones(n_points)
    return (kps, conf)

def get_holistic_keypoints(frames, holistic=mp_holistic.Holistic(static_image_mode=False, model_complexity=2)):
    if False:
        return 10
    "\n    For videos, it's optimal to create with `static_image_mode=False` for each video.\n    https://google.github.io/mediapipe/solutions/holistic.html#static_image_mode\n    "
    keypoints = []
    confs = []
    for frame in frames:
        results = holistic.process(frame)
        (body_data, body_conf) = process_body_landmarks(results.pose_landmarks, N_BODY_LANDMARKS)
        (face_data, face_conf) = process_other_landmarks(results.face_landmarks, N_FACE_LANDMARKS)
        (lh_data, lh_conf) = process_other_landmarks(results.left_hand_landmarks, N_HAND_LANDMARKS)
        (rh_data, rh_conf) = process_other_landmarks(results.right_hand_landmarks, N_HAND_LANDMARKS)
        data = np.concatenate([body_data, face_data, lh_data, rh_data])
        conf = np.concatenate([body_conf, face_conf, lh_conf, rh_conf])
        keypoints.append(data)
        confs.append(conf)
    holistic.close()
    del holistic
    gc.collect()
    keypoints = np.stack(keypoints)
    confs = np.stack(confs)
    return (keypoints, confs)

def gen_keypoints_for_frames(frames, save_path):
    if False:
        for i in range(10):
            print('nop')
    (pose_kps, pose_confs) = get_holistic_keypoints(frames)
    body_kps = np.concatenate([pose_kps[:, :33, :], pose_kps[:, 501:, :]], axis=1)
    confs = np.concatenate([pose_confs[:, :33], pose_confs[:, 501:]], axis=1)
    d = {'keypoints': body_kps, 'confidences': confs}
    with open(save_path + '.pkl', 'wb') as f:
        pickle.dump(d, f, protocol=4)

def load_frames_from_video(video_path):
    if False:
        for i in range(10):
            print('nop')
    frames = []
    vidcap = cv2.VideoCapture(video_path)
    while vidcap.isOpened():
        (success, img) = vidcap.read()
        if not success:
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
    vidcap.release()
    return np.asarray(frames)

def load_frames_from_folder(frames_folder, patterns=['*.jpg']):
    if False:
        return 10
    images = []
    for pattern in patterns:
        images.extend(glob(f'{frames_folder}/{pattern}'))
    images = natsorted(list(set(images)))
    if not images:
        exit(f'ERROR: No frames in folder: {frames_folder}')
    frames = []
    for img_path in images:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
    return np.asarray(frames)

def gen_keypoints_for_video(video_path, save_path):
    if False:
        i = 10
        return i + 15
    if not os.path.isfile(video_path):
        print('SKIPPING MISSING FILE:', video_path)
        return
    frames = load_frames_from_video(video_path)
    gen_keypoints_for_frames(frames, save_path)

def gen_keypoints_for_folder(folder, save_path, file_patterns):
    if False:
        while True:
            i = 10
    frames = load_frames_from_folder(folder, file_patterns)
    gen_keypoints_for_frames(frames, save_path)

def generate_pose(dataset, save_folder, worker_index, num_workers, counter):
    if False:
        while True:
            i = 10
    num_splits = math.ceil(len(dataset) / num_workers)
    end_index = min((worker_index + 1) * num_splits, len(dataset))
    for index in range(worker_index * num_splits, end_index):
        (imgs, label, video_id) = dataset.read_data(index)
        save_path = os.path.join(save_folder, video_id)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        gen_keypoints_for_frames(imgs, save_path)
        counter.increment()

def dump_pose_for_dataset(dataset, save_folder, num_workers=multiprocessing.cpu_count()):
    if False:
        i = 10
        return i + 15
    os.makedirs(save_folder, exist_ok=True)
    processes = []
    counter = Counter()
    for i in tqdm(range(num_workers), desc='Creating sub-processes...'):
        p = multiprocessing.Process(target=generate_pose, args=(dataset, save_folder, i, num_workers, counter))
        p.start()
        processes.append(p)
    total_samples = len(dataset)
    with tqdm(total=total_samples) as pbar:
        while counter.value < total_samples:
            pbar.update(counter.value - pbar.n)
            time.sleep(2)
    for i in range(num_workers):
        processes[i].join()
    print(f'Pose data successfully saved to: {save_folder}')

def extraction(file_name):
    if False:
        for i in range(10):
            print('nop')
    n_cores = multiprocessing.cpu_count()
    switch_mode_flag = '/home/tester/finalProject/switch_mode_flag.txt'
    with open(switch_mode_flag, 'r') as file:
        mode_flag = file.read()
    if mode_flag == '1':
        print('MODE 2 EXTRACTION UP')
        DIR = '/home/tester/finalProject/mode2_videos'
        SAVE_DIR = '/home/tester/finalProject/videos_after'
        SAVE_DIR_DESKTOP = '/home/tester/Desktop/videos'
        os.makedirs(SAVE_DIR, exist_ok=True)
        file_paths = []
        save_paths = []
        for file in os.listdir(DIR):
            if file_name in file:
                file_paths.append(os.path.join(DIR, file))
                save_paths.append(os.path.join(SAVE_DIR, file.replace('.mp4', '')))
        Parallel(n_jobs=n_cores, backend='threading')((delayed(gen_keypoints_for_video)(path, save_path) for (path, save_path) in tqdm(zip(file_paths, save_paths))))
    else:
        print('MODE 1 EXTRACTION UP')
        DIR = '/home/tester/finalProject/videos'
        SAVE_DIR = '/home/tester/finalProject/videos_after'
        SAVE_DIR_DESKTOP = '/home/tester/Desktop/videos'
        os.makedirs(SAVE_DIR, exist_ok=True)
        file_paths = []
        save_paths = []
        for file in os.listdir(DIR):
            if file_name in file:
                file_paths.append(os.path.join(DIR, file))
                save_paths.append(os.path.join(SAVE_DIR, file.replace('.mp4', '')))
        Parallel(n_jobs=n_cores, backend='threading')((delayed(gen_keypoints_for_video)(path, save_path) for (path, save_path) in tqdm(zip(file_paths, save_paths))))