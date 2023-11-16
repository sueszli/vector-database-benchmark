import numpy as np
import torch
from moviepy.editor import VideoFileClip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os
import re
'\nModality (01 = full-AV, 02 = video-only, 03 = audio-only).\nVocal channel (01 = speech, 02 = song).\nEmotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).\nEmotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the \'neutral\' emotion.\nStatement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").\nRepetition (01 = 1st repetition, 02 = 2nd repetition).\n    Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).\n'

def transform(img):
    if False:
        return 10
    return np.asarray(Image.fromarray(img).crop((256, 0, 1024, 720)).resize((192, 180)))

def clean_video_list(video_path):
    if False:
        i = 10
        return i + 15
    video_file_list = os.listdir(video_path)
    pattern = '01.+'
    return [x for (i, x) in enumerate(video_file_list) if re.match(pattern, x)]

def get_samples(sample_name):
    if False:
        for i in range(10):
            print('nop')
    clip = VideoFileClip(sample_name)
    audio = np.array(list(clip.audio.set_fps(16000).iter_frames())).mean(1)
    count = 0
    audio_sample_tuple = []
    frame_sample_tuple = []
    label_tuple = []
    sub_clip_frame = []
    sub_clip_audio = []
    frame_count = 0
    capture_size = True
    for frame in clip.iter_frames():
        frame_count += 1
        frame = transform(frame)
        frame = frame / 255.0
        frame_audio = audio[count * 16000 // 30:count * 16000 // 30 + 533]
        pattern = '.+?-.+?-(.+?)-(.+?)-'
        label = re.match(pattern, sample_name).groups()
        sub_clip_audio.append(frame_audio)
        sub_clip_frame.append(np.moveaxis(np.moveaxis(frame, -1, 0), -1, -2))
        if frame_count == 15:
            frame_count = 0
            if capture_size:
                audio_size = np.size(sub_clip_audio)
                capture_size = False
            if not capture_size and np.size(sub_clip_audio) != audio_size:
                break
            audio_sample_tuple.append(sub_clip_audio)
            frame_sample_tuple.append(sub_clip_frame)
            label_tuple.append(np.float64(label[0]))
            sub_clip_frame = []
            sub_clip_audio = []
        count += 1
    return (np.array(audio_sample_tuple), np.array(frame_sample_tuple), np.array([i - 1 for i in label_tuple]))

class FrameDataSet(Dataset):

    def __init__(self, config, status='train'):
        if False:
            while True:
                i = 10
        self.directory = config['directory']
        pattern = 'Video_Speech_Actor_.+'
        self.dir_list = [i for i in os.listdir(self.directory) if re.match(pattern, i)]
        self.full_file_name_list = []
        self.full_data_set_audio = torch.tensor([], dtype=torch.float64)
        self.full_data_set_frame = torch.tensor([], dtype=torch.float64)
        self.full_label_set = torch.tensor([], dtype=torch.float64)
        for i in self.dir_list:
            current_list = clean_video_list(self.directory + i + '/Actor_' + i[-2:] + '/')
            for j in current_list:
                self.full_file_name_list.append(self.directory + i + '/Actor_' + i[-2:] + '/' + j)
        portion = round(config['portion'] * round(len(self.full_file_name_list)))
        if status == 'train':
            self.full_file_name_list = self.full_file_name_list[0:portion]
        elif status == 'eval':
            self.full_file_name_list = self.full_file_name_list[portion:]
        else:
            print('??????????????????WTF/????????????')

    def __getitem__(self, index):
        if False:
            return 10
        return self.full_file_name_list[index]

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self.full_file_name_list)