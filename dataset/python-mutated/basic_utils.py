import os
import pickle
import random
import shutil
import zipfile
from collections import OrderedDict
import cv2
import numpy as np
import torch
import ujson as json
from PIL import Image
from torchvision import transforms

def init_transform_dict(input_res=224):
    if False:
        while True:
            i = 10
    '\n        The implementation of transforms functions.\n        The default image resolution is 224.\n        The normalize parameter follows the mainstream setting.\n    '
    tsfm_dict = {'clip_test': transforms.Compose([transforms.Resize(input_res, interpolation=Image.BICUBIC), transforms.CenterCrop(input_res), transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))]), 'clip_train': transforms.Compose([transforms.RandomResizedCrop(input_res, scale=(0.5, 1.0)), transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0, saturation=0, hue=0), transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])}
    return tsfm_dict

def load_data(feature_path, mydevice):
    if False:
        for i in range(10):
            print('nop')
    "\n        Loading dataset from 'feature_path' as a retrieval docs.\n        The default dataset is MSRVTT-9K.\n\n        Args:\n            feature_path: 'VoP_msrvtt9k_features.pkl'\n            mydevice: device(type='cuda', index=0)\n\n        Returns:\n            [text_embeds, vid_embeds_pooled, vid_ids, texts]\n    "
    feature_content = torch.load(feature_path)
    text_embeds = feature_content['text_embeds'].to(device=mydevice)
    vid_embeds_pooled = feature_content['vid_embeds'].to(device=mydevice)
    vid_ids = feature_content['vid_ids']
    texts = feature_content['texts']
    return [text_embeds, vid_embeds_pooled, vid_ids, texts]

def load_json(filename):
    if False:
        i = 10
        return i + 15
    '\n        Load json files.\n    '
    with open(filename, 'r') as f:
        return json.load(f)

def set_seed(seed):
    if False:
        i = 10
        return i + 15
    '\n        Set random seed.\n    '
    if seed >= 0:
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_state_dict(checkpoint_path):
    if False:
        while True:
            i = 10
    '\n        Load pre-train parameters for VoP.\n    '
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for (k, v) in state_dict.items():
        new_state_dict[k.replace('module.', '')] = v
    state_dict = new_state_dict
    return state_dict

def get_valid_frames(cap, num_frames, vlen, sample='rand'):
    if False:
        i = 10
        return i + 15
    "\n        Get indexes of sampled frames.\n\n        Args:\n            cap: cv2.VideoCapture\n            num_frames: int - number of frames to sample\n            vlen: video length, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 325\n            sample: 'rand' | 'uniform' how to sample\n\n        Returns:\n            frames: torch.tensor of stacked sampled video frames\n                    of dim (num_frames, C, H, W)\n            frame_idxs: list(int) indices of where the frames where sampled\n    "
    acc_samples = min(num_frames, vlen)
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    ranges = []
    for (idx, interv) in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    if sample == 'rand':
        frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
    else:
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    frames = []
    for index in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        (ret, frame) = cap.read()
        if not ret:
            n_tries = 5
            for _ in range(n_tries):
                (ret, frame) = cap.read()
                if ret:
                    break
        if not ret:
            return (None, None)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame)
        frame = frame.permute(2, 0, 1)
        frames.append(frame)
    while len(frames) < num_frames:
        frames.append(frames[-1].clone())
    return (frames, frame_idxs)

def load_frames_from_video(video_path, num_frames, sample='rand'):
    if False:
        i = 10
        return i + 15
    "\n        Get indexes of sampled frames.\n\n        Args:\n            video_path: the local video path\n            num_frames: Frame number, 12 frames for each video\n            sample: 'rand' | 'uniform' how to sample\n\n        Returns:\n            frames: torch.tensor of stacked sampled video frames\n                    of dim (num_frames, C, H, W)\n            frame_idxs: list(int) indices of where the frames where sampled\n    "
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), video_path
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    (frames, frame_idxs) = get_valid_frames(cap, num_frames, vlen, sample)
    frames = torch.stack(frames).float() / 255
    cap.release()
    return (frames, frame_idxs)