import ast
import os
import subprocess
from collections import defaultdict
import random
import numpy as np
import pandas as pd

def video_format_conversion(video_path, output_path, h264_format=False):
    if False:
        i = 10
        return i + 15
    '\n    Encode video in a different format.\n\n    :param video_path: str.\n        Path to input video\n    :param output_path: str.\n        Path where converted video will be written to.\n    :param h264_format: boolean.\n        Set to true to save time if input is in h264_format.\n    :return: None.\n    '
    if not h264_format:
        subprocess.run(['ffmpeg', '-i', video_path, '-c', 'copy', '-map', '0', output_path])
    else:
        subprocess.run(['ffmpeg', '-i', video_path, '-vcodec', 'libx264', output_path])

def parse_video_file_name(row):
    if False:
        print('Hello World!')
    '\n    Extract file basename from file path\n\n    :param row: Pandas.Series.\n        One row of the video annotation output from the VIA tool.\n    :return: str.\n        The file basename\n    '
    video_file = ast.literal_eval(row.file_list)[0]
    return os.path.basename(video_file).replace('%20', ' ')

def read_classes_file(classes_filepath):
    if False:
        print('Hello World!')
    '\n    Read file that maps class names to class IDs. The file should be in the format:\n        ActionName1 0\n        ActionName2 1\n\n    :param classes_filepath: str\n        The filepath of the classes file\n    :return: dict\n        Mapping of class names to class IDs\n    '
    classes = {}
    with open(classes_filepath) as class_file:
        for line in class_file:
            (class_name, class_id) = line.split(' ')
            classes[class_name] = class_id.rstrip()
    return classes

def create_clip_file_name(row, clip_file_format='mp4'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Create the output clip file name.\n\n    :param row: pandas.Series.\n        One row of the video annotation output from the VIA tool.\n        This function requires the output from VIA tool contains a column '# CSV_HEADER = metadata_id'.\n    :param clip_file_format: str.\n        The format of the output clip file.\n    :return: str.\n        The output clip file name.\n    "
    video_file = os.path.splitext(row['file_list'])[0]
    clip_id = row['# CSV_HEADER = metadata_id']
    clip_file = '{}_{}.{}'.format(video_file, clip_id, clip_file_format)
    return clip_file

def get_clip_action_label(row):
    if False:
        i = 10
        return i + 15
    "\n    Get the action label of the positive clips.\n    This function requires the output from VIA tool contains a column 'metadata'.\n\n    :param row: pandas.Series.\n        One row of the video annotation output.\n    :return: str.\n    "
    label_dict = ast.literal_eval(row.metadata)
    track_key = list(label_dict.keys())[0]
    return label_dict[track_key]

def _extract_clip_ffmpeg(start_time, duration, video_path, clip_path, ffmpeg_path=None):
    if False:
        return 10
    '\n    Using ffmpeg to extract clip from the video based on the start time and duration of the clip.\n\n    :param start_time: float.\n        The start time of the clip.\n    :param duration: float.\n        The duration of the clip.\n    :param video_path: str.\n        The path of the input video.\n    :param clip_path: str.\n        The path of the output clip.\n    :param ffmpeg_path: str.\n        The path of the ffmpeg. This is optional, which you could use when you have not added the\n        ffmpeg to the path environment variable.\n    :return: None.\n    '
    subprocess.run([os.path.join(ffmpeg_path, 'ffmpeg') if ffmpeg_path is not None else 'ffmpeg', '-ss', str(start_time), '-i', video_path, '-t', str(duration), clip_path, '-codec', 'copy', '-y'])

def extract_clip(row, video_dir, clip_dir, ffmpeg_path=None):
    if False:
        i = 10
        return i + 15
    '\n    Extract the postivie clip based on a row of the output annotation file.\n\n    :param row: pandas.Series.\n        One row of the video annotation output.\n    :param video_dir: str.\n        The directory of the input videos.\n    :param clip_dir: str.\n        The directory of the output positive clips.\n    :param ffmpeg_path: str.\n        The path of the ffmpeg. This is optional, which you could use when you have not added the\n        ffmpeg to the path environment variable.\n    :return: None.\n    '
    if not os.path.exists(clip_dir):
        os.makedirs(clip_dir)
    if 'temporal_segment_start' in row.index:
        start_time = row.temporal_segment_start
        if 'temporal_segment_end' not in row.index:
            raise ValueError("There is no column named 'temporal_segment_end'. Cannot get the full details of the action temporal intervals.")
        end_time = row.temporal_segment_end
    elif 'temporal_coordinates' in row.index:
        (start_time, end_time) = ast.literal_eval(row.temporal_coordinates)
    else:
        raise Exception('There is no temporal information in the csv.')
    clip_sub_dir = os.path.join(clip_dir, row.clip_action_label)
    if not os.path.exists(clip_sub_dir):
        os.makedirs(clip_sub_dir)
    duration = end_time - start_time
    video_file = row.file_list
    video_path = os.path.join(video_dir, video_file)
    clip_file = row.clip_file_name
    clip_path = os.path.join(clip_sub_dir, clip_file)
    if os.path.exists(clip_path):
        print('Extracted clip already exists. Skipping extraction.')
        return
    if not os.path.exists(video_path):
        raise ValueError("The video path '{}' is not valid.".format(video_path))
    _extract_clip_ffmpeg(start_time, duration, video_path, clip_path, ffmpeg_path)

def get_video_length(video_file_path):
    if False:
        return 10
    '\n    Get the video length in milliseconds.\n\n    :param video_file_path: str.\n        The path of the video file.\n    :return: (str, str).\n        Tuple of video duration (in string), and error message of the ffprobe command if any.\n    '
    cmd_list = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_file_path]
    result = subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if len(result.stderr) > 0:
        raise RuntimeError(result.stderr)
    return float(result.stdout)

def check_interval_overlaps(clip_start, clip_end, interval_list):
    if False:
        i = 10
        return i + 15
    '\n    Check whether a clip overlaps any intervals from a list of intervals.\n\n    param clip_start: float\n        Time in seconds of the start of the clip.\n    param clip_end: float\n        Time in seconds of the end of the clip.\n    param interval_list: list of tuples (float, float)\n        List of time intervals\n    return: Boolean\n        True if the clip overlaps any of the intervals in interval list.\n    '
    overlapping = False
    for interval in interval_list:
        if clip_start < interval[1] and clip_end > interval[0]:
            overlapping = True
    return overlapping

def _merge_temporal_interval(temporal_interval_list):
    if False:
        for i in range(10):
            print('nop')
    '\n    Merge the temporal intervals in the input temporal interval list. This is for situations\n    when different actions have overlap temporal interval. e.g if the input temporal interval list\n    is [(1.0, 3.0), (2.0, 4.0)], then [(1.0, 4.0)] will be returned.\n\n    :param temporal_interval_list: list of tuples.\n        List of tuples with (temporal interval start time, temporal interval end time).\n    :return: list of tuples.\n        The merged temporal interval list.\n    '
    temporal_interval_list_sorted = sorted(temporal_interval_list, key=lambda x: x[0])
    i = 0
    while i < len(temporal_interval_list_sorted) - 1:
        (a1, b1) = temporal_interval_list_sorted[i]
        (a2, b2) = temporal_interval_list_sorted[i + 1]
        if a2 <= b1:
            del temporal_interval_list_sorted[i]
            temporal_interval_list_sorted[i] = [a1, max(b1, b2)]
        else:
            i += 1
    return temporal_interval_list_sorted

def _split_interval(interval, left_ignore_clip_length, right_ignore_clip_length, clip_length, skip_clip_length=0):
    if False:
        print('Hello World!')
    '\n    Split the negative sample interval into the sub-intervals which will serve as the start and end of\n    the negative sample clips.\n\n    :param interval: tuple of (float, float).\n        Tuple of start and end of the negative sample interval.\n    :param left_ignore_clip_length: float.\n        The clip length to ignore in the left/start of the interval. This is used to avoid creating\n        negative sample clips with edges too close to positive samples. The same applies to right_ignore_clip_length.\n    :param right_ignore_clip_length: float.\n        The clip length to ignore in the right/end of the interval.\n    :param clip_length: float.\n        The clip length of the created negative clips.\n    :param skip_clip_length: float.\n        The skipped video length between two negative samples, this can be used to reduce the\n        number of the negative samples.\n    :return: list of tuples.\n        List of start and end time of the negative clips.\n    '
    (left, right) = interval
    if left_ignore_clip_length + right_ignore_clip_length >= right - left:
        return []
    new_left = left + left_ignore_clip_length
    new_right = right - right_ignore_clip_length
    if new_right - new_left < clip_length:
        return []
    interval_start_list = np.arange(new_left, new_right, clip_length + skip_clip_length)
    interval_end_list = interval_start_list + clip_length
    if interval_end_list[-1] > new_right:
        interval_start_list = interval_start_list[:-1]
        interval_end_list = interval_end_list[:-1]
    res = list(zip(list(interval_start_list), list(interval_end_list)))
    return res

def _split_interval_list(interval_list, left_ignore_clip_length, right_ignore_clip_length, clip_length, skip_clip_length=0):
    if False:
        while True:
            i = 10
    '\n    Taking the interval list of the eligible negative sample time intervals, return the list of the\n    start time and the end time of the negative clips.\n\n    :param interval_list: list of tuples.\n        List of the tuples containing the start time and end time of the eligible negative\n        sample time intervals.\n    :param left_ignore_clip_length: float.\n        See split_interval.\n    :param right_ignore_clip_length: float.\n        See split_interval.\n    :param clip_length: float.\n        See split_interval.\n    :param skip_clip_length: float.\n        See split_interval\n    :return: list of tuples.\n        List of start and end time of the negative clips.\n    '
    interval_res = []
    for i in range(len(interval_list)):
        interval_res += _split_interval(interval_list[i], left_ignore_clip_length=left_ignore_clip_length, right_ignore_clip_length=right_ignore_clip_length, clip_length=clip_length, skip_clip_length=skip_clip_length)
    return interval_res

def extract_contiguous_negative_clips(video_file, video_dir, video_info_df, negative_clip_dir, clip_format, no_action_class, ignore_clip_length, clip_length, skip_clip_length=0, ffmpeg_path=None):
    if False:
        return 10
    '\n    Extract the negative sample for a single video file.\n\n    :param video_file: str.\n        The name of the input video file.\n    :param video_dir: str.\n        The directory of the input video.\n    :param video_info_df: pandas.DataFrame.\n        The data frame which contains the video annotation output.\n    :param negative_clip_dir: str.\n        The directory of the output negative clips.\n    :param clip_format: str.\n        The format of the output negative clips.\n    :param ignore_clip_length: float.\n        The clip length to ignore in the left/start of the interval. This is used to avoid creating\n        negative sample clips with edges too close to positive samples.\n    :param clip_length: float.\n        The clip length of the created negative clips.\n    :param ffmpeg_path: str.\n        The path of the ffmpeg. This is optional, which you could use when you have not added the\n        ffmpeg to the path environment variable.\n    :param skip_clip_length: float.\n        The skipped video length between two negative samples, this can be used to reduce the\n        number of the negative samples.\n    :return: pandas.DataFrame.\n        The data frame which contains start and end time of the negative clips.\n    '
    video_file_path = os.path.join(video_dir, video_file)
    video_duration = get_video_length(video_file_path)
    if 'temporal_coordinates' in video_info_df.columns:
        temporal_interval_series = video_info_df.loc[video_info_df['file_list'] == video_file, 'temporal_coordinates']
        temporal_interval_list = temporal_interval_series.apply(lambda x: ast.literal_eval(x)).tolist()
    elif 'temporal_segment_start' in video_info_df.columns:
        video_start_list = video_info_df.loc[video_info_df['file_list'] == video_file, 'temporal_segment_start'].tolist()
        video_end_list = video_info_df.loc[video_info_df['file_list'] == video_file, 'temporal_segment_end'].tolist()
        temporal_interval_list = list(zip(video_start_list, video_end_list))
    else:
        raise Exception('There is no temporal information in the csv.')
    if not all((len(temporal_interval) % 2 == 0 for temporal_interval in temporal_interval_list)):
        raise ValueError('There is at least one time interval in {} having only one end point.'.format(str(temporal_interval_list)))
    temporal_interval_list = _merge_temporal_interval(temporal_interval_list)
    negative_sample_interval_list = [0.0] + [t for interval in temporal_interval_list for t in interval] + [video_duration]
    negative_sample_interval_list = [[negative_sample_interval_list[2 * i], negative_sample_interval_list[2 * i + 1]] for i in range(len(negative_sample_interval_list) // 2)]
    clip_interval_list = _split_interval_list(negative_sample_interval_list, left_ignore_clip_length=ignore_clip_length, right_ignore_clip_length=ignore_clip_length, clip_length=clip_length, skip_clip_length=skip_clip_length)
    if not os.path.exists(negative_clip_dir):
        os.makedirs(negative_clip_dir)
    negative_clip_file_list = []
    for (i, clip_interval) in enumerate(clip_interval_list):
        start_time = clip_interval[0]
        duration = clip_interval[1] - clip_interval[0]
        video_fname = os.path.splitext(os.path.basename(video_file_path))[0]
        clip_fname = video_fname + no_action_class + str(i)
        clip_subdir_fname = os.path.join(no_action_class, clip_fname)
        negative_clip_file_list.append(clip_subdir_fname)
        _extract_clip_ffmpeg(start_time, duration, video_file_path, os.path.join(negative_clip_dir, clip_fname + '.' + clip_format), ffmpeg_path)
    return pd.DataFrame({'negative_clip_file_name': negative_clip_file_list, 'clip_duration': clip_interval_list, 'video_file': video_file})

def extract_sampled_negative_clips(video_info_df, num_negative_samples, video_files, video_dir, clip_dir, classes, no_action_class, negative_clip_length, clip_format, label_filepath):
    if False:
        for i in range(10):
            print('nop')
    '\n    Extract randomly sampled negative clips from a set of videos.\n\n    param video_info_df: Pandas.DataFrame\n        DataFrame containing annotated video information\n    param num_negative_samples: int\n        Number of negative samples to extract\n    param video_files: listof str\n        List of original video files\n    param video_dir: str\n        Directory of original videos\n    param clip_dir: str\n        Directory of extracted clips\n    param classes: dict\n        Classes dictionary\n    param no_action_class: str\n        Name of no action class\n    param negative_clip_length: float\n        Length of clips in seconds\n    param clip_format: str\n        Format for video files\n    param label_filepath: str\n        Path to the label file\n    return: None\n    '
    video_len = {}
    for video in video_files:
        video_len[video] = get_video_length(os.path.join(video_dir, video))
    positive_intervals = defaultdict(list)
    for (index, row) in video_info_df.iterrows():
        clip_file = row.file_list
        int_start = row.temporal_segment_start
        int_end = row.temporal_segment_end
        segment_int = (int_start, int_end)
        positive_intervals[clip_file].append(segment_int)
    clips_sampled = 0
    while clips_sampled < num_negative_samples:
        negative_sample_file = video_files[random.randint(0, len(video_files) - 1)]
        duration = video_len[negative_sample_file]
        clip_start = random.uniform(0.0, duration)
        clip_end = clip_start + negative_clip_length
        if clip_end > duration:
            continue
        if negative_sample_file in positive_intervals.keys():
            clip_positive_intervals = positive_intervals[negative_sample_file]
            if check_interval_overlaps(clip_start, clip_end, clip_positive_intervals):
                continue
        video_path = os.path.join(video_dir, negative_sample_file)
        video_fname = os.path.splitext(negative_sample_file)[0]
        clip_fname = video_fname + no_action_class + str(clips_sampled)
        clip_subdir_fname = os.path.join(no_action_class, clip_fname)
        _extract_clip_ffmpeg(clip_start, negative_clip_length, video_path, os.path.join(clip_dir, clip_subdir_fname + '.' + clip_format))
        with open(label_filepath, 'a') as f:
            f.write('"' + clip_subdir_fname + '"' + ' ' + str(classes[no_action_class]) + '\n')
        clips_sampled += 1