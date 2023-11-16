from __future__ import print_function
import os
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
from pyAudioAnalysis import utilities
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
eps = 1e-08
' Time-domain audio features '

def beat_extraction(short_features, window_size, plot=False):
    if False:
        i = 10
        return i + 15
    '\n    This function extracts an estimate of the beat rate for a musical signal.\n    ARGUMENTS:\n     - short_features:     a np array (n_feats x numOfShortTermWindows)\n     - window_size:        window size in seconds\n    RETURNS:\n     - bpm:            estimates of beats per minute\n     - ratio:          a confidence measure\n    '
    selected_features = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    max_beat_time = int(round(2.0 / window_size))
    hist_all = np.zeros((max_beat_time,))
    for (ii, i) in enumerate(selected_features):
        dif_threshold = 2.0 * np.abs(short_features[i, 0:-1] - short_features[i, 1:]).mean()
        if dif_threshold <= 0:
            dif_threshold = 1e-16
        [pos1, _] = utilities.peakdet(short_features[i, :], dif_threshold)
        position_diffs = []
        for j in range(len(pos1) - 1):
            position_diffs.append(pos1[j + 1] - pos1[j])
        (histogram_times, histogram_edges) = np.histogram(position_diffs, np.arange(0.5, max_beat_time + 1.5))
        hist_centers = (histogram_edges[0:-1] + histogram_edges[1:]) / 2.0
        histogram_times = histogram_times.astype(float) / short_features.shape[1]
        hist_all += histogram_times
        if plot:
            plt.subplot(9, 2, ii + 1)
            plt.plot(short_features[i, :], 'k')
            for k in pos1:
                plt.plot(k, short_features[i, k], 'k*')
            f1 = plt.gca()
            f1.axes.get_xaxis().set_ticks([])
            f1.axes.get_yaxis().set_ticks([])
    if plot:
        plt.show(block=False)
        plt.figure()
    max_indices = np.argmax(hist_all)
    bpms = 60 / (hist_centers * window_size)
    bpm = bpms[max_indices]
    ratio = hist_all[max_indices] / (hist_all.sum() + eps)
    if plot:
        hist_all = hist_all[bpms < 500]
        bpms = bpms[bpms < 500]
        plt.plot(bpms, hist_all, 'k')
        plt.xlabel('Beats per minute')
        plt.ylabel('Freq Count')
        plt.show(block=True)
    return (bpm, ratio)

def mid_feature_extraction(signal, sampling_rate, mid_window, mid_step, short_window, short_step):
    if False:
        i = 10
        return i + 15
    '\n    Mid-term feature extraction\n    '
    (short_features, short_feature_names) = ShortTermFeatures.feature_extraction(signal, sampling_rate, short_window, short_step)
    n_stats = 2
    n_feats = len(short_features)
    mid_window_ratio = round((mid_window - (short_window - short_step)) / short_step)
    mt_step_ratio = int(round(mid_step / short_step))
    (mid_features, mid_feature_names) = ([], [])
    for i in range(n_stats * n_feats):
        mid_features.append([])
        mid_feature_names.append('')
    for i in range(n_feats):
        cur_position = 0
        num_short_features = len(short_features[i])
        mid_feature_names[i] = short_feature_names[i] + '_' + 'mean'
        mid_feature_names[i + n_feats] = short_feature_names[i] + '_' + 'std'
        while cur_position < num_short_features:
            end = cur_position + mid_window_ratio
            if end > num_short_features:
                end = num_short_features
            cur_st_feats = short_features[i][cur_position:end]
            mid_features[i].append(np.mean(cur_st_feats))
            mid_features[i + n_feats].append(np.std(cur_st_feats))
            cur_position += mt_step_ratio
    mid_features = np.array(mid_features)
    mid_features = np.nan_to_num(mid_features)
    return (mid_features, short_features, mid_feature_names)
' Feature Extraction Wrappers\n - The first two feature extraction wrappers are used to extract \n   long-term averaged audio features for a list of WAV files stored in a \n   given category.\n   It is important to note that, one single feature is extracted per WAV \n   file (not the whole sequence of feature vectors)\n\n '

def directory_feature_extraction(folder_path, mid_window, mid_step, short_window, short_step, compute_beat=True):
    if False:
        print('Hello World!')
    '\n    This function extracts the mid-term features of the WAVE files of a \n    particular folder.\n\n    The resulting feature vector is extracted by long-term averaging the\n    mid-term features.\n    Therefore ONE FEATURE VECTOR is extracted for each WAV file.\n\n    ARGUMENTS:\n        - folder_path:        the path of the WAVE directory\n        - mid_window, mid_step:    mid-term window and step (in seconds)\n        - short_window, short_step:    short-term window and step (in seconds)\n    '
    mid_term_features = np.array([])
    process_times = []
    types = ('*.wav', '*.aif', '*.aiff', '*.mp3', '*.au', '*.ogg')
    wav_file_list = []
    for files in types:
        wav_file_list.extend(glob.glob(os.path.join(folder_path, files)))
    wav_file_list = sorted(wav_file_list)
    (wav_file_list2, mid_feature_names) = ([], [])
    for (i, file_path) in enumerate(wav_file_list):
        print('Analyzing file {0:d} of {1:d}: {2:s}'.format(i + 1, len(wav_file_list), file_path))
        if os.stat(file_path).st_size == 0:
            print('   (EMPTY FILE -- SKIPPING)')
            continue
        (sampling_rate, signal) = audioBasicIO.read_audio_file(file_path)
        if sampling_rate == 0:
            continue
        t1 = time.time()
        signal = audioBasicIO.stereo_to_mono(signal)
        if signal.shape[0] < float(sampling_rate) / 5:
            print('  (AUDIO FILE TOO SMALL - SKIPPING)')
            continue
        wav_file_list2.append(file_path)
        if compute_beat:
            (mid_features, short_features, mid_feature_names) = mid_feature_extraction(signal, sampling_rate, round(mid_window * sampling_rate), round(mid_step * sampling_rate), round(sampling_rate * short_window), round(sampling_rate * short_step))
            (beat, beat_conf) = beat_extraction(short_features, short_step)
        else:
            (mid_features, _, mid_feature_names) = mid_feature_extraction(signal, sampling_rate, round(mid_window * sampling_rate), round(mid_step * sampling_rate), round(sampling_rate * short_window), round(sampling_rate * short_step))
        mid_features = np.transpose(mid_features)
        mid_features = mid_features.mean(axis=0)
        if not np.isnan(mid_features).any() and (not np.isinf(mid_features).any()):
            if compute_beat:
                mid_features = np.append(mid_features, beat)
                mid_features = np.append(mid_features, beat_conf)
                mid_feature_names += ['bpm', 'ratio']
            if len(mid_term_features) == 0:
                mid_term_features = mid_features
            else:
                mid_term_features = np.vstack((mid_term_features, mid_features))
            t2 = time.time()
            duration = float(len(signal)) / sampling_rate
            process_times.append((t2 - t1) / duration)
    if len(process_times) > 0:
        print('Feature extraction complexity ratio: {0:.1f} x realtime'.format(1.0 / np.mean(np.array(process_times))))
    return (mid_term_features, wav_file_list2, mid_feature_names)

def multiple_directory_feature_extraction(path_list, mid_window, mid_step, short_window, short_step, compute_beat=False):
    if False:
        print('Hello World!')
    "\n    Same as dirWavFeatureExtraction, but instead of a single dir it\n    takes a list of paths as input and returns a list of feature matrices.\n    EXAMPLE:\n    [features, classNames] =\n           a.dirsWavFeatureExtraction(['audioData/classSegmentsRec/noise',\n                                       'audioData/classSegmentsRec/speech',\n                                       'audioData/classSegmentsRec/brush-teeth',\n                                       'audioData/classSegmentsRec/shower'], 1, \n                                       1, 0.02, 0.02);\n\n    It can be used during the training process of a classification model ,\n    in order to get feature matrices from various audio classes (each stored in\n    a separate path)\n    "
    features = []
    class_names = []
    file_names = []
    for (i, d) in enumerate(path_list):
        (f, fn, feature_names) = directory_feature_extraction(d, mid_window, mid_step, short_window, short_step, compute_beat=compute_beat)
        if f.shape[0] > 0:
            features.append(f)
            file_names.append(fn)
            if d[-1] == os.sep:
                class_names.append(d.split(os.sep)[-2])
            else:
                class_names.append(d.split(os.sep)[-1])
    return (features, class_names, file_names)

def directory_feature_extraction_no_avg(folder_path, mid_window, mid_step, short_window, short_step):
    if False:
        print('Hello World!')
    '\n    This function extracts the mid-term features of the WAVE\n    files of a particular folder without averaging each file.\n\n    ARGUMENTS:\n        - folder_path:          the path of the WAVE directory\n        - mid_window, mid_step:    mid-term window and step (in seconds)\n        - short_window, short_step:    short-term window and step (in seconds)\n    RETURNS:\n        - X:                A feature matrix\n        - Y:                A matrix of file labels\n        - filenames:\n    '
    wav_file_list = []
    signal_idx = np.array([])
    mid_features = np.array([])
    types = ('*.wav', '*.aif', '*.aiff', '*.ogg')
    for files in types:
        wav_file_list.extend(glob.glob(os.path.join(folder_path, files)))
    wav_file_list = sorted(wav_file_list)
    for (i, file_path) in enumerate(wav_file_list):
        (sampling_rate, signal) = audioBasicIO.read_audio_file(file_path)
        if sampling_rate == 0:
            continue
        signal = audioBasicIO.stereo_to_mono(signal)
        (mid_feature_vector, _, _) = mid_feature_extraction(signal, sampling_rate, round(mid_window * sampling_rate), round(mid_step * sampling_rate), round(sampling_rate * short_window), round(sampling_rate * short_step))
        mid_feature_vector = np.transpose(mid_feature_vector)
        if len(mid_features) == 0:
            mid_features = mid_feature_vector
            signal_idx = np.zeros((mid_feature_vector.shape[0],))
        else:
            mid_features = np.vstack((mid_features, mid_feature_vector))
            signal_idx = np.append(signal_idx, i * np.ones((mid_feature_vector.shape[0],)))
    return (mid_features, signal_idx, wav_file_list)
'\nThe following two feature extraction wrappers extract features for given audio\nfiles, however  NO LONG-TERM AVERAGING is performed. Therefore, the output for\neach audio file is NOT A SINGLE FEATURE VECTOR but a whole feature matrix.\n\nAlso, another difference between the following two wrappers and the previous\nis that they NO LONG-TERM AVERAGING IS PERFORMED. In other words, the WAV\nfiles in these functions are not used as uniform samples that need to be\naveraged but as sequences\n'

def mid_feature_extraction_to_file(file_path, mid_window, mid_step, short_window, short_step, output_file, store_short_features=False, store_csv=False, plot=False):
    if False:
        print('Hello World!')
    '\n    This function is used as a wrapper to:\n    a) read the content of a WAV file\n    b) perform mid-term feature extraction on that signal\n    c) write the mid-term feature sequences to a np file\n    d) optionally write contents to csv file as well\n    e) optionally write short-term features in csv and np file\n    '
    (sampling_rate, signal) = audioBasicIO.read_audio_file(file_path)
    signal = audioBasicIO.stereo_to_mono(signal)
    (mid_features, short_features, _) = mid_feature_extraction(signal, sampling_rate, round(sampling_rate * mid_window), round(sampling_rate * mid_step), round(sampling_rate * short_window), round(sampling_rate * short_step))
    if store_short_features:
        np.save(output_file + '_st', short_features)
        if plot:
            print('Short-term np file: ' + output_file + '_st.npy saved')
        if store_csv:
            np.savetxt(output_file + '_st.csv', short_features.T, delimiter=',')
            if plot:
                print('Short-term CSV file: ' + output_file + '_st.csv saved')
    np.save(output_file + '_mt', mid_features)
    if plot:
        print('Mid-term np file: ' + output_file + '_mt.npy saved')
    if store_csv:
        np.savetxt(output_file + '_mt.csv', mid_features.T, delimiter=',')
        if plot:
            print('Mid-term CSV file: ' + output_file + '_mt.csv saved')

def mid_feature_extraction_file_dir(folder_path, mid_window, mid_step, short_window, short_step, store_short_features=False, store_csv=False, plot=False):
    if False:
        while True:
            i = 10
    types = (folder_path + os.sep + '*.wav',)
    files_list = []
    for t in types:
        files_list.extend(glob.glob(t))
    for f in files_list:
        output_path = f
        mid_feature_extraction_to_file(f, mid_window, mid_step, short_window, short_step, output_path, store_short_features, store_csv, plot)