from __future__ import print_function
import os
import csv
import glob
import scipy
import sklearn
import numpy as np
import hmmlearn.hmm
import sklearn.cluster
import pickle as cpickle
import matplotlib.pyplot as plt
from scipy.spatial import distance
import sklearn.discriminant_analysis
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
import pyAudioAnalysis.audioBasicIO as audioBasicIO
import pyAudioAnalysis.audioTrainTest as at
import pyAudioAnalysis.MidTermFeatures as mtf
import pyAudioAnalysis.ShortTermFeatures as stf
' General utility functions '

def smooth_moving_avg(signal, window=11):
    if False:
        while True:
            i = 10
    window = int(window)
    if signal.ndim != 1:
        raise ValueError('')
    if signal.size < window:
        raise ValueError('Input vector needs to be bigger than window size.')
    if window < 3:
        return signal
    s = np.r_[2 * signal[0] - signal[window - 1::-1], signal, 2 * signal[-1] - signal[-1:-window:-1]]
    w = np.ones(window, 'd')
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window:-window + 1]

def self_similarity_matrix(feature_vectors):
    if False:
        while True:
            i = 10
    '\n    This function computes the self-similarity matrix for a sequence\n    of feature vectors.\n    ARGUMENTS:\n     - feature_vectors:    a np matrix (nDims x nVectors) whose i-th column\n                           corresponds to the i-th feature vector\n\n    RETURNS:\n     - sim_matrix:         the self-similarity matrix (nVectors x nVectors)\n    '
    scaler = StandardScaler()
    norm_feature_vectors = scaler.fit_transform(feature_vectors.T).T
    sim_matrix = 1.0 - distance.squareform(distance.pdist(norm_feature_vectors.T, 'cosine'))
    return sim_matrix

def labels_to_segments(labels, window):
    if False:
        i = 10
        return i + 15
    "\n    ARGUMENTS:\n     - labels:     a sequence of class labels (per time window)\n     - window:     window duration (in seconds)\n\n    RETURNS:\n     - segments:   a sequence of segment's limits: segs[i,0] is start and\n                   segs[i,1] are start and end point of segment i\n     - classes:    a sequence of class flags: class[i] is the class ID of\n                   the i-th segment\n    "
    if len(labels) == 1:
        segs = [0, window]
        classes = labels
        return (segs, classes)
    num_segs = 0
    index = 0
    classes = []
    segment_list = []
    cur_label = labels[index]
    while index < len(labels) - 1:
        previous_value = cur_label
        while True:
            index += 1
            compare_flag = labels[index]
            if (compare_flag != cur_label) | (index == len(labels) - 1):
                num_segs += 1
                cur_label = labels[index]
                segment_list.append(index * window)
                classes.append(previous_value)
                break
    segments = np.zeros((len(segment_list), 2))
    for i in range(len(segment_list)):
        if i > 0:
            segments[i, 0] = segment_list[i - 1]
        segments[i, 1] = segment_list[i]
    return (segments, classes)

def segments_to_labels(start_times, end_times, labels, window):
    if False:
        print('Hello World!')
    '\n    This function converts segment endpoints and respective segment\n    labels to fix-sized class labels.\n    ARGUMENTS:\n     - start_times:  segment start points (in seconds)\n     - end_times:    segment endpoints (in seconds)\n     - labels:       segment labels\n     - window:      fix-sized window (in seconds)\n    RETURNS:\n     - flags:    np array of class indices\n     - class_names:    list of classnames (strings)\n    '
    flags = []
    class_names = list(set(labels))
    index = window / 2.0
    while index < end_times[-1]:
        for i in range(len(start_times)):
            if start_times[i] < index <= end_times[i]:
                break
        flags.append(class_names.index(labels[i]))
        index += window
    return (np.array(flags), class_names)

def compute_metrics(confusion_matrix, class_names):
    if False:
        print('Hello World!')
    '\n    This function computes the precision, recall and f1 measures,\n    given a confusion matrix\n    '
    f1 = []
    recall = []
    precision = []
    n_classes = confusion_matrix.shape[0]
    if len(class_names) != n_classes:
        print('Error in computePreRec! Confusion matrix and class_names list must be of the same size!')
    else:
        for (i, c) in enumerate(class_names):
            precision.append(confusion_matrix[i, i] / np.sum(confusion_matrix[:, i]))
            recall.append(confusion_matrix[i, i] / np.sum(confusion_matrix[i, :]))
            f1.append(2 * precision[-1] * recall[-1] / (precision[-1] + recall[-1]))
    return (recall, precision, f1)

def read_segmentation_gt(gt_file):
    if False:
        for i in range(10):
            print('nop')
    "\n    This function reads a segmentation ground truth file,\n    following a simple CSV format with the following columns:\n    <segment start>,<segment end>,<class label>\n\n    ARGUMENTS:\n     - gt_file:       the path of the CSV segment file\n    RETURNS:\n     - seg_start:     a np array of segments' start positions\n     - seg_end:       a np array of segments' ending positions\n     - seg_label:     a list of respective class labels (strings)\n    "
    with open(gt_file, 'rt') as f_handle:
        reader = csv.reader(f_handle, delimiter='\t')
        start_times = []
        end_times = []
        labels = []
        for row in reader:
            if len(row) == 3:
                start_times.append(float(row[0]))
                end_times.append(float(row[1]))
                labels.append(row[2])
    return (np.array(start_times), np.array(end_times), labels)

def plot_segmentation_results(flags_ind, flags_ind_gt, class_names, mt_step, evaluate_only=False):
    if False:
        while True:
            i = 10
    '\n    This function plots statistics on the classification-segmentation results \n    produced either by the fix-sized supervised method or the HMM method.\n    It also computes the overall accuracy achieved by the respective method \n    if ground-truth is available.\n    '
    flags = [class_names[int(f)] for f in flags_ind]
    (segments, classes) = labels_to_segments(flags, mt_step)
    min_len = min(flags_ind.shape[0], flags_ind_gt.shape[0])
    if min_len > 0:
        accuracy = np.sum(flags_ind[0:min_len] == flags_ind_gt[0:min_len]) / float(min_len)
    else:
        accuracy = -1
    if not evaluate_only:
        duration = segments[-1, 1]
        s_percentages = np.zeros((len(class_names),))
        percentages = np.zeros((len(class_names),))
        av_durations = np.zeros((len(class_names),))
        for i_seg in range(segments.shape[0]):
            s_percentages[class_names.index(classes[i_seg])] += segments[i_seg, 1] - segments[i_seg, 0]
        for i in range(s_percentages.shape[0]):
            percentages[i] = 100.0 * s_percentages[i] / duration
            class_sum = sum((1 for c in classes if c == class_names[i]))
            if class_sum > 0:
                av_durations[i] = s_percentages[i] / class_sum
            else:
                av_durations[i] = 0.0
        for i in range(percentages.shape[0]):
            print(class_names[i], percentages[i], av_durations[i])
        font = {'size': 10}
        plt.rc('font', **font)
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.set_yticks(np.array(range(len(class_names))))
        ax1.axis((0, duration, -1, len(class_names)))
        ax1.set_yticklabels(class_names)
        ax1.plot(np.array(range(len(flags_ind))) * mt_step + mt_step / 2.0, flags_ind)
        if flags_ind_gt.shape[0] > 0:
            ax1.plot(np.array(range(len(flags_ind_gt))) * mt_step + mt_step / 2.0, flags_ind_gt + 0.05, '--r')
        plt.xlabel('time (seconds)')
        if accuracy >= 0:
            plt.title('Accuracy = {0:.1f}%'.format(100.0 * accuracy))
        ax2 = fig.add_subplot(223)
        plt.title('Classes percentage durations')
        ax2.axis((0, len(class_names) + 1, 0, 100))
        ax2.set_xticks(np.array(range(len(class_names) + 1)))
        ax2.set_xticklabels([' '] + class_names)
        print(np.array(range(len(class_names))), percentages)
        ax2.bar(np.array(range(len(class_names))) + 0.5, percentages)
        ax3 = fig.add_subplot(224)
        plt.title('Segment average duration per class')
        ax3.axis((0, len(class_names) + 1, 0, av_durations.max()))
        ax3.set_xticks(np.array(range(len(class_names) + 1)))
        ax3.set_xticklabels([' '] + class_names)
        ax3.bar(np.array(range(len(class_names))) + 0.5, av_durations)
        fig.tight_layout()
        plt.show()
    return accuracy

def evaluate_speaker_diarization(labels, labels_gt):
    if False:
        while True:
            i = 10
    min_len = min(labels.shape[0], labels_gt.shape[0])
    labels = labels[0:min_len]
    labels_gt = labels_gt[0:min_len]
    unique_flags = np.unique(labels)
    unique_flags_gt = np.unique(labels_gt)
    contigency_matrix = np.zeros((unique_flags.shape[0], unique_flags_gt.shape[0]))
    for i in range(min_len):
        contigency_matrix[int(np.nonzero(unique_flags == labels[i])[0]), int(np.nonzero(unique_flags_gt == labels_gt[i])[0])] += 1.0
    (columns, rows) = contigency_matrix.shape
    row_sum = np.sum(contigency_matrix, axis=0)
    column_sum = np.sum(contigency_matrix, axis=1)
    matrix_sum = np.sum(contigency_matrix)
    purity_clust = np.zeros((columns,))
    purity_speak = np.zeros((rows,))
    for i in range(columns):
        purity_clust[i] = np.max(contigency_matrix[i, :]) / column_sum[i]
    for j in range(rows):
        purity_speak[j] = np.max(contigency_matrix[:, j]) / row_sum[j]
    purity_cluster_m = np.sum(purity_clust * column_sum) / matrix_sum
    purity_speaker_m = np.sum(purity_speak * row_sum) / matrix_sum
    return (purity_cluster_m, purity_speaker_m)

def train_hmm_compute_statistics(features, labels):
    if False:
        i = 10
        return i + 15
    '\n    This function computes the statistics used to train\n    an HMM joint segmentation-classification model\n    using a sequence of sequential features and respective labels\n\n    ARGUMENTS:\n     - features:  a np matrix of feature vectors (numOfDimensions x n_wins)\n     - labels:    a np array of class indices (n_wins x 1)\n    RETURNS:\n     - class_priors:            matrix of prior class probabilities\n                                (n_classes x 1)\n     - transmutation_matrix:    transition matrix (n_classes x n_classes)\n     - means:                   means matrix (numOfDimensions x 1)\n     - cov:                     deviation matrix (numOfDimensions x 1)\n    '
    unique_labels = np.unique(labels)
    n_comps = len(unique_labels)
    n_feats = features.shape[0]
    if features.shape[1] < labels.shape[0]:
        print('trainHMM warning: number of short-term feature vectors must be greater or equal to the labels length!')
        labels = labels[0:features.shape[1]]
    class_priors = np.zeros((n_comps,))
    for (i, u_label) in enumerate(unique_labels):
        class_priors[i] = np.count_nonzero(labels == u_label)
    class_priors = class_priors / class_priors.sum()
    transmutation_matrix = np.zeros((n_comps, n_comps))
    for i in range(labels.shape[0] - 1):
        transmutation_matrix[int(labels[i]), int(labels[i + 1])] += 1
    for i in range(n_comps):
        transmutation_matrix[i, :] /= transmutation_matrix[i, :].sum()
    means = np.zeros((n_comps, n_feats))
    for i in range(n_comps):
        means[i, :] = np.array(features[:, np.nonzero(labels == unique_labels[i])[0]].mean(axis=1))
    cov = np.zeros((n_comps, n_feats))
    for i in range(n_comps):
        '\n        cov[i, :, :] = np.cov(features[:, np.nonzero(labels == u_labels[i])[0]])\n        '
        cov[i, :] = np.std(features[:, np.nonzero(labels == unique_labels[i])[0]], axis=1)
    return (class_priors, transmutation_matrix, means, cov)

def train_hmm_from_file(wav_file, gt_file, hmm_model_name, mid_window, mid_step):
    if False:
        return 10
    '\n    This function trains a HMM model for segmentation-classification\n    using a single annotated audio file\n    ARGUMENTS:\n     - wav_file:        the path of the audio filename\n     - gt_file:         the path of the ground truth filename\n                       (a csv file of the form <segment start in seconds>,\n                       <segment end in seconds>,<segment label> in each row\n     - hmm_model_name:   the name of the HMM model to be stored\n     - mt_win:          mid-term window size\n     - mt_step:         mid-term window step\n    RETURNS:\n     - hmm:            an object to the resulting HMM\n     - class_names:     a list of class_names\n\n    After training, hmm, class_names, along with the mt_win and mt_step\n    values are stored in the hmm_model_name file\n    '
    (seg_start, seg_end, seg_labs) = read_segmentation_gt(gt_file)
    (flags, class_names) = segments_to_labels(seg_start, seg_end, seg_labs, mid_step)
    (sampling_rate, signal) = audioBasicIO.read_audio_file(wav_file)
    (features, _, _) = mtf.mid_feature_extraction(signal, sampling_rate, mid_window * sampling_rate, mid_step * sampling_rate, round(sampling_rate * 0.05), round(sampling_rate * 0.05))
    (class_priors, transumation_matrix, means, cov) = train_hmm_compute_statistics(features, flags)
    hmm = hmmlearn.hmm.GaussianHMM(class_priors.shape[0], 'diag')
    hmm.covars_ = cov
    hmm.means_ = means
    hmm.startprob_ = class_priors
    hmm.transmat_ = transumation_matrix
    save_hmm(hmm_model_name, hmm, class_names, mid_window, mid_step)
    return (hmm, class_names)

def train_hmm_from_directory(folder_path, hmm_model_name, mid_window, mid_step):
    if False:
        i = 10
        return i + 15
    '\n    This function trains a HMM model for segmentation-classification using\n    a where WAV files and .segment (ground-truth files) are stored\n    ARGUMENTS:\n     - folder_path:     the path of the data diretory\n     - hmm_model_name:  the name of the HMM model to be stored\n     - mt_win:          mid-term window size\n     - mt_step:         mid-term window step\n    RETURNS:\n     - hmm:            an object to the resulting HMM\n     - class_names:    a list of class_names\n\n    After training, hmm, class_names, along with the mt_win\n    and mt_step values are stored in the hmm_model_name file\n    '
    flags_all = np.array([])
    class_names_all = []
    for (i, f) in enumerate(glob.glob(folder_path + os.sep + '*.wav')):
        wav_file = f
        gt_file = f.replace('.wav', '.segments')
        if os.path.isfile(gt_file):
            (seg_start, seg_end, seg_labs) = read_segmentation_gt(gt_file)
            (flags, class_names) = segments_to_labels(seg_start, seg_end, seg_labs, mid_step)
            for c in class_names:
                if c not in class_names_all:
                    class_names_all.append(c)
            (sampling_rate, signal) = audioBasicIO.read_audio_file(wav_file)
            (feature_vector, _, _) = mtf.mid_feature_extraction(signal, sampling_rate, mid_window * sampling_rate, mid_step * sampling_rate, round(sampling_rate * 0.05), round(sampling_rate * 0.05))
            flag_len = len(flags)
            feat_cols = feature_vector.shape[1]
            min_sm = min(feat_cols, flag_len)
            feature_vector = feature_vector[:, 0:min_sm]
            flags = flags[0:min_sm]
            flags_new = []
            for (j, fl) in enumerate(flags):
                flags_new.append(class_names_all.index(class_names_all[flags[j]]))
            flags_all = np.append(flags_all, np.array(flags_new))
            if i == 0:
                f_all = feature_vector
            else:
                f_all = np.concatenate((f_all, feature_vector), axis=1)
    (class_priors, transmutation_matrix, means, cov) = train_hmm_compute_statistics(f_all, flags_all)
    hmm = hmmlearn.hmm.GaussianHMM(class_priors.shape[0], 'diag')
    hmm.covars_ = cov
    hmm.means_ = means
    hmm.startprob_ = class_priors
    hmm.transmat_ = transmutation_matrix
    save_hmm(hmm_model_name, hmm, class_names_all, mid_window, mid_step)
    return (hmm, class_names_all)

def save_hmm(hmm_model_name, model, classes, mid_window, mid_step):
    if False:
        for i in range(10):
            print('nop')
    'Save HMM model'
    with open(hmm_model_name, 'wb') as f_handle:
        cpickle.dump(model, f_handle, protocol=cpickle.HIGHEST_PROTOCOL)
        cpickle.dump(classes, f_handle, protocol=cpickle.HIGHEST_PROTOCOL)
        cpickle.dump(mid_window, f_handle, protocol=cpickle.HIGHEST_PROTOCOL)
        cpickle.dump(mid_step, f_handle, protocol=cpickle.HIGHEST_PROTOCOL)

def hmm_segmentation(audio_file, hmm_model_name, plot_results=False, gt_file=''):
    if False:
        i = 10
        return i + 15
    (sampling_rate, signal) = audioBasicIO.read_audio_file(audio_file)
    with open(hmm_model_name, 'rb') as f_handle:
        hmm = cpickle.load(f_handle)
        class_names = cpickle.load(f_handle)
        mid_window = cpickle.load(f_handle)
        mid_step = cpickle.load(f_handle)
    (features, _, _) = mtf.mid_feature_extraction(signal, sampling_rate, mid_window * sampling_rate, mid_step * sampling_rate, round(sampling_rate * 0.05), round(sampling_rate * 0.05))
    labels = hmm.predict(features.T)
    (labels_gt, class_names_gt, accuracy, cm) = load_ground_truth(gt_file, labels, class_names, mid_step, plot_results)
    return (labels, class_names, accuracy, cm)

def load_ground_truth_segments(gt_file, mt_step):
    if False:
        while True:
            i = 10
    (seg_start, seg_end, seg_labels) = read_segmentation_gt(gt_file)
    (labels, class_names) = segments_to_labels(seg_start, seg_end, seg_labels, mt_step)
    labels_temp = []
    for (index, label) in enumerate(labels):
        if class_names[labels[index]] in class_names:
            labels_temp.append(class_names.index(class_names[labels[index]]))
        else:
            labels_temp.append(-1)
    labels = np.array(labels_temp)
    return (labels, class_names)

def calculate_confusion_matrix(predictions, ground_truth, classes):
    if False:
        return 10
    cm = np.zeros((len(classes), len(classes)))
    for index in range(min(predictions.shape[0], ground_truth.shape[0])):
        cm[int(ground_truth[index]), int(predictions[index])] += 1
    return cm

def mid_term_file_classification(input_file, model_name, model_type, plot_results=False, gt_file=''):
    if False:
        return 10
    "\n    This function performs mid-term classification of an audio stream.\n    Towards this end, supervised knowledge is used,\n    i.e. a pre-trained classifier.\n    ARGUMENTS:\n        - input_file:        path of the input WAV file\n        - model_name:        name of the classification model\n        - model_type:        svm or knn depending on the classifier type\n        - plot_results:      True if results are to be plotted using\n                             matplotlib along with a set of statistics\n        - gt_file:           path to the ground truth file, if exists, \n                             for calculating classification performance\n    RETURNS:\n    labels, class_names, accuracy, cm\n          - labels:         a sequence of segment's labels: segs[i] is the label\n                            of the i-th segment\n          - class_names:    a string sequence of class_names used in classification:\n                            class_names[i] is the name of classes[i]\n          - accuracy:       the accuracy of the classification.\n          - cm:             the confusion matrix of this classification\n    "
    labels = []
    accuracy = 0.0
    class_names = []
    cm = np.array([])
    if not os.path.isfile(model_name):
        print('mtFileClassificationError: input model_type not found!')
        return (labels, class_names, accuracy, cm)
    if model_type == 'knn':
        (classifier, mean, std, class_names, mt_win, mid_step, st_win, st_step, compute_beat) = at.load_model_knn(model_name)
    else:
        (classifier, mean, std, class_names, mt_win, mid_step, st_win, st_step, compute_beat) = at.load_model(model_name)
    if compute_beat:
        print('Model ' + model_name + ' contains long-term music features (beat etc) and cannot be used in segmentation')
        return (labels, class_names, accuracy, cm)
    (sampling_rate, signal) = audioBasicIO.read_audio_file(input_file)
    if sampling_rate == 0:
        return (labels, class_names, accuracy, cm)
    signal = audioBasicIO.stereo_to_mono(signal)
    (mt_feats, _, _) = mtf.mid_feature_extraction(signal, sampling_rate, mt_win * sampling_rate, mid_step * sampling_rate, round(sampling_rate * st_win), round(sampling_rate * st_step))
    posterior_matrix = []
    for col_index in range(mt_feats.shape[1]):
        feature_vector = (mt_feats[:, col_index] - mean) / std
        (label_predicted, posterior) = at.classifier_wrapper(classifier, model_type, feature_vector)
        labels.append(label_predicted)
        posterior_matrix.append(np.max(posterior))
    labels = np.array(labels)
    (segs, classes) = labels_to_segments(labels, mid_step)
    for i in range(len(segs)):
        print(segs[i], classes[i])
    segs[-1] = len(signal) / float(sampling_rate)
    (labels_gt, class_names_gt, accuracy, cm) = load_ground_truth(gt_file, labels, class_names, mid_step, plot_results)
    return (labels, class_names, accuracy, cm)

def load_ground_truth(gt_file, labels, class_names, mid_step, plot_results):
    if False:
        i = 10
        return i + 15
    accuracy = 0
    cm = np.array([])
    labels_gt = np.array([])
    if os.path.isfile(gt_file):
        (labels_gt, class_names_gt) = load_ground_truth_segments(gt_file, mid_step)
        labels_new = []
        for (il, l) in enumerate(labels):
            if class_names[int(l)] in class_names_gt:
                labels_new.append(class_names_gt.index(class_names[int(l)]))
            else:
                labels_new.append(-1)
        labels_new = np.array(labels_new)
        cm = calculate_confusion_matrix(labels_new, labels_gt, class_names_gt)
        accuracy = plot_segmentation_results(labels_new, labels_gt, class_names_gt, mid_step, not plot_results)
        if accuracy >= 0:
            print('Overall Accuracy: {0:.2f}'.format(accuracy))
    return (labels_gt, class_names, accuracy, cm)

def evaluate_segmentation_classification_dir(dir_name, model_name, method_name):
    if False:
        while True:
            i = 10
    accuracies = []
    class_names = []
    cm_total = np.array([])
    for (index, wav_file) in enumerate(glob.glob(dir_name + os.sep + '*.wav')):
        print(wav_file)
        gt_file = wav_file.replace('.wav', '.segments')
        if method_name.lower() in ['svm', 'svm_rbf', 'knn', 'randomforest', 'gradientboosting', 'extratrees']:
            (flags_ind, class_names, accuracy, cm_temp) = mid_term_file_classification(wav_file, model_name, method_name, False, gt_file)
        else:
            (flags_ind, class_names, accuracy, cm_temp) = hmm_segmentation(wav_file, model_name, False, gt_file)
        if accuracy > 0:
            if not index:
                cm_total = np.copy(cm_temp)
            else:
                cm_total = cm_total + cm_temp
            accuracies.append(accuracy)
            print(cm_temp, class_names)
            print(cm_total)
    if len(cm_total.shape) > 1:
        cm_total = cm_total / np.sum(cm_total)
        (rec, pre, f1) = compute_metrics(cm_total, class_names)
        print(' - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
        print('Average Accuracy: {0:.1f}'.format(100.0 * np.array(accuracies).mean()))
        print('Average recall: {0:.1f}'.format(100.0 * np.array(rec).mean()))
        print('Average precision: {0:.1f}'.format(100.0 * np.array(pre).mean()))
        print('Average f1: {0:.1f}'.format(100.0 * np.array(f1).mean()))
        print('Median Accuracy: {0:.1f}'.format(100.0 * np.median(np.array(accuracies))))
        print('Min Accuracy: {0:.1f}'.format(100.0 * np.array(accuracies).min()))
        print('Max Accuracy: {0:.1f}'.format(100.0 * np.array(accuracies).max()))
    else:
        print('Confusion matrix was empty, accuracy for every file was 0')

def silence_removal(signal, sampling_rate, st_win, st_step, smooth_window=0.5, weight=0.5, plot=False):
    if False:
        print('Hello World!')
    '\n    Event Detection (silence removal)\n    ARGUMENTS:\n         - signal:                the input audio signal\n         - sampling_rate:               sampling freq\n         - st_win, st_step:    window size and step in seconds\n         - smoothWindow:     (optinal) smooth window (in seconds)\n         - weight:           (optinal) weight factor (0 < weight < 1)\n                              the higher, the more strict\n         - plot:             (optinal) True if results are to be plotted\n    RETURNS:\n         - seg_limits:    list of segment limits in seconds (e.g [[0.1, 0.9],\n                          [1.4, 3.0]] means that\n                          the resulting segments are (0.1 - 0.9) seconds\n                          and (1.4, 3.0) seconds\n    '
    if weight >= 1:
        weight = 0.99
    if weight <= 0:
        weight = 0.01
    signal = audioBasicIO.stereo_to_mono(signal)
    (st_feats, _) = stf.feature_extraction(signal, sampling_rate, st_win * sampling_rate, st_step * sampling_rate)
    st_energy = st_feats[1, :]
    en = np.sort(st_energy)
    st_windows_fraction = int(len(en) / 10)
    low_threshold = np.mean(en[0:st_windows_fraction]) + 1e-15
    high_threshold = np.mean(en[-st_windows_fraction:-1]) + 1e-15
    low_energy = st_feats[:, np.where(st_energy <= low_threshold)[0]]
    high_energy = st_feats[:, np.where(st_energy >= high_threshold)[0]]
    features = [low_energy.T, high_energy.T]
    (features, labels) = at.features_to_matrix(features)
    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features)
    mean = scaler.mean_
    std = scaler.scale_
    svm = at.train_svm(features_norm, labels, 1.0)
    prob_on_set = []
    for index in range(st_feats.shape[1]):
        cur_fv = (st_feats[:, index] - mean) / std
        prob_on_set.append(svm.predict_proba(cur_fv.reshape(1, -1))[0][1])
    prob_on_set = np.array(prob_on_set)
    prob_on_set = smooth_moving_avg(prob_on_set, smooth_window / st_step)
    prog_on_set_sort = np.sort(prob_on_set)
    nt = int(prog_on_set_sort.shape[0] / 10)
    threshold = np.mean((1 - weight) * prog_on_set_sort[0:nt]) + weight * np.mean(prog_on_set_sort[-nt:])
    max_indices = np.where(prob_on_set > threshold)[0]
    index = 0
    seg_limits = []
    time_clusters = []
    while index < len(max_indices):
        cur_cluster = [max_indices[index]]
        if index == len(max_indices) - 1:
            break
        while max_indices[index + 1] - cur_cluster[-1] <= 2:
            cur_cluster.append(max_indices[index + 1])
            index += 1
            if index == len(max_indices) - 1:
                break
        index += 1
        time_clusters.append(cur_cluster)
        seg_limits.append([cur_cluster[0] * st_step, cur_cluster[-1] * st_step])
    min_duration = 0.2
    seg_limits_2 = []
    for s_lim in seg_limits:
        if s_lim[1] - s_lim[0] > min_duration:
            seg_limits_2.append(s_lim)
    seg_limits = seg_limits_2
    if plot:
        time_x = np.arange(0, signal.shape[0] / float(sampling_rate), 1.0 / sampling_rate)
        plt.subplot(2, 1, 1)
        plt.plot(time_x, signal)
        for s_lim in seg_limits:
            plt.axvline(x=s_lim[0], color='red')
            plt.axvline(x=s_lim[1], color='red')
        plt.subplot(2, 1, 2)
        plt.plot(np.arange(0, prob_on_set.shape[0] * st_step, st_step), prob_on_set)
        plt.title('Signal')
        for s_lim in seg_limits:
            plt.axvline(x=s_lim[0], color='red')
            plt.axvline(x=s_lim[1], color='red')
        plt.title('svm Probability')
        plt.show()
    return seg_limits

def speaker_diarization(filename, n_speakers, mid_window=1.0, mid_step=0.1, short_window=0.1, lda_dim=0, plot_res=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    ARGUMENTS:\n        - filename:        the name of the WAV file to be analyzed\n        - n_speakers       the number of speakers (clusters) in\n                           the recording (<=0 for unknown)\n        - mid_window (opt)    mid-term window size\n        - mid_step (opt)    mid-term window step\n        - short_window  (opt)    short-term window size\n        - lda_dim (opt     LDA dimension (0 for no LDA)\n        - plot_res         (opt)   0 for not plotting the results 1 for plotting\n    '
    (sampling_rate, signal) = audioBasicIO.read_audio_file(filename)
    signal = audioBasicIO.stereo_to_mono(signal)
    duration = len(signal) / sampling_rate
    base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/models')
    (classifier_all, mean_all, std_all, class_names_all, _, _, _, _, _) = at.load_model(os.path.join(base_dir, 'svm_rbf_speaker_10'))
    (classifier_fm, mean_fm, std_fm, class_names_fm, _, _, _, _, _) = at.load_model(os.path.join(base_dir, 'svm_rbf_speaker_male_female'))
    (mid_feats, st_feats, a) = mtf.mid_feature_extraction(signal, sampling_rate, mid_window * sampling_rate, mid_step * sampling_rate, round(sampling_rate * 0.05), round(sampling_rate * 0.05))
    mid_term_features = np.zeros((mid_feats.shape[0] + len(class_names_all) + len(class_names_fm), mid_feats.shape[1]))
    for index in range(mid_feats.shape[1]):
        feature_norm_all = (mid_feats[:, index] - mean_all) / std_all
        feature_norm_fm = (mid_feats[:, index] - mean_fm) / std_fm
        (_, p1) = at.classifier_wrapper(classifier_all, 'svm_rbf', feature_norm_all)
        (_, p2) = at.classifier_wrapper(classifier_fm, 'svm_rbf', feature_norm_fm)
        start = mid_feats.shape[0]
        end = mid_feats.shape[0] + len(class_names_all)
        mid_term_features[0:mid_feats.shape[0], index] = mid_feats[:, index]
        mid_term_features[start:end, index] = p1 + 0.0001
        mid_term_features[end:, index] = p2 + 0.0001
    scaler = StandardScaler()
    mid_feats_norm = scaler.fit_transform(mid_term_features.T)
    dist_all = np.sum(distance.squareform(distance.pdist(mid_feats_norm.T)), axis=0)
    m_dist_all = np.mean(dist_all)
    i_non_outliers = np.nonzero(dist_all < 1.1 * m_dist_all)[0]
    mt_feats_norm_or = mid_feats_norm
    mid_feats_norm = mid_feats_norm[:, i_non_outliers]
    if lda_dim > 0:
        window_ratio = int(round(mid_window / short_window))
        step_ratio = int(round(short_window / short_window))
        mt_feats_to_red = []
        num_of_features = len(st_feats)
        num_of_stats = 2
        for index in range(num_of_stats * num_of_features):
            mt_feats_to_red.append([])
        for index in range(num_of_features):
            cur_pos = 0
            feat_len = len(st_feats[index])
            while cur_pos < feat_len:
                n1 = cur_pos
                n2 = cur_pos + window_ratio
                if n2 > feat_len:
                    n2 = feat_len
                short_features = st_feats[index][n1:n2]
                mt_feats_to_red[index].append(np.mean(short_features))
                mt_feats_to_red[index + num_of_features].append(np.std(short_features))
                cur_pos += step_ratio
        mt_feats_to_red = np.array(mt_feats_to_red)
        mt_feats_to_red_2 = np.zeros((mt_feats_to_red.shape[0] + len(class_names_all) + len(class_names_fm), mt_feats_to_red.shape[1]))
        limit = mt_feats_to_red.shape[0] + len(class_names_all)
        for index in range(mt_feats_to_red.shape[1]):
            feature_norm_all = (mt_feats_to_red[:, index] - mean_all) / std_all
            feature_norm_fm = (mt_feats_to_red[:, index] - mean_fm) / std_fm
            (_, p1) = at.classifier_wrapper(classifier_all, 'svm_rbf', feature_norm_all)
            (_, p2) = at.classifier_wrapper(classifier_fm, 'svm_rbf', feature_norm_fm)
            mt_feats_to_red_2[0:mt_feats_to_red.shape[0], index] = mt_feats_to_red[:, index]
            mt_feats_to_red_2[mt_feats_to_red.shape[0]:limit, index] = p1 + 0.0001
            mt_feats_to_red_2[limit:, index] = p2 + 0.0001
        mt_feats_to_red = mt_feats_to_red_2
        scaler = StandardScaler()
        mt_feats_to_red = scaler.fit_transform(mt_feats_to_red.T).T
        labels = np.zeros((mt_feats_to_red.shape[1],))
        lda_step = 1.0
        lda_step_ratio = lda_step / short_window
        for index in range(labels.shape[0]):
            labels[index] = int(index * short_window / lda_step_ratio)
        clf = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=lda_dim)
        mid_feats_norm = clf.fit_transform(mt_feats_to_red.T, labels)
    if n_speakers <= 0:
        s_range = range(2, 10)
    else:
        s_range = [n_speakers]
    cluster_labels = []
    sil_all = []
    cluster_centers = []
    for speakers in s_range:
        k_means = sklearn.cluster.KMeans(n_clusters=speakers)
        k_means.fit(mid_feats_norm)
        cls = k_means.labels_
        cluster_labels.append(cls)
        sil_1 = []
        sil_2 = []
        for c in range(speakers):
            clust_per_cent = np.nonzero(cls == c)[0].shape[0] / float(len(cls))
            if clust_per_cent < 0.02:
                sil_1.append(0.0)
                sil_2.append(0.0)
            else:
                mt_feats_norm_temp = mid_feats_norm[cls == c, :]
                dist = distance.pdist(mt_feats_norm_temp.T)
                sil_1.append(np.mean(dist) * clust_per_cent)
                sil_temp = []
                for c2 in range(speakers):
                    if c2 != c:
                        clust_per_cent_2 = np.nonzero(cls == c2)[0].shape[0] / float(len(cls))
                        mid_features_temp = mid_feats_norm[cls == c2, :]
                        dist = distance.cdist(mt_feats_norm_temp, mid_features_temp)
                        sil_temp.append(np.mean(dist) * (clust_per_cent + clust_per_cent_2) / 2.0)
                sil_temp = np.array(sil_temp)
                sil_2.append(min(sil_temp))
        sil_1 = np.array(sil_1)
        sil_2 = np.array(sil_2)
        sil = []
        for c in range(speakers):
            sil.append((sil_2[c] - sil_1[c]) / (max(sil_2[c], sil_1[c]) + 1e-05))
        sil_all.append(np.mean(sil))
    imax = int(np.argmax(sil_all))
    num_speakers = s_range[imax]
    if lda_dim <= 0:
        for index in range(1):
            (start_prob, transmat, means, cov) = train_hmm_compute_statistics(mt_feats_norm_or.T, cls)
            hmm = hmmlearn.hmm.GaussianHMM(start_prob.shape[0], 'diag')
            hmm.startprob_ = start_prob
            hmm.transmat_ = transmat
            hmm.means_ = means
            hmm.covars_ = cov
            cls = hmm.predict(mt_feats_norm_or)
    cls = scipy.signal.medfilt(cls, 5)
    class_names = ['speaker{0:d}'.format(c) for c in range(num_speakers)]
    gt_file = filename.replace('.wav', '.segments')
    if os.path.isfile(gt_file):
        (seg_start, seg_end, seg_labs) = read_segmentation_gt(gt_file)
        (flags_gt, class_names_gt) = segments_to_labels(seg_start, seg_end, seg_labs, mid_step)
    if plot_res:
        fig = plt.figure()
        if n_speakers > 0:
            ax1 = fig.add_subplot(111)
        else:
            ax1 = fig.add_subplot(211)
        ax1.set_yticks(np.array(range(len(class_names))))
        ax1.axis((0, duration, -1, len(class_names)))
        ax1.set_yticklabels(class_names)
        ax1.plot(np.array(range(len(cls))) * mid_step + mid_step / 2.0, cls)
    (purity_cluster_m, purity_speaker_m) = (-1, -1)
    if os.path.isfile(gt_file):
        if plot_res:
            ax1.plot(np.array(range(len(flags_gt))) * mid_step + mid_step / 2.0, flags_gt, 'r')
        (purity_cluster_m, purity_speaker_m) = evaluate_speaker_diarization(cls, flags_gt)
        print('{0:.1f}\t{1:.1f}'.format(100 * purity_cluster_m, 100 * purity_speaker_m))
        if plot_res:
            plt.title('Cluster purity: {0:.1f}% - Speaker purity: {1:.1f}%'.format(100 * purity_cluster_m, 100 * purity_speaker_m))
    if plot_res:
        plt.xlabel('time (seconds)')
        if n_speakers <= 0:
            plt.subplot(212)
            plt.plot(s_range, sil_all)
            plt.xlabel('number of clusters')
            plt.ylabel("average clustering's sillouette")
        plt.show()
    return (cls, purity_cluster_m, purity_speaker_m)

def speaker_diarization_evaluation(folder_name, lda_dimensions):
    if False:
        i = 10
        return i + 15
    '\n        This function prints the cluster purity and speaker purity for\n        each WAV file stored in a provided directory (.SEGMENT files\n         are needed as ground-truth)\n        ARGUMENTS:\n            - folder_name:     the full path of the folder where the WAV and\n                               segment (ground-truth) files are stored\n            - lda_dimensions:  a list of LDA dimensions (0 for no LDA)\n    '
    types = ('*.wav',)
    wav_files = []
    for files in types:
        wav_files.extend(glob.glob(os.path.join(folder_name, files)))
    wav_files = sorted(wav_files)
    num_speakers = []
    for wav_file in wav_files:
        gt_file = wav_file.replace('.wav', '.segments')
        if os.path.isfile(gt_file):
            (_, _, seg_labs) = read_segmentation_gt(gt_file)
            num_speakers.append(len(list(set(seg_labs))))
        else:
            num_speakers.append(-1)
    for dim in lda_dimensions:
        print('LDA = {0:d}'.format(dim))
        for (i, wav_file) in enumerate(wav_files):
            speaker_diarization(wav_file, num_speakers[i], 2.0, 0.2, 0.05, dim, plot_res=False)

def music_thumbnailing(signal, sampling_rate, short_window=1.0, short_step=0.5, thumb_size=10.0, limit_1=0, limit_2=1):
    if False:
        while True:
            i = 10
    '\n    This function detects instances of the most representative part of a\n    music recording, also called "music thumbnails".\n    A technique similar to the one proposed in [1], however a wider set of\n    audio features is used instead of chroma features.\n    In particular the following steps are followed:\n     - Extract short-term audio features. Typical short-term window size: 1\n       second\n     - Compute the self-similarity matrix, i.e. all pairwise similarities\n       between feature vectors\n     - Apply a diagonal mask is as a moving average filter on the values of the\n       self-similarty matrix.\n       The size of the mask is equal to the desirable thumbnail length.\n     - Find the position of the maximum value of the new (filtered)\n       self-similarity matrix. The audio segments that correspond to the\n       diagonial around that position are the selected thumbnails\n    \n\n    ARGUMENTS:\n     - signal:            input signal\n     - sampling_rate:            sampling frequency\n     - short_window:     window size (in seconds)\n     - short_step:    window step (in seconds)\n     - thumb_size:    desider thumbnail size (in seconds)\n    \n    RETURNS:\n     - A1:            beginning of 1st thumbnail (in seconds)\n     - A2:            ending of 1st thumbnail (in seconds)\n     - B1:            beginning of 2nd thumbnail (in seconds)\n     - B2:            ending of 2nd thumbnail (in seconds)\n\n    USAGE EXAMPLE:\n       import audioFeatureExtraction as aF\n     [fs, x] = basicIO.readAudioFile(input_file)\n     [A1, A2, B1, B2] = musicThumbnailing(x, fs)\n\n    [1] Bartsch, M. A., & Wakefield, G. H. (2005). Audio thumbnailing\n    of popular music using chroma-based representations.\n    Multimedia, IEEE Transactions on, 7(1), 96-104.\n    '
    signal = audioBasicIO.stereo_to_mono(signal)
    (st_feats, _) = stf.feature_extraction(signal, sampling_rate, sampling_rate * short_window, sampling_rate * short_step)
    sim_matrix = self_similarity_matrix(st_feats)
    m_filter = int(round(thumb_size / short_step))
    diagonal = np.eye(m_filter, m_filter)
    sim_matrix = scipy.signal.convolve2d(sim_matrix, diagonal, 'valid')
    min_sm = np.min(sim_matrix)
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            if abs(i - j) < 5.0 / short_step or i > j:
                sim_matrix[i, j] = min_sm
    sim_matrix[0:int(limit_1 * sim_matrix.shape[0]), :] = min_sm
    sim_matrix[:, 0:int(limit_1 * sim_matrix.shape[0])] = min_sm
    sim_matrix[int(limit_2 * sim_matrix.shape[0]):, :] = min_sm
    sim_matrix[:, int(limit_2 * sim_matrix.shape[0]):] = min_sm
    (rows, cols) = np.unravel_index(sim_matrix.argmax(), sim_matrix.shape)
    i1 = rows
    i2 = rows
    j1 = cols
    j2 = cols
    while i2 - i1 < m_filter:
        if i1 <= 0 or j1 <= 0 or i2 >= sim_matrix.shape[0] - 2 or (j2 >= sim_matrix.shape[1] - 2):
            break
        if sim_matrix[i1 - 1, j1 - 1] > sim_matrix[i2 + 1, j2 + 1]:
            i1 -= 1
            j1 -= 1
        else:
            i2 += 1
            j2 += 1
    return (short_step * i1, short_step * i2, short_step * j1, short_step * j2, sim_matrix)