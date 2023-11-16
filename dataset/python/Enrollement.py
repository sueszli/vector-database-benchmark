import csv
import os
from pathlib import Path

import joblib
import neurokit2 as nk
import numpy as np
import pandas as pd
from numpy import mean
from scipy.signal import lfilter, find_peaks, peak_widths, peak_prominences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from DataProcessing import k_nearest_neighbour_on_waves, get_time, get_amplitude, get_distance, get_slope, get_angle, \
    analyze_dataset, remove_nan
from Filters import HighPassFilter, BandStopFilter, LowPassFilter, SmoothSignal


def process_new_data():
    old_headers = pd.read_csv('datasets/dataset.csv', nrows=0).columns.tolist()
    headers = pd.read_csv('datasets/analyzed_dataset.csv', nrows=0).columns.tolist()
    signal_to_append = []
    for file in Path("./classify/to_be_processed/").glob('*.csv'):
        extracted_signal = signal_processing("classify/to_be_processed/" + file.name)
        if len(extracted_signal) == 0:
            os.remove("classify/to_be_processed/" + file.name)
            continue
        patient_datas = pd.Series(extracted_signal, index=old_headers)
        signal_to_append.append(patient_datas[headers].tolist())
        os.remove("classify/to_be_processed/" + file.name)
    with open('classify/to_predict.csv', 'a') as file:
        writer = csv.writer(file)
        if os.stat('classify/to_predict.csv').st_size == 0:
            writer.writerow(headers)
        writer.writerows(signal_to_append)


# wrapper so that the processing can be used also for predictions
def wr_process_processing(filename, dataset):
    df = pd.read_csv(dataset)
    signal_processed = signal_processing(filename)
    if len(signal_processed) == 0:
        return
    patient_datas = pd.Series(signal_processed, index=df.columns)
    df = df.append(patient_datas, ignore_index=True)
    df.to_csv(dataset, index=False)


def signal_processing(filename):
    enroll_file = open(filename, "r")
    patient_name = ""
    signal_list = []
    for index, line in enumerate(enroll_file):
        if index == 0:
            key, value = line.split(",")
            patient_name = value
        if index > 12:
            # signal acquisition
            signal_list.append(float(line.replace(",", ".")) / 1000)
    denoised_ecg = lfilter(HighPassFilter(), 1, signal_list)
    denoised_ecg = lfilter(BandStopFilter(), 1, denoised_ecg)
    denoised_ecg = lfilter(LowPassFilter(), 1, denoised_ecg)

    cleaned_signal = SmoothSignal(denoised_ecg)

    # only keep best r peaks with prominence = 1
    r_peak, _ = find_peaks(cleaned_signal, prominence=0.25, distance=100)

    # discard signals that have too few r peaks detected
    if len(r_peak) < 15:
        print("patient:", patient_name)
        print("!!!! Enrollment not successful - non enough peaks")
        return []

    signal_dwt, waves_dwt = nk.ecg_delineate(cleaned_signal, rpeaks=r_peak, sampling_rate=512, method="dwt",
                                             show=False,
                                             show_type='peaks')

    t_peaks = k_nearest_neighbour_on_waves(waves_dwt['ECG_T_Peaks'])
    p_peaks = k_nearest_neighbour_on_waves(waves_dwt['ECG_P_Peaks'])
    q_peaks = k_nearest_neighbour_on_waves(waves_dwt['ECG_Q_Peaks'])
    s_peaks = k_nearest_neighbour_on_waves(waves_dwt['ECG_S_Peaks'])
    r_peak = k_nearest_neighbour_on_waves(r_peak)

    if len(t_peaks) == 0 or len(p_peaks) == 0 or len(q_peaks) == 0 or len(s_peaks) == 0 or len(r_peak) == 0:
        print("Signal not strong enough")
        return []

    Tx = mean(peak_widths(cleaned_signal, t_peaks))
    Px = mean(peak_widths(cleaned_signal, p_peaks))
    Qx = mean(peak_widths(cleaned_signal, q_peaks))
    Sx = mean(peak_widths(cleaned_signal, s_peaks))

    Ty = mean(peak_prominences(cleaned_signal, t_peaks))
    Py = mean(peak_prominences(cleaned_signal, p_peaks))
    Qy = mean(peak_prominences(cleaned_signal, q_peaks))
    Sy = mean(peak_prominences(cleaned_signal, s_peaks))

    final_peaks = []
    final_peaks.extend(p_peaks)
    final_peaks.extend(t_peaks)
    final_peaks.extend(q_peaks)
    final_peaks.extend(r_peak)
    final_peaks.extend(s_peaks)
    final_peaks.sort()

    # continue only if all peaks were extracted
    if len(final_peaks) % 5 != 0:
        print("patient:", patient_name)
        print("!!!! Enrollment not successful - incorrect number of peaks")
        return []

    features_time = [Tx, Px, Qx, Sx]
    features_time.extend(get_time(final_peaks, cleaned_signal))
    features_amplitude = [Ty, Py, Qy, Sy]
    features_amplitude.extend(get_amplitude(final_peaks, cleaned_signal))
    features_distance = get_distance(final_peaks, cleaned_signal)
    features_slope = get_slope(final_peaks, cleaned_signal)
    features_angle = get_angle(final_peaks, cleaned_signal)

    to_file = []
    to_file.append(patient_name.replace("\n", ""))
    to_file.extend(features_time)
    to_file.extend(features_amplitude)
    to_file.extend(features_distance)
    to_file.extend(features_slope)
    to_file.extend(features_angle)

    enroll_file.close()
    return to_file


def train_new_classifier(dataset, predictions):
    best_models = [n.name for n in Path('.').glob('*.joblib')]

    if len(best_models) == 0:
        print("No model found!")
        return

    if len(best_models) > 1:
        print("Too many models found!")
        return

    best_model = best_models[0]
    model = joblib.load(best_model)

    X = pd.read_csv(dataset)

    # encode categorical value
    enc = LabelEncoder()
    y = enc.fit_transform(X.pop('PATIENT_NAME'))
    np.save('classes.npy', enc.classes_)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=42)

    model.fit(X_train, y_train)
    print(model.score(X_train, y_train))

    joblib.dump(model, 'model.joblib', compress=3)

    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)

    y_pred = enc.inverse_transform(y_pred)
    y_test = enc.inverse_transform(y_test)

    new_df = pd.DataFrame(y_test, columns=['REAL'])
    new_df.insert(0, "PREDICTED", y_pred)
    new_df.insert(2, "SCORES", list(y_scores))

    new_df.to_csv(predictions, index=False)


def start_enrollment(dataset, balanced_dataset, analyzed_dataset, predictions):
    for p in Path('./enrollements/').glob('*.csv'):
        wr_process_processing("enrollements/" + p.name, dataset)
        os.remove("enrollements/" + p.name)

    analyze_dataset(analyzed_dataset, balanced_dataset, dataset)
    train_new_classifier(balanced_dataset, predictions)
