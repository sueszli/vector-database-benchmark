import math
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
import wfdb
from imblearn.over_sampling import RandomOverSampler
from numpy import mean
from scipy.signal import lfilter, find_peaks, peak_widths, peak_prominences
from scipy.spatial import distance
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from Filters import HighPassFilter, BandStopFilter, LowPassFilter, SmoothSignal

def getAngle(a, b, c):
    if False:
        return 10
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang + 360 if ang < 0 else ang

def get_time(final_peaks, cleaned_signal):
    if False:
        while True:
            i = 10
    PQ_time_list = []
    PT_time_list = []
    QS_time_list = []
    QT_time_list = []
    ST_time_list = []
    PS_time_list = []
    for i in range(0, len(final_peaks), 5):
        P = (final_peaks[i], cleaned_signal[final_peaks[i]])
        Q = (final_peaks[i + 1], cleaned_signal[final_peaks[i + 1]])
        R = (final_peaks[i + 2], cleaned_signal[final_peaks[i + 2]])
        S = (final_peaks[i + 3], cleaned_signal[final_peaks[i + 3]])
        T = (final_peaks[i + 4], cleaned_signal[final_peaks[i + 4]])
        PQ_time_list.append(abs(P[0] - Q[0]))
        PT_time_list.append(abs(P[0] - T[0]))
        QS_time_list.append(abs(Q[0] - S[0]))
        QT_time_list.append(abs(Q[0] - T[0]))
        ST_time_list.append(abs(S[0] - T[0]))
        PS_time_list.append(abs(P[0] - S[0]))
    PQ_time = mean(PQ_time_list)
    PT_time = mean(PT_time_list)
    QS_time = mean(QS_time_list)
    QT_time = mean(QT_time_list)
    ST_time = mean(ST_time_list)
    PS_time = mean(PS_time_list)
    PQ_QS_time = PT_time / QS_time
    QT_QS_time = QT_time / QS_time
    return [PQ_time, PT_time, QS_time, QT_time, ST_time, PS_time, PQ_QS_time, QT_QS_time]

def get_amplitude(final_peaks, cleaned_signal):
    if False:
        i = 10
        return i + 15
    PQ_ampl_list = []
    QR_ampl_list = []
    RS_ampl_list = []
    ST_ampl_list = []
    QS_ampl_list = []
    PS_ampl_list = []
    PT_ampl_list = []
    QT_ampl_list = []
    for i in range(0, len(final_peaks), 5):
        P = (final_peaks[i], cleaned_signal[final_peaks[i]])
        Q = (final_peaks[i + 1], cleaned_signal[final_peaks[i + 1]])
        R = (final_peaks[i + 2], cleaned_signal[final_peaks[i + 2]])
        S = (final_peaks[i + 3], cleaned_signal[final_peaks[i + 3]])
        T = (final_peaks[i + 4], cleaned_signal[final_peaks[i + 4]])
        PQ_ampl_list.append(abs(P[1] - Q[1]))
        QR_ampl_list.append(abs(Q[1] - R[1]))
        RS_ampl_list.append(abs(R[1] - S[1]))
        ST_ampl_list.append(abs(S[1] - T[1]))
        QS_ampl_list.append(abs(Q[1] - S[1]))
        PS_ampl_list.append(abs(P[1] - S[1]))
        PT_ampl_list.append(abs(P[1] - T[1]))
        QT_ampl_list.append(abs(Q[1] - T[1]))
    PQ_ampl = mean(PQ_ampl_list)
    QR_ampl = mean(QR_ampl_list)
    RS_ampl = mean(RS_ampl_list)
    QS_ampl = mean(QS_ampl_list)
    ST_ampl = mean(ST_ampl_list)
    PS_ampl = mean(PS_ampl_list)
    PT_ampl = mean(PT_ampl_list)
    QT_ampl = mean(QT_ampl_list)
    ST_QS_ampl = ST_ampl / QS_ampl
    RS_QR_ampl = RS_ampl / QR_ampl
    PQ_QS_ampl = PQ_ampl / QS_ampl
    PQ_QT_ampl = PQ_ampl / QT_ampl
    PQ_PS_ampl = PQ_ampl / PS_ampl
    PQ_QR_ampl = PQ_ampl / QR_ampl
    PQ_RS_ampl = PQ_ampl / RS_ampl
    RS_QS_ampl = RS_ampl / QS_ampl
    RS_QT_ampl = RS_ampl / QT_ampl
    ST_PQ_ampl = ST_ampl / PQ_ampl
    ST_QT_ampl = ST_ampl / QT_ampl
    return [PQ_ampl, QR_ampl, RS_ampl, QS_ampl, ST_ampl, PS_ampl, PT_ampl, QT_ampl, ST_QS_ampl, RS_QR_ampl, PQ_QS_ampl, PQ_QT_ampl, PQ_PS_ampl, PQ_QR_ampl, PQ_RS_ampl, RS_QS_ampl, RS_QT_ampl, ST_PQ_ampl, ST_QT_ampl]

def get_distance(final_peaks, cleaned_signal):
    if False:
        return 10
    PQ_dist_list = []
    QR_dist_list = []
    RS_dist_list = []
    ST_dist_list = []
    QS_dist_list = []
    PR_dist_list = []
    for i in range(0, len(final_peaks), 5):
        P = (final_peaks[i], cleaned_signal[final_peaks[i]])
        Q = (final_peaks[i + 1], cleaned_signal[final_peaks[i + 1]])
        R = (final_peaks[i + 2], cleaned_signal[final_peaks[i + 2]])
        S = (final_peaks[i + 3], cleaned_signal[final_peaks[i + 3]])
        T = (final_peaks[i + 4], cleaned_signal[final_peaks[i + 4]])
        PQ_dist_list.append(distance.euclidean(P, Q))
        QR_dist_list.append(distance.euclidean(Q, R))
        RS_dist_list.append(distance.euclidean(R, S))
        ST_dist_list.append(distance.euclidean(S, T))
        QS_dist_list.append(distance.euclidean(Q, S))
        PR_dist_list.append(distance.euclidean(P, R))
    PQ_dist = mean(PQ_dist_list)
    QR_dist = mean(QR_dist_list)
    RS_dist = mean(RS_dist_list)
    ST_dist = mean(ST_dist_list)
    QS_dist = mean(QS_dist_list)
    PR_dist = mean(PR_dist_list)
    ST_QS_dist = ST_dist / QS_dist
    RS_QR_dist = RS_dist / QR_dist
    return [PQ_dist, QR_dist, RS_dist, ST_dist, QS_dist, PR_dist, ST_QS_dist, RS_QR_dist]

def get_slope(final_peaks, cleaned_signal):
    if False:
        return 10
    PQ_slope_list = []
    QR_slope_list = []
    RS_slope_list = []
    ST_slope_list = []
    QS_slope_list = []
    PT_slope_list = []
    PS_slope_list = []
    QT_slope_list = []
    PR_slope_list = []
    for i in range(0, len(final_peaks), 5):
        P = (final_peaks[i], cleaned_signal[final_peaks[i]])
        Q = (final_peaks[i + 1], cleaned_signal[final_peaks[i + 1]])
        R = (final_peaks[i + 2], cleaned_signal[final_peaks[i + 2]])
        S = (final_peaks[i + 3], cleaned_signal[final_peaks[i + 3]])
        T = (final_peaks[i + 4], cleaned_signal[final_peaks[i + 4]])
        PQ_slope_list.append((Q[1] - P[1]) / (Q[0] - P[0]))
        QR_slope_list.append((R[1] - Q[1]) / (R[0] - Q[0]))
        RS_slope_list.append((S[1] - R[1]) / (S[0] - R[0]))
        ST_slope_list.append((T[1] - S[1]) / (T[0] - S[0]))
        QS_slope_list.append((S[1] - Q[1]) / (S[0] - Q[0]))
        PT_slope_list.append((T[1] - P[1]) / (T[0] - P[0]))
        PS_slope_list.append((S[1] - P[1]) / (S[0] - P[0]))
        QT_slope_list.append((T[1] - Q[1]) / (T[0] - Q[0]))
        PR_slope_list.append((R[1] - P[1]) / (R[0] - P[0]))
    PQ_slope = mean(PQ_slope_list)
    QR_slope = mean(QR_slope_list)
    RS_slope = mean(RS_slope_list)
    ST_slope = mean(ST_slope_list)
    QS_slope = mean(QS_slope_list)
    PT_slope = mean(PT_slope_list)
    PS_slope = mean(PS_slope_list)
    QT_slope = mean(QT_slope_list)
    PR_slope = mean(PR_slope_list)
    return [PQ_slope, QR_slope, RS_slope, ST_slope, QS_slope, PT_slope, PS_slope, QT_slope, PR_slope]

def get_angle(final_peaks, cleaned_signal):
    if False:
        for i in range(10):
            print('nop')
    PQR_angle_list = []
    QRS_angle_list = []
    RST_angle_list = []
    RQS_angle_list = []
    RSQ_angle_list = []
    RTS_angle_list = []
    for i in range(0, len(final_peaks), 5):
        P = (final_peaks[i], cleaned_signal[final_peaks[i]])
        Q = (final_peaks[i + 1], cleaned_signal[final_peaks[i + 1]])
        R = (final_peaks[i + 2], cleaned_signal[final_peaks[i + 2]])
        S = (final_peaks[i + 3], cleaned_signal[final_peaks[i + 3]])
        T = (final_peaks[i + 4], cleaned_signal[final_peaks[i + 4]])
        PQR_angle_list.append(getAngle(P, Q, R))
        QRS_angle_list.append(getAngle(Q, R, S))
        RST_angle_list.append(getAngle(R, S, T))
        RQS_angle_list.append(getAngle(R, Q, S))
        RSQ_angle_list.append(getAngle(R, S, Q))
        RTS_angle_list.append(getAngle(R, T, S))
    PQR_angle = mean(PQR_angle_list)
    QRS_angle = mean(QRS_angle_list)
    RST_angle = mean(RST_angle_list)
    RQS_angle = mean(RQS_angle_list)
    RSQ_angle = mean(RSQ_angle_list)
    RTS_angle = mean(RTS_angle_list)
    return [PQR_angle, QRS_angle, RST_angle, RQS_angle, RSQ_angle, RTS_angle]

def k_nearest_neighbour_on_waves(data):
    if False:
        print('Hello World!')
    imputer = KNNImputer(n_neighbors=2)
    waves_without_nan = imputer.fit_transform(np.reshape(data, (-1, 1)))
    return waves_without_nan.astype(int).ravel()

def create_dataset(base_path, new_dataset):
    if False:
        i = 10
        return i + 15
    patient_names = pd.read_fwf('ptb-diagnostic-ecg-database-1.0.0/RECORDS', dtype=str)
    headers = ['PATIENT_NAME', 'Tx', 'Px', 'Qx', 'Sx', 'PQ_time', 'PT_time', 'QS_time', 'QT_time', 'ST_time', 'PS_time', 'PQ_QS_time', 'QT_QS_time', 'Ty', 'Py', 'Qy', 'Sy', 'PQ_ampl', 'QR_ampl', 'RS_ampl', 'QS_ampl', 'ST_ampl', 'PS_ampl', 'PT_ampl', 'QT_ampl', 'ST_QS_ampl', 'RS_QR_ampl', 'PQ_QS_ampl', 'PQ_QT_ampl', 'PQ_PS_ampl', 'PQ_QR_ampl', 'PQ_RS_ampl', 'RS_QS_ampl', 'RS_QT_ampl', 'ST_PQ_ampl', 'ST_QT_ampl', 'PQ_dist', 'QR_dist', 'RS_dist', 'ST_dist', 'QS_dist', 'PR_dist', 'ST_QS_dist', 'RS_QR_dist', 'PQ_slope', 'QR_slope', 'RS_slope', 'ST_slope', 'QS_slope', 'PT_slope', 'PS_slope', 'QT_slope', 'PR_slope', 'PQR_angle', 'QRS_angle', 'RST_angle', 'RQS_angle', 'RSQ_angle', 'RTS_angle']
    df = pd.DataFrame(columns=headers)
    for i in range(patient_names.size):
        patient_id = str(patient_names['PATIENTS'][i])
        path = base_path + patient_id
        record = wfdb.rdrecord(path, channel_names=['v4'])
        signal = record.p_signal.ravel()
        denoised_ecg = lfilter(HighPassFilter(), 1, signal)
        denoised_ecg = lfilter(BandStopFilter(), 1, denoised_ecg)
        denoised_ecg = lfilter(LowPassFilter(), 1, denoised_ecg)
        cleaned_signal = SmoothSignal(denoised_ecg)
        (r_peak, _) = find_peaks(cleaned_signal, prominence=1, distance=100)
        if len(r_peak) < 15:
            continue
        (signal_dwt, waves_dwt) = nk.ecg_delineate(cleaned_signal, rpeaks=r_peak, sampling_rate=1000, method='dwt', show=False, show_type='peaks')
        t_peaks = k_nearest_neighbour_on_waves(waves_dwt['ECG_T_Peaks'])
        p_peaks = k_nearest_neighbour_on_waves(waves_dwt['ECG_P_Peaks'])
        q_peaks = k_nearest_neighbour_on_waves(waves_dwt['ECG_Q_Peaks'])
        s_peaks = k_nearest_neighbour_on_waves(waves_dwt['ECG_S_Peaks'])
        r_peak = k_nearest_neighbour_on_waves(r_peak)
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
        if len(final_peaks) % 5 != 0:
            continue
        features_time = [Tx, Px, Qx, Sx]
        features_time.extend(get_time(final_peaks, cleaned_signal))
        features_amplitude = [Ty, Py, Qy, Sy]
        features_amplitude.extend(get_amplitude(final_peaks, cleaned_signal))
        features_distance = get_distance(final_peaks, cleaned_signal)
        features_slope = get_slope(final_peaks, cleaned_signal)
        features_angle = get_angle(final_peaks, cleaned_signal)
        to_file = []
        to_file.append(patient_id.split('/')[0])
        to_file.extend(features_time)
        to_file.extend(features_amplitude)
        to_file.extend(features_distance)
        to_file.extend(features_slope)
        to_file.extend(features_angle)
        patient_datas = pd.Series(to_file, index=df.columns)
        df = df.append(patient_datas, ignore_index=True)
    df.to_csv(new_dataset, index=False)

def plot_classes(dataset):
    if False:
        print('Hello World!')
    df = pd.read_csv(dataset)
    print(df['PATIENT_NAME'].value_counts())
    plt.title('Classes distribution')
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.hist(df['PATIENT_NAME'], bins=50)
    plt.xlabel('patients')
    plt.ylabel('Number of instances')
    plt.savefig('plot/classes_balancement_before.svg', dpi=1200)
    plt.tight_layout()
    plt.clf()

def balance_dataset(dataset, balanced_dataset):
    if False:
        i = 10
        return i + 15
    df = pd.read_csv(dataset)
    y = df.pop('PATIENT_NAME')
    X = df
    headers = X.columns
    oversample = RandomOverSampler(random_state=42)
    (X, y) = oversample.fit_resample(X, y)
    new_df = pd.DataFrame(X, columns=headers)
    new_df.insert(0, 'PATIENT_NAME', y)
    print(new_df['PATIENT_NAME'].value_counts())
    new_df.to_csv(balanced_dataset, index=False)

def feature_importance_analysis(dataset, analyzed_dataset):
    if False:
        i = 10
        return i + 15
    df = pd.read_csv(dataset)
    X = df.copy()
    y = X.pop('PATIENT_NAME')
    feature_names = X.columns
    enc = LabelEncoder()
    y = enc.fit_transform(y)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, shuffle=True, test_size=0.3)
    forest = RandomForestRegressor()
    forest.fit(X_train, y_train)
    print(f'model score on training data: {forest.score(X_train, y_train)}')
    result = permutation_importance(forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
    forest_importances = pd.Series(result.importances_mean, index=feature_names)
    (fig, ax) = plt.subplots(figsize=(10, 20))
    forest_importances.plot.barh(yerr=result.importances_std, ax=ax, log=True)
    ax.set_title('Feature importances using permutation on full model')
    ax.set_xlabel('Mean accuracy decrease')
    plt.grid(which='both')
    fig.tight_layout()
    fig.savefig('plot/feature_importance.svg', dpi=1200)
    plt.clf()
    features = []
    for (i, imp) in enumerate(result.importances_std):
        if imp < 0.001:
            features.append(feature_names[i])
    for feat in features:
        df = df.drop(feat, axis=1)
    df.to_csv(analyzed_dataset, index=False)

def remove_nan(dataset):
    if False:
        i = 10
        return i + 15
    df = pd.read_csv(dataset)
    y = df.pop('PATIENT_NAME')
    X = df
    headers = X.columns
    imputer = KNNImputer()
    X = imputer.fit_transform(X)
    new_df = pd.DataFrame(X, columns=headers)
    new_df.insert(0, 'PATIENT_NAME', y)
    new_df.to_csv(dataset, index=False)

def analyze_dataset(analyzed_dataset, balanced_dataset, dataset):
    if False:
        print('Hello World!')
    remove_nan(dataset)
    feature_importance_analysis(dataset, analyzed_dataset)
    balance_dataset(analyzed_dataset, balanced_dataset)

def data_processing(base_path, dataset, balanced_dataset, analyzed_dataset):
    if False:
        for i in range(10):
            print('nop')
    create_dataset(base_path, dataset)
    plot_classes(dataset)
    analyze_dataset(analyzed_dataset, balanced_dataset, dataset)