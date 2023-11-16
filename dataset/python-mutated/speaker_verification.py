import os
import pickle
import wave
import numpy as np
import pyaudio
import python_speech_features as mfcc
from datetime import datetime
from scipy.io.wavfile import read
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from util import record_audio, speech_to_text

def calculate_delta(array):
    if False:
        print('Hello World!')
    (rows, cols) = array.shape
    deltas = np.zeros((rows, 20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i - j < 0:
                first = 0
            else:
                first = i - j
            if i + j > rows - 1:
                second = rows - 1
            else:
                second = i + j
            index.append((second, first))
            j += 1
        deltas[i] = (array[index[0][0]] - array[index[0][1]] + 2 * (array[index[1][0]] - array[index[1][1]])) / 10
    return deltas

def extract_features(audio, rate):
    if False:
        print('Hello World!')
    mfcc_feature = mfcc.mfcc(audio, rate, 0.025, 0.01, 20, nfft=1200, appendEnergy=True)
    mfcc_feature = preprocessing.scale(mfcc_feature)
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature, delta))
    return combined

def record_sample(name):
    if False:
        print('Hello World!')
    for count in range(5):
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        CHUNK = 1024
        RECORD_SECONDS = 20
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=2, frames_per_buffer=CHUNK)
        Recordframes = []
        print('----------------------Recording ' + f'{count + 1}---------------------')
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            Recordframes.append(data)
        print('--------------------------Recording Stopped-------------------------')
        stream.stop_stream()
        stream.close()
        audio.terminate()
        OUTPUT_FILENAME = name + '-1-' + str(count) + '.wav'
        WAVE_OUTPUT_FILENAME = os.path.join('training_data/voice/' + name, OUTPUT_FILENAME)
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(Recordframes))
        waveFile.close()

def train_model():
    if False:
        print('Hello World!')
    src = os.listdir('C:\\Users\\nuc\\Desktop\\Smart-Device-DT\\AI_System\\training_data\\voice\\')
    dest = 'models/voice/'
    count = 1
    features = np.asarray(())
    for (i, path) in enumerate(src):
        sub_src = os.listdir('C:\\Users\\nuc\\Desktop\\Smart-Device-DT\\AI_System\\training_data\\voice\\' + path)
        for (j, sub_path) in enumerate(sub_src):
            (sr, audio) = read('D:\\Smart-Device-Controller-System\\training_data\\voice\\' + path + '\\' + sub_path)
            vector = extract_features(audio, sr)
            if features.size == 0:
                features = vector
            else:
                features = np.vstack((features, vector))
            if count == 22:
                gmm = GaussianMixture(n_components=22, max_iter=200, covariance_type='diag', n_init=3)
                gmm.fit(features)
                picklefile = path.split('-')[0] + '.gmm'
                pickle.dump(gmm, open(dest + picklefile, 'wb'))
                print('Modelling completed for speaker:', picklefile, 'with data point =', features.shape)
                features = np.asarray(())
                count = 0
            count += 1

def test_model(audio):
    if False:
        print('Hello World!')
    src = 'models/voice/'
    gmm_files = [os.path.join(src, fname) for fname in os.listdir(src) if fname.endswith('.gmm')]
    models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]
    speakers = [fname.split('\\')[-1].split('.gmm')[0] for fname in gmm_files]
    (sr, audio) = read(audio)
    vector = extract_features(audio, sr)
    log_likelihood = np.zeros(len(models))
    for i in range(len(models)):
        gmm = models[i]
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()
    winner_score = np.max(log_likelihood)
    winner = np.argmax(log_likelihood)
    winner_name = speakers[winner][13:]
    if winner_score <= -21.5:
        winner_name = 'Unknown'
    print(winner_name, winner_score, winner, log_likelihood)
    return winner_name

def evaluate_model():
    if False:
        i = 10
        return i + 15
    test_data = os.listdir('D:\\Smart-Device-Controller-System\\testing_data\\voice')
    for (i, path) in enumerate(test_data):
        sub_src = os.listdir('D:\\Smart-Device-Controller-System\\testing_data\\voice\\' + path)
        for (j, sub_path) in enumerate(sub_src):
            print('Data: ' + path + ', Label: ' + test_model('D:\\Smart-Device-Controller-System\\testing_data\\voice\\' + path + '\\' + sub_path))

def train_model_1_person(name):
    if False:
        for i in range(10):
            print('nop')
    src = os.listdir('C:\\Users\\nuc\\Desktop\\Smart-Device-DT\\AI_System\\training_data\\voice\\' + name)
    dest = 'models/voice/'
    count = 1
    features = np.asarray(())
    for (j, sub_path) in enumerate(src):
        (sr, audio) = read('C:\\Users\\nuc\\Desktop\\Smart-Device-DT\\AI_System\\training_data\\voice\\' + name + '\\' + sub_path)
        vector = extract_features(audio, sr)
        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))
        if count == 15:
            gmm = GaussianMixture(n_components=15, max_iter=200, covariance_type='diag', n_init=3)
            gmm.fit(features)
            picklefile = name + '.gmm'
            pickle.dump(gmm, open(dest + picklefile, 'wb'))
            print('Modelling completed for speaker:', picklefile, 'with data point =', features.shape)
            features = np.asarray(())
            count = 0
        count += 1