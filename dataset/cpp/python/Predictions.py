from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from Enrollement import process_new_data


def predict_class():
    best_models = [n.name for n in Path('.').glob('*.joblib')]

    if len(best_models) == 0:
        print("No model found!")
        return

    if len(best_models) > 1:
        print("Too many models found!")
        return

    best_model = best_models[0]
    model = joblib.load(best_model)

    process_new_data()

    data_to_predict = pd.read_csv('classify/to_predict.csv')

    y = data_to_predict.pop('PATIENT_NAME')
    X = data_to_predict.copy()

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    enc = LabelEncoder()
    enc.classes_ = np.load('classes.npy', allow_pickle=True)

    imputer = KNNImputer()
    X = imputer.fit_transform(X)

    y_pred = model.predict(X)
    y_pred = enc.inverse_transform(y_pred)

    i = 0
    while i < len(y):
        if y_pred[i] == y[i]:
            print("CORRECTLY RECOGNIZED:", y_pred[i])
        else:
            print("NOT RECOGNIZED:", "PREDICTED", y_pred[i], "INSTEAD OF", y[i])
        i += 1

    f = open("classify/to_predict.csv", "w")
    f.truncate()
    f.close()