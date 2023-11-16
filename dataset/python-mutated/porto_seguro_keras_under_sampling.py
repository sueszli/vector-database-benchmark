"""
==========================================================
Porto Seguro: balancing samples in mini-batches with Keras
==========================================================

This example compares two strategies to train a neural-network on the Porto
Seguro Kaggle data set [1]_. The data set is imbalanced and we show that
balancing each mini-batch allows to improve performance and reduce the training
time.

References
----------

.. [1] https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data

"""
print(__doc__)
from collections import Counter
import numpy as np
import pandas as pd
training_data = pd.read_csv('./input/train.csv')
testing_data = pd.read_csv('./input/test.csv')
y_train = training_data[['id', 'target']].set_index('id')
X_train = training_data.drop(['target'], axis=1).set_index('id')
X_test = testing_data.set_index('id')
print(f"The data set is imbalanced: {Counter(y_train['target'])}")
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

def convert_float64(X):
    if False:
        return 10
    return X.astype(np.float64)
numerical_columns = [name for name in X_train.columns if '_calc_' in name and '_bin' not in name]
numerical_pipeline = make_pipeline(FunctionTransformer(func=convert_float64, validate=False), StandardScaler())
categorical_columns = [name for name in X_train.columns if '_cat' in name]
categorical_pipeline = make_pipeline(SimpleImputer(missing_values=-1, strategy='most_frequent'), OneHotEncoder(categories='auto'))
preprocessor = ColumnTransformer([('numerical_preprocessing', numerical_pipeline, numerical_columns), ('categorical_preprocessing', categorical_pipeline, categorical_columns)], remainder='drop')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential

def make_model(n_features):
    if False:
        print('Hello World!')
    model = Sequential()
    model.add(Dense(200, input_shape=(n_features,), kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, kernel_initializer='glorot_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(50, kernel_initializer='glorot_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.15))
    model.add(Dense(25, kernel_initializer='glorot_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
import time
from functools import wraps

def timeit(f):
    if False:
        print('Hello World!')

    @wraps(f)
    def wrapper(*args, **kwds):
        if False:
            while True:
                i = 10
        start_time = time.time()
        result = f(*args, **kwds)
        elapsed_time = time.time() - start_time
        print(f'Elapsed computation time: {elapsed_time:.3f} secs')
        return (elapsed_time, result)
    return wrapper
import tensorflow
from sklearn.metrics import roc_auc_score
from sklearn.utils import parse_version
tf_version = parse_version(tensorflow.__version__)

@timeit
def fit_predict_imbalanced_model(X_train, y_train, X_test, y_test):
    if False:
        print('Hello World!')
    model = make_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=2, verbose=1, batch_size=1000)
    if tf_version < parse_version('2.6'):
        predict_method = 'predict_proba'
    else:
        predict_method = 'predict'
    y_pred = getattr(model, predict_method)(X_test, batch_size=1000)
    return roc_auc_score(y_test, y_pred)
from imblearn.keras import BalancedBatchGenerator

@timeit
def fit_predict_balanced_model(X_train, y_train, X_test, y_test):
    if False:
        i = 10
        return i + 15
    model = make_model(X_train.shape[1])
    training_generator = BalancedBatchGenerator(X_train, y_train, batch_size=1000, random_state=42)
    model.fit(training_generator, epochs=5, verbose=1)
    y_pred = model.predict(X_test, batch_size=1000)
    return roc_auc_score(y_test, y_pred)
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10)
cv_results_imbalanced = []
cv_time_imbalanced = []
cv_results_balanced = []
cv_time_balanced = []
for (train_idx, valid_idx) in skf.split(X_train, y_train):
    X_local_train = preprocessor.fit_transform(X_train.iloc[train_idx])
    y_local_train = y_train.iloc[train_idx].values.ravel()
    X_local_test = preprocessor.transform(X_train.iloc[valid_idx])
    y_local_test = y_train.iloc[valid_idx].values.ravel()
    (elapsed_time, roc_auc) = fit_predict_imbalanced_model(X_local_train, y_local_train, X_local_test, y_local_test)
    cv_time_imbalanced.append(elapsed_time)
    cv_results_imbalanced.append(roc_auc)
    (elapsed_time, roc_auc) = fit_predict_balanced_model(X_local_train, y_local_train, X_local_test, y_local_test)
    cv_time_balanced.append(elapsed_time)
    cv_results_balanced.append(roc_auc)
df_results = pd.DataFrame({'Balanced model': cv_results_balanced, 'Imbalanced model': cv_results_imbalanced})
df_results = df_results.unstack().reset_index()
df_time = pd.DataFrame({'Balanced model': cv_time_balanced, 'Imbalanced model': cv_time_imbalanced})
df_time = df_time.unstack().reset_index()
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure()
sns.boxplot(y='level_0', x=0, data=df_time)
sns.despine(top=True, right=True, left=True)
plt.xlabel('time [s]')
plt.ylabel('')
plt.title('Computation time difference using a random under-sampling')
plt.figure()
sns.boxplot(y='level_0', x=0, data=df_results, whis=10.0)
sns.despine(top=True, right=True, left=True)
ax = plt.gca()
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: '%i%%' % (100 * x)))
plt.xlabel('ROC-AUC')
plt.ylabel('')
plt.title('Difference in terms of ROC-AUC using a random under-sampling')