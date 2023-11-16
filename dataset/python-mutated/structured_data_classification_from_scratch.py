"""
Title: Structured data classification from scratch
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/06/09
Last modified: 2020/06/09
Description: Binary classification of structured data including numerical and categorical features.
Accelerator: GPU
"""
"\n## Introduction\n\nThis example demonstrates how to do structured data classification, starting from a raw\nCSV file. Our data includes both numerical and categorical features. We will use Keras\npreprocessing layers to normalize the numerical features and vectorize the categorical\nones.\n\nNote that this example should be run with TensorFlow 2.5 or higher.\n\n### The dataset\n\n[Our dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease) is provided by the\nCleveland Clinic Foundation for Heart Disease.\nIt's a CSV file with 303 rows. Each row contains information about a patient (a\n**sample**), and each column describes an attribute of the patient (a **feature**). We\nuse the features to predict whether a patient has a heart disease (**binary\nclassification**).\n\nHere's the description of each feature:\n\nColumn| Description| Feature Type\n------------|--------------------|----------------------\nAge | Age in years | Numerical\nSex | (1 = male; 0 = female) | Categorical\nCP | Chest pain type (0, 1, 2, 3, 4) | Categorical\nTrestbpd | Resting blood pressure (in mm Hg on admission) | Numerical\nChol | Serum cholesterol in mg/dl | Numerical\nFBS | fasting blood sugar in 120 mg/dl (1 = true; 0 = false) | Categorical\nRestECG | Resting electrocardiogram results (0, 1, 2) | Categorical\nThalach | Maximum heart rate achieved | Numerical\nExang | Exercise induced angina (1 = yes; 0 = no) | Categorical\nOldpeak | ST depression induced by exercise relative to rest | Numerical\nSlope | Slope of the peak exercise ST segment | Numerical\nCA | Number of major vessels (0-3) colored by fluoroscopy | Both numerical & categorical\nThal | 3 = normal; 6 = fixed defect; 7 = reversible defect | Categorical\nTarget | Diagnosis of heart disease (1 = true; 0 = false) | Target\n"
'\n## Setup\n'
import tensorflow as tf
import pandas as pd
import keras
from keras import layers
"\n## Preparing the data\n\nLet's download the data and load it into a Pandas dataframe:\n"
file_url = 'http://storage.googleapis.com/download.tensorflow.org/data/heart.csv'
dataframe = pd.read_csv(file_url)
'\nThe dataset includes 303 samples with 14 columns per sample (13 features, plus the target\nlabel):\n'
dataframe.shape
"\nHere's a preview of a few samples:\n"
dataframe.head()
'\nThe last column, "target", indicates whether the patient has a heart disease (1) or not\n(0).\n\nLet\'s split the data into a training and validation set:\n'
val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
train_dataframe = dataframe.drop(val_dataframe.index)
print(f'Using {len(train_dataframe)} samples for training and {len(val_dataframe)} for validation')
"\nLet's generate `tf.data.Dataset` objects for each dataframe:\n"

def dataframe_to_dataset(dataframe):
    if False:
        print('Hello World!')
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds
train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)
'\nEach `Dataset` yields a tuple `(input, target)` where `input` is a dictionary of features\nand `target` is the value `0` or `1`:\n'
for (x, y) in train_ds.take(1):
    print('Input:', x)
    print('Target:', y)
"\nLet's batch the datasets:\n"
train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)
'\n## Feature preprocessing with Keras layers\n\n\nThe following features are categorical features encoded as integers:\n\n- `sex`\n- `cp`\n- `fbs`\n- `restecg`\n- `exang`\n- `ca`\n\nWe will encode these features using **one-hot encoding**. We have two options\nhere:\n\n - Use `CategoryEncoding()`, which requires knowing the range of input values\n and will error on input outside the range.\n - Use `IntegerLookup()` which will build a lookup table for inputs and reserve\n an output index for unkown input values.\n\nFor this example, we want a simple solution that will handle out of range inputs\nat inference, so we will use `IntegerLookup()`.\n\nWe also have a categorical feature encoded as a string: `thal`. We will create an\nindex of all possible features and encode output using the `StringLookup()` layer.\n\nFinally, the following feature are continuous numerical features:\n\n- `age`\n- `trestbps`\n- `chol`\n- `thalach`\n- `oldpeak`\n- `slope`\n\nFor each of these features, we will use a `Normalization()` layer to make sure the mean\nof each feature is 0 and its standard deviation is 1.\n\nBelow, we define 3 utility functions to do the operations:\n\n- `encode_numerical_feature` to apply featurewise normalization to numerical features.\n- `encode_string_categorical_feature` to first turn string inputs into integer indices,\nthen one-hot encode these integer indices.\n- `encode_integer_categorical_feature` to one-hot encode integer categorical features.\n'
from keras.layers import IntegerLookup
from keras.layers import Normalization
from keras.layers import StringLookup

def encode_numerical_feature(feature, name, dataset):
    if False:
        return 10
    normalizer = Normalization()
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))
    normalizer.adapt(feature_ds)
    encoded_feature = normalizer(feature)
    return encoded_feature

def encode_categorical_feature(feature, name, dataset, is_string):
    if False:
        i = 10
        return i + 15
    lookup_class = StringLookup if is_string else IntegerLookup
    lookup = lookup_class(output_mode='binary')
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))
    lookup.adapt(feature_ds)
    encoded_feature = lookup(feature)
    return encoded_feature
'\n## Build a model\n\nWith this done, we can create our end-to-end model:\n'
sex = keras.Input(shape=(1,), name='sex', dtype='int64')
cp = keras.Input(shape=(1,), name='cp', dtype='int64')
fbs = keras.Input(shape=(1,), name='fbs', dtype='int64')
restecg = keras.Input(shape=(1,), name='restecg', dtype='int64')
exang = keras.Input(shape=(1,), name='exang', dtype='int64')
ca = keras.Input(shape=(1,), name='ca', dtype='int64')
thal = keras.Input(shape=(1,), name='thal', dtype='string')
age = keras.Input(shape=(1,), name='age')
trestbps = keras.Input(shape=(1,), name='trestbps')
chol = keras.Input(shape=(1,), name='chol')
thalach = keras.Input(shape=(1,), name='thalach')
oldpeak = keras.Input(shape=(1,), name='oldpeak')
slope = keras.Input(shape=(1,), name='slope')
all_inputs = [sex, cp, fbs, restecg, exang, ca, thal, age, trestbps, chol, thalach, oldpeak, slope]
sex_encoded = encode_categorical_feature(sex, 'sex', train_ds, False)
cp_encoded = encode_categorical_feature(cp, 'cp', train_ds, False)
fbs_encoded = encode_categorical_feature(fbs, 'fbs', train_ds, False)
restecg_encoded = encode_categorical_feature(restecg, 'restecg', train_ds, False)
exang_encoded = encode_categorical_feature(exang, 'exang', train_ds, False)
ca_encoded = encode_categorical_feature(ca, 'ca', train_ds, False)
thal_encoded = encode_categorical_feature(thal, 'thal', train_ds, True)
age_encoded = encode_numerical_feature(age, 'age', train_ds)
trestbps_encoded = encode_numerical_feature(trestbps, 'trestbps', train_ds)
chol_encoded = encode_numerical_feature(chol, 'chol', train_ds)
thalach_encoded = encode_numerical_feature(thalach, 'thalach', train_ds)
oldpeak_encoded = encode_numerical_feature(oldpeak, 'oldpeak', train_ds)
slope_encoded = encode_numerical_feature(slope, 'slope', train_ds)
all_features = layers.concatenate([sex_encoded, cp_encoded, fbs_encoded, restecg_encoded, exang_encoded, slope_encoded, ca_encoded, thal_encoded, age_encoded, trestbps_encoded, chol_encoded, thalach_encoded, oldpeak_encoded])
x = layers.Dense(32, activation='relu')(all_features)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(all_inputs, output)
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
"\nLet's visualize our connectivity graph:\n"
keras.utils.plot_model(model, show_shapes=True, rankdir='LR')
'\n## Train the model\n'
model.fit(train_ds, epochs=50, validation_data=val_ds)
'\nWe quickly get to 80% validation accuracy.\n'
'\n## Inference on new data\n\nTo get a prediction for a new sample, you can simply call `model.predict()`. There are\njust two things you need to do:\n\n1. wrap scalars into a list so as to have a batch dimension (models only process batches\nof data, not single samples)\n2. Call `convert_to_tensor` on each feature\n'
sample = {'age': 60, 'sex': 1, 'cp': 1, 'trestbps': 145, 'chol': 233, 'fbs': 1, 'restecg': 2, 'thalach': 150, 'exang': 0, 'oldpeak': 2.3, 'slope': 3, 'ca': 0, 'thal': 'fixed'}
input_dict = {name: tf.convert_to_tensor([value]) for (name, value) in sample.items()}
predictions = model.predict(input_dict)
print(f'This particular patient had a {100 * predictions[0][0]:.1f} percent probability of having a heart disease, as evaluated by our model.')