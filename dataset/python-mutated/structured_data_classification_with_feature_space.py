"""
Title: Structured data classification with FeatureSpace
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2022/11/09
Last modified: 2022/11/09
Description: Classify tabular data in a few lines of code.
Accelerator: GPU
"""
"\n## Introduction\n\nThis example demonstrates how to do structured data classification\n(also known as tabular data classification), starting from a raw\nCSV file. Our data includes numerical features,\nand integer categorical features, and string categorical features.\nWe will use the utility `keras.utils.FeatureSpace` to index,\npreprocess, and encode our features.\n\nThe code is adapted from the example\n[Structured data classification from scratch](https://keras.io/examples/structured_data/structured_data_classification_from_scratch/).\nWhile the previous example managed its own low-level feature preprocessing and\nencoding with Keras preprocessing layers, in this example we\ndelegate everything to `FeatureSpace`, making the workflow\nextremely quick and easy.\n\nNote that this example should be run with TensorFlow 2.12 or higher.\nBefore the release of TensorFlow 2.12, you can use `tf-nightly`.\n\n### The dataset\n\n[Our dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease) is provided by the\nCleveland Clinic Foundation for Heart Disease.\nIt's a CSV file with 303 rows. Each row contains information about a patient (a\n**sample**), and each column describes an attribute of the patient (a **feature**). We\nuse the features to predict whether a patient has a heart disease\n(**binary classification**).\n\nHere's the description of each feature:\n\nColumn| Description| Feature Type\n------------|--------------------|----------------------\nAge | Age in years | Numerical\nSex | (1 = male; 0 = female) | Categorical\nCP | Chest pain type (0, 1, 2, 3, 4) | Categorical\nTrestbpd | Resting blood pressure (in mm Hg on admission) | Numerical\nChol | Serum cholesterol in mg/dl | Numerical\nFBS | fasting blood sugar in 120 mg/dl (1 = true; 0 = false) | Categorical\nRestECG | Resting electrocardiogram results (0, 1, 2) | Categorical\nThalach | Maximum heart rate achieved | Numerical\nExang | Exercise induced angina (1 = yes; 0 = no) | Categorical\nOldpeak | ST depression induced by exercise relative to rest | Numerical\nSlope | Slope of the peak exercise ST segment | Numerical\nCA | Number of major vessels (0-3) colored by fluoroscopy | Both numerical & categorical\nThal | 3 = normal; 6 = fixed defect; 7 = reversible defect | Categorical\nTarget | Diagnosis of heart disease (1 = true; 0 = false) | Target\n"
'\n## Setup\n'
import tensorflow as tf
import pandas as pd
import keras
from keras.utils import FeatureSpace
keras.config.disable_traceback_filtering()
"\n## Preparing the data\n\nLet's download the data and load it into a Pandas dataframe:\n"
file_url = 'http://storage.googleapis.com/download.tensorflow.org/data/heart.csv'
dataframe = pd.read_csv(file_url)
'\nThe dataset includes 303 samples with 14 columns per sample\n(13 features, plus the target label):\n'
print(dataframe.shape)
"\nHere's a preview of a few samples:\n"
dataframe.head()
'\nThe last column, "target", indicates whether the patient\nhas a heart disease (1) or not (0).\n\nLet\'s split the data into a training and validation set:\n'
val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
train_dataframe = dataframe.drop(val_dataframe.index)
print('Using %d samples for training and %d for validation' % (len(train_dataframe), len(val_dataframe)))
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
'\n## Configuring a `FeatureSpace`\n\nTo configure how each feature should be preprocessed,\nwe instantiate a `keras.utils.FeatureSpace`, and we\npass to it a dictionary that maps the name of our features\nto a string that describes the feature type.\n\nWe have a few "integer categorical" features such as `"FBS"`,\none "string categorical" feature (`"thal"`),\nand a few numerical features, which we\'d like to normalize\n-- except `"age"`, which we\'d like to discretize into\na number of bins.\n\nWe also use the `crosses` argument\nto capture *feature interactions* for some categorical\nfeatures, that is to say, create additional features\nthat represent value co-occurrences for these categorical features.\nYou can compute feature crosses like this for arbitrary sets of\ncategorical features -- not just tuples of two features.\nBecause the resulting co-occurences are hashed\ninto a fixed-sized vector, you don\'t need to worry about whether\nthe co-occurence space is too large.\n'
feature_space = FeatureSpace(features={'sex': 'integer_categorical', 'cp': 'integer_categorical', 'fbs': 'integer_categorical', 'restecg': 'integer_categorical', 'exang': 'integer_categorical', 'ca': 'integer_categorical', 'thal': 'string_categorical', 'age': 'float_discretized', 'trestbps': 'float_normalized', 'chol': 'float_normalized', 'thalach': 'float_normalized', 'oldpeak': 'float_normalized', 'slope': 'float_normalized'}, crosses=[('sex', 'age'), ('thal', 'ca')], crossing_dim=32, output_mode='concat')
'\n## Further customizing a `FeatureSpace`\n\nSpecifying the feature type via a string name is quick and easy,\nbut sometimes you may want to further configure the preprocessing\nof each feature. For instance, in our case, our categorical\nfeatures don\'t have a large set of possible values -- it\'s only\na handful of values per feature (e.g. `1` and `0` for the feature `"FBS"`),\nand all possible values are represented in the training set.\nAs a result, we don\'t need to reserve an index to represent "out of vocabulary" values\nfor these features -- which would have been the default behavior.\nBelow, we just specify `num_oov_indices=0` in each of these features\nto tell the feature preprocessor to skip "out of vocabulary" indexing.\n\nOther customizations you have access to include specifying the number of\nbins for discretizing features of type `"float_discretized"`,\nor the dimensionality of the hashing space for feature crossing.\n'
feature_space = FeatureSpace(features={'sex': FeatureSpace.integer_categorical(num_oov_indices=0), 'cp': FeatureSpace.integer_categorical(num_oov_indices=0), 'fbs': FeatureSpace.integer_categorical(num_oov_indices=0), 'restecg': FeatureSpace.integer_categorical(num_oov_indices=0), 'exang': FeatureSpace.integer_categorical(num_oov_indices=0), 'ca': FeatureSpace.integer_categorical(num_oov_indices=0), 'thal': FeatureSpace.string_categorical(num_oov_indices=0), 'age': FeatureSpace.float_discretized(num_bins=30), 'trestbps': FeatureSpace.float_normalized(), 'chol': FeatureSpace.float_normalized(), 'thalach': FeatureSpace.float_normalized(), 'oldpeak': FeatureSpace.float_normalized(), 'slope': FeatureSpace.float_normalized()}, crosses=[FeatureSpace.cross(feature_names=('sex', 'age'), crossing_dim=64), FeatureSpace.cross(feature_names=('thal', 'ca'), crossing_dim=16)], output_mode='concat')
'\n## Adapt the `FeatureSpace` to the training data\n\nBefore we start using the `FeatureSpace` to build a model, we have\nto adapt it to the training data. During `adapt()`, the `FeatureSpace` will:\n\n- Index the set of possible values for categorical features.\n- Compute the mean and variance for numerical features to normalize.\n- Compute the value boundaries for the different bins for numerical features to discretize.\n\nNote that `adapt()` should be called on a `tf.data.Dataset` which yields dicts\nof feature values -- no labels.\n'
train_ds_with_no_labels = train_ds.map(lambda x, _: x)
feature_space.adapt(train_ds_with_no_labels)
'\nAt this point, the `FeatureSpace` can be called on a dict of raw feature values, and will return a\nsingle concatenate vector for each sample, combining encoded features and feature crosses.\n'
for (x, _) in train_ds.take(1):
    preprocessed_x = feature_space(x)
    print('preprocessed_x.shape:', preprocessed_x.shape)
    print('preprocessed_x.dtype:', preprocessed_x.dtype)
"\n## Two ways to manage preprocessing: as part of the `tf.data` pipeline, or in the model itself\n\nThere are two ways in which you can leverage your `FeatureSpace`:\n\n### Asynchronous preprocessing in `tf.data`\n\nYou can make it part of your data pipeline, before the model. This enables asynchronous parallel\npreprocessing of the data on CPU before it hits the model. Do this if you're training on GPU or TPU,\nor if you want to speed up preprocessing. Usually, this is always the right thing to do during training.\n\n### Synchronous preprocessing in the model\n\nYou can make it part of your model. This means that the model will expect dicts of raw feature\nvalues, and the preprocessing batch will be done synchronously (in a blocking manner) before the\nrest of the forward pass. Do this if you want to have an end-to-end model that can process\nraw feature values -- but keep in mind that your model will only be able to run on CPU,\nsince most types of feature preprocessing (e.g. string preprocessing) are not GPU or TPU compatible.\n\nDo not do this on GPU / TPU or in performance-sensitive settings. In general, you want to do in-model\npreprocessing when you do inference on CPU.\n\nIn our case, we will apply the `FeatureSpace` in the tf.data pipeline during training, but we will\ndo inference with an end-to-end model that includes the `FeatureSpace`.\n"
"\nLet's create a training and validation dataset of preprocessed batches:\n"
preprocessed_train_ds = train_ds.map(lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE)
preprocessed_train_ds = preprocessed_train_ds.prefetch(tf.data.AUTOTUNE)
preprocessed_val_ds = val_ds.map(lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE)
preprocessed_val_ds = preprocessed_val_ds.prefetch(tf.data.AUTOTUNE)
'\n## Build a model\n\nTime to build a model -- or rather two models:\n\n- A training model that expects preprocessed features (one sample = one vector)\n- An inference model that expects raw features (one sample = dict of raw feature values)\n'
dict_inputs = feature_space.get_inputs()
encoded_features = feature_space.get_encoded_features()
x = keras.layers.Dense(32, activation='relu')(encoded_features)
x = keras.layers.Dropout(0.5)(x)
predictions = keras.layers.Dense(1, activation='sigmoid')(x)
training_model = keras.Model(inputs=encoded_features, outputs=predictions)
training_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
inference_model = keras.Model(inputs=dict_inputs, outputs=predictions)
"\n## Train the model\n\nLet's train our model for 50 epochs. Note that feature preprocessing is happening\nas part of the tf.data pipeline, not as part of the model.\n"
training_model.fit(preprocessed_train_ds, epochs=20, validation_data=preprocessed_val_ds, verbose=2)
'\nWe quickly get to 80% validation accuracy.\n'
'\n## Inference on new data with the end-to-end model\n\nNow, we can use our inference model (which includes the `FeatureSpace`)\nto make predictions based on dicts of raw features values, as follows:\n'
sample = {'age': 60, 'sex': 1, 'cp': 1, 'trestbps': 145, 'chol': 233, 'fbs': 1, 'restecg': 2, 'thalach': 150, 'exang': 0, 'oldpeak': 2.3, 'slope': 3, 'ca': 0, 'thal': 'fixed'}
input_dict = {name: tf.convert_to_tensor([value]) for (name, value) in sample.items()}
predictions = inference_model.predict(input_dict)
print(f'This particular patient had a {100 * predictions[0][0]:.2f}% probability of having a heart disease, as evaluated by our model.')