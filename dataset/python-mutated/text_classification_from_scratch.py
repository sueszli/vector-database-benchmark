"""
Title: Text classification from scratch
Authors: Mark Omernick, Francois Chollet
Date created: 2019/11/06
Last modified: 2020/05/17
Description: Text sentiment classification starting from raw text files.
Accelerator: GPU
"""
'\n## Introduction\n\nThis example shows how to do text classification starting from raw text (as\na set of text files on disk). We demonstrate the workflow on the IMDB sentiment\nclassification dataset (unprocessed version). We use the `TextVectorization` layer for\n word splitting & indexing.\n'
'\n## Setup\n'
import tensorflow as tf
import keras
from keras.layers import TextVectorization
from keras import layers
import string
import re
import os
from pathlib import Path
"\n## Load the data: IMDB movie review sentiment classification\n\nLet's download the data and inspect its structure.\n"
fpath = keras.utils.get_file(origin='https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
dirpath = Path(fpath).parent.absolute()
os.system(f'tar -xf {fpath} -C {dirpath}')
'\nThe `aclImdb` folder contains a `train` and `test` subfolder:\n'
os.system(f'ls {dirpath}/aclImdb')
os.system(f'ls {dirpath}/aclImdb/train')
os.system(f'ls {dirpath}/aclImdb/test')
'\nThe `aclImdb/train/pos` and `aclImdb/train/neg` folders contain text files, each of\n which represents one review (either positive or negative):\n'
os.system(f'cat {dirpath}/aclImdb/train/pos/6248_7.txt')
"\nWe are only interested in the `pos` and `neg` subfolders, so let's delete the rest:\n"
os.system(f'rm -r {dirpath}/aclImdb/train/unsup')
"\nYou can use the utility `keras.utils.text_dataset_from_directory` to\ngenerate a labeled `tf.data.Dataset` object from a set of text files on disk filed\n into class-specific folders.\n\nLet's use it to generate the training, validation, and test datasets. The validation\nand training datasets are generated from two subsets of the `train` directory, with 20%\nof samples going to the validation dataset and 80% going to the training dataset.\n\nHaving a validation dataset in addition to the test dataset is useful for tuning\nhyperparameters, such as the model architecture, for which the test dataset should not\nbe used.\n\nBefore putting the model out into the real world however, it should be retrained using all\navailable training data (without creating a validation dataset), so its performance is maximized.\n\nWhen using the `validation_split` & `subset` arguments, make sure to either specify a\nrandom seed, or to pass `shuffle=False`, so that the validation & training splits you\nget have no overlap.\n"
batch_size = 32
(raw_train_ds, raw_val_ds) = keras.utils.text_dataset_from_directory(f'{dirpath}/aclImdb/train', batch_size=batch_size, validation_split=0.2, subset='both', seed=1337)
raw_test_ds = keras.utils.text_dataset_from_directory(f'{dirpath}/aclImdb/test', batch_size=batch_size)
print(f'Number of batches in raw_train_ds: {raw_train_ds.cardinality()}')
print(f'Number of batches in raw_val_ds: {raw_val_ds.cardinality()}')
print(f'Number of batches in raw_test_ds: {raw_test_ds.cardinality()}')
"\nLet's preview a few samples:\n"
for (text_batch, label_batch) in raw_train_ds.take(1):
    for i in range(5):
        print(text_batch.numpy()[i])
        print(label_batch.numpy()[i])
'\n## Prepare the data\n\nIn particular, we remove `<br />` tags.\n'

def custom_standardization(input_data):
    if False:
        print('Hello World!')
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, f'[{re.escape(string.punctuation)}]', '')
max_features = 20000
embedding_dim = 128
sequence_length = 500
vectorize_layer = TextVectorization(standardize=custom_standardization, max_tokens=max_features, output_mode='int', output_sequence_length=sequence_length)
text_ds = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)
'\n## Two options to vectorize the data\n\nThere are 2 ways we can use our text vectorization layer:\n\n**Option 1: Make it part of the model**, so as to obtain a model that processes raw\nstrings, like this:\n'
"\n\n```python\ntext_input = keras.Input(shape=(1,), dtype=tf.string, name='text')\nx = vectorize_layer(text_input)\nx = layers.Embedding(max_features + 1, embedding_dim)(x)\n...\n```\n\n**Option 2: Apply it to the text dataset** to obtain a dataset of word indices, then\n feed it into a model that expects integer sequences as inputs.\n\nAn important difference between the two is that option 2 enables you to do\n**asynchronous CPU processing and buffering** of your data when training on GPU.\nSo if you're training the model on GPU, you probably want to go with this option to get\nthe best performance. This is what we will do below.\n\nIf we were to export our model to production, we'd ship a model that accepts raw\nstrings as input, like in the code snippet for option 1 above. This can be done after\ntraining. We do this in the last section.\n"

def vectorize_text(text, label):
    if False:
        return 10
    text = tf.expand_dims(text, -1)
    return (vectorize_layer(text), label)
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)
train_ds = train_ds.cache().prefetch(buffer_size=10)
val_ds = val_ds.cache().prefetch(buffer_size=10)
test_ds = test_ds.cache().prefetch(buffer_size=10)
'\n## Build a model\n\nWe choose a simple 1D convnet starting with an `Embedding` layer.\n'
inputs = keras.Input(shape=(sequence_length,), dtype='int64')
x = keras.layers.Embedding(max_features, embedding_dim)(inputs)
x = layers.Dropout(0.5)(x)
x = layers.Conv1D(128, 7, padding='valid', activation='relu', strides=3)(x)
x = layers.Conv1D(128, 7, padding='valid', activation='relu', strides=3)(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(1, activation='sigmoid', name='predictions')(x)
model = keras.Model(inputs, predictions)
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
'\n## Train the model\n'
epochs = 3
model.fit(train_ds, validation_data=val_ds, epochs=epochs)
'\n## Evaluate the model on the test set\n'
model.evaluate(test_ds)
'\n## Make an end-to-end model\n\nIf you want to obtain a model capable of processing raw strings, you can simply\ncreate a new model (using the weights we just trained):\n'
inputs = keras.Input(shape=(1,), dtype='string')
indices = vectorize_layer(inputs)
outputs = model(indices)
end_to_end_model = keras.Model(inputs, outputs)
end_to_end_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
end_to_end_model.evaluate(raw_test_ds)