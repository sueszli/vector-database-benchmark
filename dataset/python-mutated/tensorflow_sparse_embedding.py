import os
import re
import string
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
from bigdl.nano.tf.keras.layers import Embedding
from bigdl.nano.tf.optimizers import SparseAdam
from bigdl.nano.tf.keras import Model
max_features = 20000
embedding_dim = 128

def create_datasets():
    if False:
        print('Hello World!')
    ((raw_train_ds, raw_val_ds, raw_test_ds), info) = tfds.load('imdb_reviews', data_dir='/tmp/data', split=['train[:80%]', 'train[80%:]', 'test'], as_supervised=True, batch_size=32, with_info=True)

    def custom_standardization(input_data):
        if False:
            return 10
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
        return tf.strings.regex_replace(stripped_html, f'[{re.escape(string.punctuation)}]', '')
    vectorize_layer = TextVectorization(standardize=custom_standardization, max_tokens=max_features, output_mode='int', output_sequence_length=500)
    text_ds = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(text_ds)

    def vectorize_text(text, label):
        if False:
            i = 10
            return i + 15
        text = tf.expand_dims(text, -1)
        return (vectorize_layer(text), label)
    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)
    return (train_ds, val_ds, test_ds)

def make_backbone():
    if False:
        i = 10
        return i + 15
    inputs = tf.keras.Input(shape=(None, embedding_dim))
    x = layers.Dropout(0.5)(inputs)
    x = layers.Conv1D(128, 7, padding='valid', activation='relu', strides=3)(x)
    x = layers.Conv1D(128, 7, padding='valid', activation='relu', strides=3)(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(1, activation='sigmoid', name='predictions')(x)
    model = Model(inputs, predictions)
    return model

def make_model():
    if False:
        return 10
    inputs = tf.keras.Input(shape=(None,), dtype='int64')
    x = Embedding(max_features, embedding_dim)(inputs)
    predictions = make_backbone()(x)
    model = Model(inputs, predictions)
    model.compile(loss='binary_crossentropy', optimizer=SparseAdam(), metrics=['accuracy'])
    return model
if __name__ == '__main__':
    num_epochs = int(os.environ.get('NUM_EPOCHS', 10))
    (train_ds, val_ds, test_ds) = create_datasets()
    model = make_model()
    model.fit(train_ds, validation_data=val_ds, epochs=num_epochs)
    his = model.evaluate(test_ds)