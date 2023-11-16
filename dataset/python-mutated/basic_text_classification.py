import tensorflow as tf
from tensorflow import keras
import argparse
from bigdl.dllib.utils.log4Error import invalidInputError
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.tf.estimator import Estimator
parser = argparse.ArgumentParser()
parser.add_argument('--cluster_mode', type=str, default='local', help='The mode for the Spark cluster. local, yarn or spark-submit.')
parser.add_argument('--epochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--download', type=bool, default=True, help='download dataset or not')
parser.add_argument('--data_dir', type=str, default='./dataset', help='The path of datesets where includes imdb.npz.')
parser.add_argument('--container_image', type=str, default='', help='The runtime k8s image. You can change it with your k8s image.')
parser.add_argument('--k8s_master', type=str, default='', help='The k8s master. It should be k8s://https://<k8s-apiserver-host>: <k8s-apiserver-port>.')
args = parser.parse_args()
cluster_mode = args.cluster_mode
download = args.download
if cluster_mode == 'local':
    init_orca_context(cluster_mode='local', cores=4, memory='3g')
elif cluster_mode.startswith('yarn'):
    init_orca_context(cluster_mode=cluster_mode, num_nodes=2, cores=2, driver_memory='3g')
elif cluster_mode == 'k8s':
    init_orca_context(cluster_mode='k8s', master=args.k8s_master, container_image=args.container_image, num_nodes=2, cores=4, driver_memory='3g')
elif cluster_mode == 'spark-submit':
    init_orca_context(cluster_mode='spark-submit')
else:
    print("init_orca_context failed. cluster_mode should be one of 'local', 'yarn' and 'spark-submit' but got " + cluster_mode)
print(tf.__version__)
imdb = keras.datasets.imdb
if download:
    ((train_data, train_labels), (test_data, test_labels)) = imdb.load_data(num_words=10000)
else:
    import numpy as np
    import os
    from tensorflow.keras.utils import get_file
    path = os.path.join(args.data_dir, 'imdb.npz')
    num_words = 10000
    skip_top = 0
    maxlen = None
    seed = 113
    start_char = 1
    oov_char = 2
    index_from = 3
    with np.load(path, allow_pickle=True) as f:
        (x_train, labels_train) = (f['x_train'], f['y_train'])
        (x_test, labels_test) = (f['x_test'], f['y_test'])
    rng = np.random.RandomState(seed)
    indices = np.arange(len(x_train))
    rng.shuffle(indices)
    x_train = x_train[indices]
    labels_train = labels_train[indices]
    indices = np.arange(len(x_test))
    rng.shuffle(indices)
    x_test = x_test[indices]
    labels_test = labels_test[indices]
    if start_char is not None:
        x_train = [[start_char] + [w + index_from for w in x] for x in x_train]
        x_test = [[start_char] + [w + index_from for w in x] for x in x_test]
    elif index_from:
        x_train = [[w + index_from for w in x] for x in x_train]
        x_test = [[w + index_from for w in x] for x in x_test]
    if maxlen:
        (x_train, labels_train) = _remove_long_seq(maxlen, x_train, labels_train)
        (x_test, labels_test) = _remove_long_seq(maxlen, x_test, labels_test)
    if not x_train or not x_test:
        invalidInputError(False, f'After filtering for sequences shorter than maxlen={str(maxlen)}, no sequence was kept. Increase maxlen.')
    xs = x_train + x_test
    labels = np.concatenate([labels_train, labels_test])
    if not num_words:
        num_words = max((max(x) for x in xs))
    if oov_char is not None:
        xs = [[w if skip_top <= w < num_words else oov_char for w in x] for x in xs]
    else:
        xs = [[w for w in x if skip_top <= w < num_words] for x in xs]
    idx = len(x_train)
    (x_train, y_train) = (np.array(xs[:idx], dtype='object'), labels[:idx])
    (x_test, y_test) = (np.array(xs[idx:], dtype='object'), labels[idx:])
    ((train_data, train_labels), (test_data, test_labels)) = ((x_train, y_train), (x_test, y_test))
print('Training entries: {}, labels: {}'.format(len(train_data), len(train_labels)))
print(train_data[0])
(len(train_data[0]), len(train_data[1]))
if download:
    word_index = imdb.get_word_index()
else:
    import json
    path = os.path.join(args.data_dir, 'imdb_word_index.json')
    with open(path) as f:
        word_index = json.load(f)
word_index = {k: v + 3 for (k, v) in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    if False:
        print('Hello World!')
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
decode_review(train_data[0])
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index['<PAD>'], padding='post', maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index['<PAD>'], padding='post', maxlen=256)
(len(train_data[0]), len(train_data[1]))
vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
x_val = train_data[:10000]
partial_x_train = train_data[10000:]
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]
train_dataset = tf.data.Dataset.from_tensor_slices((partial_x_train, partial_y_train))
validation_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
est = Estimator.from_keras(keras_model=model)
est.fit(data=train_dataset, batch_size=512, epochs=args.epochs, validation_data=validation_dataset)
results = est.evaluate(validation_dataset)
print(results)
stop_orca_context()