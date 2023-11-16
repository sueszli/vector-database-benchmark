import pandas as pd
import tensorflow as tf
TRAIN_URL = 'http://download.tensorflow.org/data/iris_training.csv'
TEST_URL = 'http://download.tensorflow.org/data/iris_test.csv'
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

def maybe_download():
    if False:
        return 10
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)
    return (train_path, test_path)

def load_data(y_name='Species'):
    if False:
        while True:
            i = 10
    'Returns the iris dataset as (train_x, train_y), (test_x, test_y).'
    (train_path, test_path) = maybe_download()
    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    (train_x, train_y) = (train, train.pop(y_name))
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    (test_x, test_y) = (test, test.pop(y_name))
    return ((train_x, train_y), (test_x, test_y))

def train_input_fn(features, labels, batch_size):
    if False:
        print('Hello World!')
    'An input function for training'
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset

def eval_input_fn(features, labels, batch_size):
    if False:
        print('Hello World!')
    'An input function for evaluation or prediction'
    features = dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    assert batch_size is not None, 'batch_size must not be None'
    dataset = dataset.batch(batch_size)
    return dataset
CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]

def _parse_line(line):
    if False:
        print('Hello World!')
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)
    features = dict(zip(CSV_COLUMN_NAMES, fields))
    label = features.pop('Species')
    return (features, label)

def csv_input_fn(csv_path, batch_size):
    if False:
        for i in range(10):
            print('nop')
    dataset = tf.data.TextLineDataset(csv_path).skip(1)
    dataset = dataset.map(_parse_line)
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset