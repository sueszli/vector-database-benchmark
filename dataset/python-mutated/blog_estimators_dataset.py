import os
import six.moves.urllib.request as request
import tensorflow as tf
from distutils.version import StrictVersion
tf_version = tf.__version__
print('TensorFlow version: {}'.format(tf_version))
assert StrictVersion('1.4') <= StrictVersion(tf_version), 'TensorFlow r1.4 or later is needed'
PATH = '/tmp/tf_dataset_and_estimator_apis'
PATH_DATASET = PATH + os.sep + 'dataset'
FILE_TRAIN = PATH_DATASET + os.sep + 'iris_training.csv'
FILE_TEST = PATH_DATASET + os.sep + 'iris_test.csv'
URL_TRAIN = 'http://download.tensorflow.org/data/iris_training.csv'
URL_TEST = 'http://download.tensorflow.org/data/iris_test.csv'

def download_dataset(url, file):
    if False:
        for i in range(10):
            print('nop')
    if not os.path.exists(PATH_DATASET):
        os.makedirs(PATH_DATASET)
    if not os.path.exists(file):
        data = request.urlopen(url).read()
        with open(file, 'wb') as f:
            f.write(data)
            f.close()
download_dataset(URL_TRAIN, FILE_TRAIN)
download_dataset(URL_TEST, FILE_TEST)
tf.logging.set_verbosity(tf.logging.INFO)
feature_names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']

def my_input_fn(file_path, perform_shuffle=False, repeat_count=1):
    if False:
        for i in range(10):
            print('nop')

    def decode_csv(line):
        if False:
            print('Hello World!')
        parsed_line = tf.decode_csv(line, [[0.0], [0.0], [0.0], [0.0], [0]])
        label = parsed_line[-1]
        del parsed_line[-1]
        features = parsed_line
        d = (dict(zip(feature_names, features)), label)
        return d
    dataset = tf.data.TextLineDataset(file_path).skip(1).map(decode_csv)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count)
    dataset = dataset.batch(32)
    iterator = dataset.make_one_shot_iterator()
    (batch_features, batch_labels) = iterator.get_next()
    return (batch_features, batch_labels)
next_batch = my_input_fn(FILE_TRAIN, True)
feature_columns = [tf.feature_column.numeric_column(k) for k in feature_names]
classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 10], n_classes=3, model_dir=PATH)
classifier.train(input_fn=lambda : my_input_fn(FILE_TRAIN, True, 8))
evaluate_result = classifier.evaluate(input_fn=lambda : my_input_fn(FILE_TEST, False, 4))
print('Evaluation results')
for key in evaluate_result:
    print('   {}, was: {}'.format(key, evaluate_result[key]))
predict_results = classifier.predict(input_fn=lambda : my_input_fn(FILE_TEST, False, 1))
print('Predictions on test file')
for prediction in predict_results:
    print(prediction['class_ids'][0])
prediction_input = [[5.9, 3.0, 4.2, 1.5], [6.9, 3.1, 5.4, 2.1], [5.1, 3.3, 1.7, 0.5]]

def new_input_fn():
    if False:
        for i in range(10):
            print('nop')

    def decode(x):
        if False:
            print('Hello World!')
        x = tf.split(x, 4)
        return dict(zip(feature_names, x))
    dataset = tf.data.Dataset.from_tensor_slices(prediction_input)
    dataset = dataset.map(decode)
    iterator = dataset.make_one_shot_iterator()
    next_feature_batch = iterator.get_next()
    return (next_feature_batch, None)
predict_results = classifier.predict(input_fn=new_input_fn)
print('Predictions:')
for (idx, prediction) in enumerate(predict_results):
    type = prediction['class_ids'][0]
    if type == 0:
        print('  I think: {}, is Iris Sentosa'.format(prediction_input[idx]))
    elif type == 1:
        print('  I think: {}, is Iris Versicolor'.format(prediction_input[idx]))
    else:
        print('  I think: {}, is Iris Virginica'.format(prediction_input[idx]))