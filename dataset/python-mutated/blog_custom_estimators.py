import tensorflow as tf
import os
import sys
import six.moves.urllib.request as request
from distutils.version import StrictVersion
tf.logging.set_verbosity(tf.logging.INFO)
tf_version = tf.__version__
tf.logging.info('TensorFlow version: {}'.format(tf_version))
assert StrictVersion('1.4') <= StrictVersion(tf_version), 'TensorFlow r1.4 or later is needed'
PATH = '/tmp/tf_custom_estimators'
PATH_DATASET = PATH + os.sep + 'dataset'
FILE_TRAIN = PATH_DATASET + os.sep + 'iris_training.csv'
FILE_TEST = PATH_DATASET + os.sep + 'iris_test.csv'
URL_TRAIN = 'http://download.tensorflow.org/data/iris_training.csv'
URL_TEST = 'http://download.tensorflow.org/data/iris_test.csv'

def downloadDataset(url, file):
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
downloadDataset(URL_TRAIN, FILE_TRAIN)
downloadDataset(URL_TEST, FILE_TEST)
feature_names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']

def my_input_fn(file_path, repeat_count=1, shuffle_count=1):
    if False:
        while True:
            i = 10

    def decode_csv(line):
        if False:
            i = 10
            return i + 15
        parsed_line = tf.decode_csv(line, [[0.0], [0.0], [0.0], [0.0], [0]])
        label = parsed_line[-1]
        del parsed_line[-1]
        features = parsed_line
        d = (dict(zip(feature_names, features)), label)
        return d
    dataset = tf.data.TextLineDataset(file_path).skip(1).map(decode_csv, num_parallel_calls=4).cache().shuffle(shuffle_count).repeat(repeat_count).batch(32).prefetch(1)
    iterator = dataset.make_one_shot_iterator()
    (batch_features, batch_labels) = iterator.get_next()
    return (batch_features, batch_labels)

def my_model_fn(features, labels, mode):
    if False:
        i = 10
        return i + 15
    if mode == tf.estimator.ModeKeys.PREDICT:
        tf.logging.info('my_model_fn: PREDICT, {}'.format(mode))
    elif mode == tf.estimator.ModeKeys.EVAL:
        tf.logging.info('my_model_fn: EVAL, {}'.format(mode))
    elif mode == tf.estimator.ModeKeys.TRAIN:
        tf.logging.info('my_model_fn: TRAIN, {}'.format(mode))
    feature_columns = [tf.feature_column.numeric_column(feature_names[0]), tf.feature_column.numeric_column(feature_names[1]), tf.feature_column.numeric_column(feature_names[2]), tf.feature_column.numeric_column(feature_names[3])]
    input_layer = tf.feature_column.input_layer(features, feature_columns)
    h1 = tf.layers.Dense(10, activation=tf.nn.relu)(input_layer)
    h2 = tf.layers.Dense(10, activation=tf.nn.relu)(h1)
    logits = tf.layers.Dense(3)(h2)
    predictions = {'class_ids': tf.argmax(input=logits, axis=1)}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.metrics.accuracy(labels, predictions['class_ids'])
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={'my_accuracy': accuracy})
    assert mode == tf.estimator.ModeKeys.TRAIN, 'TRAIN is only ModeKey left'
    optimizer = tf.train.AdagradOptimizer(0.05)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    tf.summary.scalar('my_accuracy', accuracy[1])
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
tf.logging.info('Before classifier construction')
classifier = tf.estimator.Estimator(model_fn=my_model_fn, model_dir=PATH)
tf.logging.info('...done constructing classifier')
tf.logging.info('Before classifier.train')
classifier.train(input_fn=lambda : my_input_fn(FILE_TRAIN, 500, 256))
tf.logging.info('...done classifier.train')
tf.logging.info('Before classifier.evaluate')
evaluate_result = classifier.evaluate(input_fn=lambda : my_input_fn(FILE_TEST, 4))
tf.logging.info('...done classifier.evaluate')
tf.logging.info('Evaluation results')
for key in evaluate_result:
    tf.logging.info('   {}, was: {}'.format(key, evaluate_result[key]))
predict_results = classifier.predict(input_fn=lambda : my_input_fn(FILE_TEST, 1))
tf.logging.info('Prediction on test file')
for prediction in predict_results:
    tf.logging.info('...{}'.format(prediction['class_ids']))
prediction_input = [[5.9, 3.0, 4.2, 1.5], [6.9, 3.1, 5.4, 2.1], [5.1, 3.3, 1.7, 0.5]]

def new_input_fn():
    if False:
        i = 10
        return i + 15

    def decode(x):
        if False:
            i = 10
            return i + 15
        x = tf.split(x, 4)
        return dict(zip(feature_names, x))
    dataset = tf.data.Dataset.from_tensor_slices(prediction_input)
    dataset = dataset.map(decode)
    iterator = dataset.make_one_shot_iterator()
    next_feature_batch = iterator.get_next()
    return (next_feature_batch, None)
predict_results = classifier.predict(input_fn=new_input_fn)
tf.logging.info('Predictions on memory')
for (idx, prediction) in enumerate(predict_results):
    type = prediction['class_ids']
    if type == 0:
        tf.logging.info('...I think: {}, is Iris Setosa'.format(prediction_input[idx]))
    elif type == 1:
        tf.logging.info('...I think: {}, is Iris Versicolor'.format(prediction_input[idx]))
    else:
        tf.logging.info('...I think: {}, is Iris Virginica'.format(prediction_input[idx]))