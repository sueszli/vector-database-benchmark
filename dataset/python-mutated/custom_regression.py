"""Regression using the DNNRegressor Estimator."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import tensorflow as tf
import automobile_data
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int, help='number of training steps')
parser.add_argument('--price_norm_factor', default=1000.0, type=float, help='price normalization factor')

def my_dnn_regression_fn(features, labels, mode, params):
    if False:
        while True:
            i = 10
    'A model function implementing DNN regression for a custom Estimator.'
    top = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params.get('hidden_units', [20]):
        top = tf.layers.dense(inputs=top, units=units, activation=tf.nn.relu)
    output_layer = tf.layers.dense(inputs=top, units=1)
    predictions = tf.squeeze(output_layer, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={'price': predictions})
    average_loss = tf.losses.mean_squared_error(labels, predictions)
    batch_size = tf.shape(labels)[0]
    total_loss = tf.to_float(batch_size) * average_loss
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = params.get('optimizer', tf.train.AdamOptimizer)
        optimizer = optimizer(params.get('learning_rate', None))
        train_op = optimizer.minimize(loss=average_loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op)
    assert mode == tf.estimator.ModeKeys.EVAL
    print(labels)
    print(predictions)
    predictions = tf.cast(predictions, tf.float64)
    rmse = tf.metrics.root_mean_squared_error(labels, predictions)
    eval_metrics = {'rmse': rmse}
    return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, eval_metric_ops=eval_metrics)

def main(argv):
    if False:
        i = 10
        return i + 15
    'Builds, trains, and evaluates the model.'
    args = parser.parse_args(argv[1:])
    ((train_x, train_y), (test_x, test_y)) = automobile_data.load_data()
    train_y /= args.price_norm_factor
    test_y /= args.price_norm_factor
    train_input_fn = automobile_data.make_dataset(args.batch_size, train_x, train_y, True, 1000)
    test_input_fn = automobile_data.make_dataset(args.batch_size, test_x, test_y)
    body_style_vocab = ['hardtop', 'wagon', 'sedan', 'hatchback', 'convertible']
    body_style = tf.feature_column.categorical_column_with_vocabulary_list(key='body-style', vocabulary_list=body_style_vocab)
    make = tf.feature_column.categorical_column_with_hash_bucket(key='make', hash_bucket_size=50)
    feature_columns = [tf.feature_column.numeric_column(key='curb-weight'), tf.feature_column.numeric_column(key='highway-mpg'), tf.feature_column.indicator_column(body_style), tf.feature_column.embedding_column(make, dimension=3)]
    model = tf.estimator.Estimator(model_fn=my_dnn_regression_fn, params={'feature_columns': feature_columns, 'learning_rate': 0.001, 'optimizer': tf.train.AdamOptimizer, 'hidden_units': [20, 20]})
    model.train(input_fn=train_input_fn, steps=args.train_steps)
    eval_result = model.evaluate(input_fn=test_input_fn)
    print('\n' + 80 * '*')
    print('\nRMS error for the test set: ${:.0f}'.format(args.price_norm_factor * eval_result['rmse']))
    print()
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)