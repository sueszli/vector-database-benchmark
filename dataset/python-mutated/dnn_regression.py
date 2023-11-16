"""Regression using the DNNRegressor Estimator."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import tensorflow as tf
import automobile_data
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=5000, type=int, help='number of training steps')
parser.add_argument('--price_norm_factor', default=1000.0, type=float, help='price normalization factor')

def main(argv):
    if False:
        for i in range(10):
            print('nop')
    'Builds, trains, and evaluates the model.'
    args = parser.parse_args(argv[1:])
    ((train_x, train_y), (test_x, test_y)) = automobile_data.load_data()
    train_y /= args.price_norm_factor
    test_y /= args.price_norm_factor
    train_input_fn = automobile_data.make_dataset(args.batch_size, train_x, train_y, True, 1000)
    test_input_fn = automobile_data.make_dataset(args.batch_size, test_x, test_y)
    body_style_vocab = ['hardtop', 'wagon', 'sedan', 'hatchback', 'convertible']
    body_style_column = tf.feature_column.categorical_column_with_vocabulary_list(key='body-style', vocabulary_list=body_style_vocab)
    make_column = tf.feature_column.categorical_column_with_hash_bucket(key='make', hash_bucket_size=50)
    feature_columns = [tf.feature_column.numeric_column(key='curb-weight'), tf.feature_column.numeric_column(key='highway-mpg'), tf.feature_column.indicator_column(body_style_column), tf.feature_column.embedding_column(make_column, dimension=3)]
    model = tf.estimator.DNNRegressor(hidden_units=[20, 20], feature_columns=feature_columns)
    model.train(input_fn=train_input_fn, steps=args.train_steps)
    eval_result = model.evaluate(input_fn=test_input_fn)
    average_loss = eval_result['average_loss']
    print('\n' + 80 * '*')
    print('\nRMS error for the test set: ${:.0f}'.format(args.price_norm_factor * average_loss ** 0.5))
    print()
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)