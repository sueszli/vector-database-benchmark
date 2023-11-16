from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
if __name__ == '__main__':
    import argparse
    from pyspark.context import SparkContext
    from pyspark.conf import SparkConf
    import tensorflow as tf
    import tensorflow_datasets as tfds
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_partitions', help='Number of output partitions', type=int, default=10)
    parser.add_argument('--output', help='HDFS directory to save examples in parallelized format', default='data/mnist')
    args = parser.parse_args()
    print('args:', args)
    sc = SparkContext(conf=SparkConf().setAppName('mnist_data_setup'))
    (mnist, info) = tfds.load('mnist', with_info=True)
    print(info.as_json)
    mnist_train = tfds.as_numpy(mnist['train'])
    mnist_test = tfds.as_numpy(mnist['test'])
    train_rdd = sc.parallelize(mnist_train, args.num_partitions).cache()
    test_rdd = sc.parallelize(mnist_test, args.num_partitions).cache()

    def to_csv(example):
        if False:
            print('Hello World!')
        return str(example['label']) + ',' + ','.join([str(i) for i in example['image'].reshape(784)])
    train_rdd.map(to_csv).saveAsTextFile(args.output + '/csv/train')
    test_rdd.map(to_csv).saveAsTextFile(args.output + '/csv/test')

    def to_tfr(example):
        if False:
            i = 10
            return i + 15
        ex = tf.train.Example(features=tf.train.Features(feature={'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[example['label'].astype('int64')])), 'image': tf.train.Feature(int64_list=tf.train.Int64List(value=example['image'].reshape(784).astype('int64')))}))
        return (bytearray(ex.SerializeToString()), None)
    train_rdd.map(to_tfr).saveAsNewAPIHadoopFile(args.output + '/tfr/train', 'org.tensorflow.hadoop.io.TFRecordFileOutputFormat', keyClass='org.apache.hadoop.io.BytesWritable', valueClass='org.apache.hadoop.io.NullWritable')
    test_rdd.map(to_tfr).saveAsNewAPIHadoopFile(args.output + '/tfr/test', 'org.tensorflow.hadoop.io.TFRecordFileOutputFormat', keyClass='org.apache.hadoop.io.BytesWritable', valueClass='org.apache.hadoop.io.NullWritable')