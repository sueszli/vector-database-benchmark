from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import numpy as np
import tensorflow as tf

def inference(it, num_workers, args):
    if False:
        for i in range(10):
            print('nop')
    from tensorflowonspark import util
    for i in it:
        worker_num = i
    print('worker_num: {}'.format(i))
    util.single_node_env()
    saved_model = tf.saved_model.load(args.export_dir, tags='serve')
    predict = saved_model.signatures['serving_default']

    def parse_tfr(example_proto):
        if False:
            for i in range(10):
                print('nop')
        feature_def = {'label': tf.io.FixedLenFeature(1, tf.int64), 'image': tf.io.FixedLenFeature(784, tf.int64)}
        features = tf.io.parse_single_example(serialized=example_proto, features=feature_def)
        image = tf.cast(features['image'], dtype=tf.float32) / 255.0
        image = tf.reshape(image, [28, 28, 1])
        label = tf.cast(features['label'], dtype=tf.float32)
        return (image, label)
    ds = tf.data.Dataset.list_files('{}/part-*'.format(args.images_labels), shuffle=False)
    ds = ds.shard(num_workers, worker_num)
    ds = ds.interleave(tf.data.TFRecordDataset)
    ds = ds.map(parse_tfr)
    ds = ds.batch(10)
    tf.io.gfile.makedirs(args.output)
    output_file = tf.io.gfile.GFile('{}/part-{:05d}'.format(args.output, worker_num), mode='w')
    for batch in ds:
        predictions = predict(conv2d_input=batch[0])
        labels = np.reshape(batch[1], -1).astype(np.int)
        preds = np.argmax(predictions['logits'], axis=1)
        for x in zip(labels, preds):
            output_file.write('{} {}\n'.format(x[0], x[1]))
    output_file.close()
if __name__ == '__main__':
    from pyspark.context import SparkContext
    from pyspark.conf import SparkConf
    sc = SparkContext(conf=SparkConf().setAppName('mnist_inference'))
    executors = sc._conf.get('spark.executor.instances')
    num_executors = int(executors) if executors is not None else 1
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster_size', help='number of nodes in the cluster (for S with labelspark Standalone)', type=int, default=num_executors)
    parser.add_argument('--images_labels', type=str, help='Directory for input images with labels')
    parser.add_argument('--export_dir', help='HDFS path to export model', type=str, default='mnist_export')
    parser.add_argument('--output', help='HDFS path to save predictions', type=str, default='predictions')
    (args, _) = parser.parse_known_args()
    print('args: {}'.format(args))
    nodes = list(range(args.cluster_size))
    nodeRDD = sc.parallelize(list(range(args.cluster_size)), args.cluster_size)
    nodeRDD.foreachPartition(lambda worker_num: inference(worker_num, args.cluster_size, args))