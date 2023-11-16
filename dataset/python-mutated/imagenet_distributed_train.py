"""A binary to train Inception in a distributed manner using multiple systems.

Please see accompanying README.md for details and instructions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from inception import inception_distributed_train
from inception.imagenet_data import ImagenetData
FLAGS = tf.app.flags.FLAGS

def main(unused_args):
    if False:
        return 10
    assert FLAGS.job_name in ['ps', 'worker'], 'job_name must be ps or worker'
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    tf.logging.info('PS hosts are: %s' % ps_hosts)
    tf.logging.info('Worker hosts are: %s' % worker_hosts)
    cluster_spec = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})
    server = tf.train.Server({'ps': ps_hosts, 'worker': worker_hosts}, job_name=FLAGS.job_name, task_index=FLAGS.task_id, protocol=FLAGS.protocol)
    if FLAGS.job_name == 'ps':
        server.join()
    else:
        dataset = ImagenetData(subset=FLAGS.subset)
        assert dataset.data_files()
        if FLAGS.task_id == 0:
            if not tf.gfile.Exists(FLAGS.train_dir):
                tf.gfile.MakeDirs(FLAGS.train_dir)
        inception_distributed_train.train(server.target, dataset, cluster_spec)
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()