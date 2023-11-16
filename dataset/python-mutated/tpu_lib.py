"""Initializes TPU system for TF 2.0."""
import tensorflow as tf

def tpu_initialize(tpu_address):
    if False:
        i = 10
        return i + 15
    'Initializes TPU for TF 2.0 training.\n\n  Args:\n    tpu_address: string, bns address of master TPU worker.\n\n  Returns:\n    A TPUClusterResolver.\n  '
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_address)
    if tpu_address not in ('', 'local'):
        tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    return cluster_resolver