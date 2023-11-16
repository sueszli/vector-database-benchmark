"""This module provides helper functions for the TensorFlow application.

Primarily, these functions help with:

* starting the TensorFlow ``tf.train.Server`` for the node (allocating GPUs as desired, and determining the node's role in the cluster).
* managing input/output data for *InputMode.SPARK*.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function
import getpass
import logging
import pkg_resources
from packaging import version
from six.moves.queue import Empty
from . import compat, marker
logger = logging.getLogger(__name__)
try:
    TF_VERSION = pkg_resources.get_distribution('tensorflow').version
except pkg_resources.DistributionNotFound:
    TF_VERSION = pkg_resources.get_distribution('tensorflow-cpu').version

def hdfs_path(ctx, path):
    if False:
        while True:
            i = 10
    'Convenience function to create a Tensorflow-compatible absolute HDFS path from relative paths\n\n  Args:\n    :ctx: TFNodeContext containing the metadata specific to this node in the cluster.\n    :path: path to convert\n\n  Returns:\n    An absolute path prefixed with the correct filesystem scheme.\n  '
    HADOOP_SCHEMES = ['adl://', 'file://', 'hdfs://', 'oss://', 's3://', 's3a://', 's3n://', 'swift://', 'viewfs://', 'wasb://']
    if any((path.startswith(scheme) for scheme in HADOOP_SCHEMES)):
        return path
    elif path.startswith('/'):
        return ctx.defaultFS + path
    elif ctx.defaultFS.startswith('hdfs://') or ctx.defaultFS.startswith('viewfs://'):
        return '{0}/user/{1}/{2}'.format(ctx.defaultFS, getpass.getuser(), path)
    elif ctx.defaultFS.startswith('file://'):
        return '{0}/{1}/{2}'.format(ctx.defaultFS, ctx.working_dir[1:], path)
    else:
        logger.warn('Unknown scheme {0} with relative path: {1}'.format(ctx.defaultFS, path))
        return '{0}/{1}'.format(ctx.defaultFS, path)

def start_cluster_server(ctx, num_gpus=1, rdma=False):
    if False:
        return 10
    "Function that wraps the creation of TensorFlow ``tf.train.Server`` for a node in a distributed TensorFlow cluster.\n\n  This is intended to be invoked from within the TF ``map_fun``, replacing explicit code to instantiate ``tf.train.ClusterSpec``\n  and ``tf.train.Server`` objects.\n\n  DEPRECATED for TensorFlow 2.x+\n\n  Args:\n    :ctx: TFNodeContext containing the metadata specific to this node in the cluster.\n    :num_gpu: number of GPUs desired\n    :rdma: boolean indicating if RDMA 'iverbs' should be used for cluster communications.\n\n  Returns:\n    A tuple of (cluster_spec, server)\n  "
    import os
    import time
    from . import gpu_info
    if version.parse(TF_VERSION) >= version.parse('2.0.0'):
        raise Exception('DEPRECATED: Use higher-level APIs like `tf.keras` or `tf.estimator`')
    logging.info('{0}: ======== {1}:{2} ========'.format(ctx.worker_num, ctx.job_name, ctx.task_index))
    cluster_spec = ctx.cluster_spec
    logging.info('{0}: Cluster spec: {1}'.format(ctx.worker_num, cluster_spec))
    if compat.is_gpu_available() and num_gpus > 0:
        my_addr = cluster_spec[ctx.job_name][ctx.task_index]
        my_host = my_addr.split(':')[0]
        flattened = [v for sublist in cluster_spec.values() for v in sublist]
        local_peers = [p for p in flattened if p.startswith(my_host)]
        my_index = local_peers.index(my_addr)
        gpu_initialized = False
        retries = 3
        while not gpu_initialized and retries > 0:
            try:
                if ctx.job_name == 'ps':
                    num_gpus = 0
                gpus_to_use = gpu_info.get_gpus(num_gpus, my_index)
                gpu_prompt = 'GPU' if num_gpus == 1 else 'GPUs'
                logging.info('{0}: Using {1}: {2}'.format(ctx.worker_num, gpu_prompt, gpus_to_use))
                os.environ['CUDA_VISIBLE_DEVICES'] = gpus_to_use
                import tensorflow as tf
                cluster = tf.train.ClusterSpec(cluster_spec)
                if rdma:
                    server = tf.train.Server(cluster, ctx.job_name, ctx.task_index, protocol='grpc+verbs')
                else:
                    server = tf.train.Server(cluster, ctx.job_name, ctx.task_index)
                gpu_initialized = True
            except Exception as e:
                print(e)
                logging.error('{0}: Failed to allocate GPU, trying again...'.format(ctx.worker_num))
                retries -= 1
                time.sleep(10)
        if not gpu_initialized:
            raise Exception('Failed to allocate GPU')
    else:
        import tensorflow as tf
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        logging.info('{0}: Using CPU'.format(ctx.worker_num))
        cluster = tf.train.ClusterSpec(cluster_spec)
        server = tf.train.Server(cluster, ctx.job_name, ctx.task_index)
    return (cluster, server)

def next_batch(mgr, batch_size, qname='input'):
    if False:
        return 10
    '*DEPRECATED*. Use TFNode.DataFeed class instead.'
    raise Exception('DEPRECATED: Use TFNode.DataFeed class instead')

def export_saved_model(sess, export_dir, tag_set, signatures):
    if False:
        i = 10
        return i + 15
    "Convenience function to export a saved_model using provided arguments\n\n  The caller specifies the saved_model signatures in a simplified python dictionary form, as follows::\n\n    signatures = {\n      'signature_def_key': {\n        'inputs': { 'input_tensor_alias': input_tensor_name },\n        'outputs': { 'output_tensor_alias': output_tensor_name },\n        'method_name': 'method'\n      }\n    }\n\n  And this function will generate the `signature_def_map` and export the saved_model.\n\n  DEPRECATED for TensorFlow 2.x+.\n\n  Args:\n    :sess: a tf.Session instance\n    :export_dir: path to save exported saved_model\n    :tag_set: string tag_set to identify the exported graph\n    :signatures: simplified dictionary representation of a TensorFlow signature_def_map\n\n  Returns:\n    A saved_model exported to disk at ``export_dir``.\n  "
    import tensorflow as tf
    if version.parse(tf.__version__) >= version.parse('2.0.0'):
        raise Exception('DEPRECATED: Use TF provided APIs instead.')
    g = sess.graph
    g._unsafe_unfinalize()
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    logging.info('===== signatures: {}'.format(signatures))
    signature_def_map = {}
    for (key, sig) in signatures.items():
        signature_def_map[key] = tf.saved_model.signature_def_utils.build_signature_def(inputs={name: tf.saved_model.utils.build_tensor_info(tensor) for (name, tensor) in sig['inputs'].items()}, outputs={name: tf.saved_model.utils.build_tensor_info(tensor) for (name, tensor) in sig['outputs'].items()}, method_name=sig['method_name'] if 'method_name' in sig else key)
    logging.info('===== signature_def_map: {}'.format(signature_def_map))
    builder.add_meta_graph_and_variables(sess, tag_set.split(','), signature_def_map=signature_def_map, clear_devices=True)
    g.finalize()
    builder.save()

def release_port(ctx):
    if False:
        return 10
    'Closes the temporary socket created to assign a port to the TF node.'
    if ctx.tmp_socket is not None:
        logger.info('Releasing assigned port: {}'.format(ctx.tmp_socket.getsockname()))
        ctx.tmp_socket.close()
        ctx.tmp_socket = None
    else:
        logger.warning('release_port() invoked with no bound socket.')

def batch_results(mgr, results, qname='output'):
    if False:
        print('Hello World!')
    '*DEPRECATED*. Use TFNode.DataFeed class instead.'
    raise Exception('DEPRECATED: Use TFNode.DataFeed class instead')

def terminate(mgr, qname='input'):
    if False:
        return 10
    '*DEPRECATED*. Use TFNode.DataFeed class instead.'
    raise Exception('DEPRECATED: Use TFNode.DataFeed class instead')

class DataFeed(object):
    """This class manages the *InputMode.SPARK* data feeding process from the perspective of the TensorFlow application.

  Args:
    :mgr: TFManager instance associated with this Python worker process.
    :train_mode: boolean indicating if the data feed is expecting an output response (e.g. inferencing).
    :qname_in: *INTERNAL_USE*
    :qname_out: *INTERNAL_USE*
    :input_mapping: *For Spark ML Pipelines only*.  Dictionary of input DataFrame columns to input TensorFlow tensors.
  """

    def __init__(self, mgr, train_mode=True, qname_in='input', qname_out='output', input_mapping=None):
        if False:
            for i in range(10):
                print('nop')
        self.mgr = mgr
        self.train_mode = train_mode
        self.qname_in = qname_in
        self.qname_out = qname_out
        self.done_feeding = False
        self.input_tensors = [tensor for (col, tensor) in sorted(input_mapping.items())] if input_mapping is not None else None
        self.queue_in = mgr.get_queue(qname_in)
        self.queue_out = mgr.get_queue(qname_out)

    def next_batch(self, batch_size):
        if False:
            print('Hello World!')
        'Gets a batch of items from the input RDD.\n\n    If multiple tensors are provided per row in the input RDD, e.g. tuple of (tensor1, tensor2, ..., tensorN) and:\n\n    * no ``input_mapping`` was provided to the DataFeed constructor, this will return an array of ``batch_size`` tuples,\n      and the caller is responsible for separating the tensors.\n    * an ``input_mapping`` was provided to the DataFeed constructor, this will return a dictionary of N tensors,\n      with tensor names as keys and arrays of length ``batch_size`` as values.\n\n    Note: if the end of the data is reached, this may return with fewer than ``batch_size`` items.\n\n    Args:\n      :batch_size: number of items to retrieve.\n\n    Returns:\n      A batch of items or a dictionary of tensors.\n    '
        tensors = [] if self.input_tensors is None else {tensor: [] for tensor in self.input_tensors}
        count = 0
        queue_in = self.queue_in
        no_input_tensors = self.input_tensors is None
        while count < batch_size:
            item = queue_in.get(block=True)
            if item is None:
                logger.info('next_batch() got None')
                queue_in.task_done()
                self.done_feeding = True
                break
            elif type(item) is marker.EndPartition:
                logger.info('next_batch() got EndPartition')
                queue_in.task_done()
                if not self.train_mode and count > 0:
                    break
            else:
                if no_input_tensors:
                    tensors.append(item)
                else:
                    for i in range(len(self.input_tensors)):
                        tensors[self.input_tensors[i]].append(item[i])
                count += 1
                queue_in.task_done()
        return tensors

    def should_stop(self):
        if False:
            while True:
                i = 10
        'Check if the feed process was told to stop (by a call to ``terminate``).'
        return self.done_feeding

    def batch_results(self, results):
        if False:
            while True:
                i = 10
        'Push a batch of output results to the Spark output RDD of ``TFCluster.inference()``.\n\n    Note: this currently expects a one-to-one mapping of input to output data, so the length of the ``results`` array should match the length of\n    the previously retrieved batch of input data.\n\n    Args:\n      :results: array of output data for the equivalent batch of input data.\n    '
        queue = self.queue_out
        for item in results:
            queue.put(item, block=True)

    def terminate(self):
        if False:
            while True:
                i = 10
        'Terminate data feeding early.\n\n    Since TensorFlow applications can often terminate on conditions unrelated to the training data (e.g. steps, accuracy, etc),\n    this method signals the data feeding process to ignore any further incoming data.  Note that Spark itself does not have a mechanism\n    to terminate an RDD operation early, so the extra partitions will still be sent to the executors (but will be ignored).  Because\n    of this, you should size your input data accordingly to avoid excessive overhead.\n    '
        logger.info('terminate() invoked')
        self.mgr.set('state', 'terminating')
        queue = self.mgr.get_queue(self.qname_in)
        count = 0
        done = False
        while not done:
            try:
                queue.get(block=True, timeout=5)
                queue.task_done()
                count += 1
            except Empty:
                logger.info('dropped {0} items from queue'.format(count))
                done = True