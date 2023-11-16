"""
This module provides a high-level API to manage the TensorFlowOnSpark cluster.

There are three main phases of operation:

1. **Reservation/Startup** - reserves a port for the TensorFlow process on each executor, starts a multiprocessing.Manager to
   listen for data/control messages, and then launches the Tensorflow main function on the executors.

2. **Data feeding** - *For InputMode.SPARK only*. Sends RDD data to the TensorFlow nodes via each executor's multiprocessing.Manager.  PS
   nodes will tie up their executors, so they won't receive any subsequent data feeding tasks.

3. **Shutdown** - sends a shutdown control message to the multiprocessing.Managers of the PS nodes and pushes end-of-feed markers into the data
   queues of the worker nodes.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function
import logging
import os
import random
import signal
import sys
import threading
import time
from pyspark.streaming import DStream
from . import reservation
from . import TFManager
from . import TFSparkNode
logger = logging.getLogger(__name__)
tf_status = {}

class InputMode(object):
    """Enum for the input modes of data feeding."""
    TENSORFLOW = 0
    SPARK = 1

class TFCluster(object):
    sc = None
    defaultFS = None
    working_dir = None
    num_executors = None
    nodeRDD = None
    cluster_id = None
    cluster_info = None
    cluster_meta = None
    input_mode = None
    queues = None
    server = None

    def train(self, dataRDD, num_epochs=0, feed_timeout=600, qname='input'):
        if False:
            while True:
                i = 10
        '*For InputMode.SPARK only*.  Feeds Spark RDD partitions into the TensorFlow worker nodes\n\n    It is the responsibility of the TensorFlow "main" function to interpret the rows of the RDD.\n\n    Since epochs are implemented via ``RDD.union()`` and the entire RDD must generally be processed in full, it is recommended\n    to set ``num_epochs`` to closely match your training termination condition (e.g. steps or accuracy).  See ``TFNode.DataFeed``\n    for more details.\n\n    Args:\n      :dataRDD: input data as a Spark RDD.\n      :num_epochs: number of times to repeat the dataset during training.\n      :feed_timeout: number of seconds after which data feeding times out (600 sec default)\n      :qname: *INTERNAL USE*.\n    '
        logger.info('Feeding training data')
        assert self.input_mode == InputMode.SPARK, 'TFCluster.train() requires InputMode.SPARK'
        assert qname in self.queues, 'Unknown queue: {}'.format(qname)
        assert num_epochs >= 0, 'num_epochs cannot be negative'
        if isinstance(dataRDD, DStream):
            dataRDD.foreachRDD(lambda rdd: rdd.foreachPartition(TFSparkNode.train(self.cluster_info, self.cluster_meta, feed_timeout=feed_timeout, qname=qname)))
        else:
            if num_epochs == 0:
                num_epochs = 10
            rdds = [dataRDD] * num_epochs
            unionRDD = self.sc.union(rdds)
            unionRDD.foreachPartition(TFSparkNode.train(self.cluster_info, self.cluster_meta, feed_timeout=feed_timeout, qname=qname))

    def inference(self, dataRDD, feed_timeout=600, qname='input'):
        if False:
            for i in range(10):
                print('nop')
        '*For InputMode.SPARK only*: Feeds Spark RDD partitions into the TensorFlow worker nodes and returns an RDD of results\n\n    It is the responsibility of the TensorFlow "main" function to interpret the rows of the RDD and provide valid data for the output RDD.\n\n    This will use the distributed TensorFlow cluster for inferencing, so the TensorFlow "main" function should be capable of inferencing.\n    Per Spark design, the output RDD will be lazily-executed only when a Spark action is invoked on the RDD.\n\n    Args:\n      :dataRDD: input data as a Spark RDD\n      :feed_timeout: number of seconds after which data feeding times out (600 sec default)\n      :qname: *INTERNAL_USE*\n\n    Returns:\n      A Spark RDD representing the output of the TensorFlow inferencing\n    '
        logger.info('Feeding inference data')
        assert self.input_mode == InputMode.SPARK, 'TFCluster.inference() requires InputMode.SPARK'
        assert qname in self.queues, 'Unknown queue: {}'.format(qname)
        return dataRDD.mapPartitions(TFSparkNode.inference(self.cluster_info, feed_timeout=feed_timeout, qname=qname))

    def shutdown(self, ssc=None, grace_secs=0, timeout=259200):
        if False:
            i = 10
            return i + 15
        'Stops the distributed TensorFlow cluster.\n\n    For InputMode.SPARK, this will be executed AFTER the `TFCluster.train()` or `TFCluster.inference()` method completes.\n    For InputMode.TENSORFLOW, this will be executed IMMEDIATELY after `TFCluster.run()` and will wait until the TF worker nodes complete.\n\n    Args:\n      :ssc: *For Streaming applications only*. Spark StreamingContext\n      :grace_secs: Grace period to wait after all executors have completed their tasks before terminating the Spark application, e.g. to allow the chief worker to perform any final/cleanup duties like exporting or evaluating the model.  Default is 0.\n      :timeout: Time in seconds to wait for TF cluster to complete before terminating the Spark application.  This can be useful if the TF code hangs for any reason.  Default is 3 days.  Use -1 to disable timeout.\n    '
        logger.info('Waiting for TensorFlow nodes to complete...')
        (ps_list, worker_list, eval_list) = ([], [], [])
        for node in self.cluster_info:
            (ps_list if node['job_name'] == 'ps' else eval_list if node['job_name'] == 'evaluator' else worker_list).append(node)
        if timeout > 0:

            def timeout_handler(signum, frame):
                if False:
                    for i in range(10):
                        print('nop')
                logger.error('TensorFlow execution timed out, exiting Spark application with error status')
                self.sc.cancelAllJobs()
                self.sc.stop()
                sys.exit(1)
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
        if ssc is not None:
            while not ssc.awaitTerminationOrTimeout(1):
                if self.server.done:
                    logger.info('Server done, stopping StreamingContext')
                    ssc.stop(stopSparkContext=False, stopGraceFully=True)
                    break
        elif self.input_mode == InputMode.TENSORFLOW:
            count = 0
            while count < 3:
                st = self.sc.statusTracker()
                jobs = st.getActiveJobsIds()
                if len(jobs) == 0:
                    break
                stages = st.getActiveStageIds()
                for i in stages:
                    si = st.getStageInfo(i)
                    if si.numActiveTasks == len(ps_list) + len(eval_list):
                        count += 1
                time.sleep(5)
        workers = len(worker_list)
        workerRDD = self.sc.parallelize(range(workers), workers)
        workerRDD.foreachPartition(TFSparkNode.shutdown(self.cluster_info, grace_secs, self.queues))
        if 'error' in tf_status:
            logger.error('Exiting Spark application with error status.')
            self.sc.cancelAllJobs()
            self.sc.stop()
            sys.exit(1)
        logger.info('Shutting down cluster')
        for node in ps_list + eval_list:
            addr = node['addr']
            authkey = node['authkey']
            m = TFManager.connect(addr, authkey)
            q = m.get_queue('control')
            q.put(None)
            q.join()
        while True:
            time.sleep(5)
            st = self.sc.statusTracker()
            jobs = st.getActiveJobsIds()
            if len(jobs) == 0:
                break
        self.server.stop()

    def tensorboard_url(self):
        if False:
            print('Hello World!')
        'Utility function to get the Tensorboard URL'
        for node in self.cluster_info:
            if node['tb_port'] != 0:
                return 'http://{0}:{1}'.format(node['host'], node['tb_port'])
        return None

def run(sc, map_fun, tf_args, num_executors, num_ps, tensorboard=False, input_mode=InputMode.TENSORFLOW, log_dir=None, driver_ps_nodes=False, master_node=None, reservation_timeout=600, queues=['input', 'output', 'error'], eval_node=False, release_port=True):
    if False:
        while True:
            i = 10
    'Starts the TensorFlowOnSpark cluster and Runs the TensorFlow "main" function on the Spark executors\n\n  Args:\n    :sc: SparkContext\n    :map_fun: user-supplied TensorFlow "main" function\n    :tf_args: ``argparse`` args, or command-line ``ARGV``.  These will be passed to the ``map_fun``.\n    :num_executors: number of Spark executors.  This should match your Spark job\'s ``--num_executors``.\n    :num_ps: number of Spark executors which are reserved for TensorFlow PS nodes.  All other executors will be used as TensorFlow worker nodes.\n    :tensorboard: boolean indicating if the chief worker should spawn a Tensorboard server.\n    :input_mode: TFCluster.InputMode\n    :log_dir: directory to save tensorboard event logs.  If None, defaults to a fixed path on local filesystem.\n    :driver_ps_nodes: run the PS nodes on the driver locally instead of on the spark executors; this help maximizing computing resources (esp. GPU). You will need to set cluster_size = num_executors + num_ps\n    :master_node: name of the "master" or "chief" node in the cluster_template, used for `tf.estimator` applications.\n    :reservation_timeout: number of seconds after which cluster reservation times out (600 sec default)\n    :queues: *INTERNAL_USE*\n    :eval_node: run evaluator node for distributed Tensorflow\n    :release_port: automatically release reserved port prior to invoking user\'s map_function.  If False, user\'s map_function must invoke ctx.release_port() prior to starting TF GRPC server.\n\n  Returns:\n    A TFCluster object representing the started cluster.\n  '
    logger.info('Reserving TFSparkNodes {0}'.format('w/ TensorBoard' if tensorboard else ''))
    if driver_ps_nodes and input_mode != InputMode.TENSORFLOW:
        raise Exception('running PS nodes on driver locally is only supported in InputMode.TENSORFLOW')
    if eval_node and input_mode != InputMode.TENSORFLOW:
        raise Exception('running evaluator nodes is only supported in InputMode.TENSORFLOW')
    num_master = 1 if master_node else 0
    num_eval = 1 if eval_node else 0
    num_workers = max(num_executors - num_ps - num_eval - num_master, 0)
    total_nodes = num_ps + num_master + num_eval + num_workers
    assert total_nodes == num_executors, 'TensorFlow cluster requires {} nodes, but only {} executors available'.format(total_nodes, num_executors)
    assert num_master + num_workers > 0, 'TensorFlow cluster requires at least one worker or master/chief node'
    executors = list(range(num_executors))
    cluster_template = {}
    if num_ps > 0:
        cluster_template['ps'] = executors[:num_ps]
        del executors[:num_ps]
    if master_node:
        cluster_template[master_node] = executors[:1]
        del executors[:1]
    if eval_node:
        cluster_template['evaluator'] = executors[:1]
        del executors[:1]
    if num_workers > 0:
        cluster_template['worker'] = executors[:num_workers]
    logger.info('cluster_template: {}'.format(cluster_template))
    defaultFS = sc._jsc.hadoopConfiguration().get('fs.defaultFS')
    if defaultFS.startswith('file://') and len(defaultFS) > 7 and defaultFS.endswith('/'):
        defaultFS = defaultFS[:-1]
    working_dir = os.getcwd()
    server = reservation.Server(num_executors)
    server_addr = server.start()
    logger.info('Starting TensorFlow on executors')
    cluster_meta = {'id': random.getrandbits(64), 'cluster_template': cluster_template, 'num_executors': num_executors, 'default_fs': defaultFS, 'working_dir': working_dir, 'server_addr': server_addr, 'release_port': release_port}
    if driver_ps_nodes:
        nodeRDD = sc.parallelize(range(num_ps, num_executors), num_executors - num_ps)
    else:
        nodeRDD = sc.parallelize(range(num_executors), num_executors)
    if driver_ps_nodes:

        def _start_ps(node_index):
            if False:
                i = 10
                return i + 15
            logger.info('starting ps node locally %d' % node_index)
            TFSparkNode.run(map_fun, tf_args, cluster_meta, tensorboard, log_dir, queues, background=input_mode == InputMode.SPARK)([node_index])
        for i in cluster_template['ps']:
            ps_thread = threading.Thread(target=lambda : _start_ps(i))
            ps_thread.daemon = True
            ps_thread.start()

    def _start(status):
        if False:
            while True:
                i = 10
        try:
            nodeRDD.foreachPartition(TFSparkNode.run(map_fun, tf_args, cluster_meta, tensorboard, log_dir, queues, background=input_mode == InputMode.SPARK))
        except Exception as e:
            logger.error('Exception in TF background thread: {}'.format(e))
            status['error'] = str(e)
    t = threading.Thread(target=_start, args=(tf_status,))
    t.daemon = True
    t.start()
    logger.info('Waiting for TFSparkNodes to start')
    cluster_info = server.await_reservations(sc, tf_status, reservation_timeout)
    logger.info('All TFSparkNodes started')
    tb_url = None
    for node in cluster_info:
        logger.info(node)
        if node['tb_port'] != 0:
            tb_url = 'http://{0}:{1}'.format(node['host'], node['tb_port'])
    if tb_url is not None:
        logger.info('========================================================================================')
        logger.info('')
        logger.info('TensorBoard running at:       {0}'.format(tb_url))
        logger.info('')
        logger.info('========================================================================================')
    tb_nodes = set()
    for node in cluster_info:
        node_id = (node['host'], node['executor_id'])
        if node_id in tb_nodes:
            msg = '\nDuplicate cluster node id detected (host={0}, executor_id={1})\nPlease ensure that:\n1. Number of executors >= number of TensorFlow nodes\n2. Number of tasks per executors is 1\n3, TFCluster.shutdown() is successfully invoked when done.\n'.strip()
            raise Exception(msg.format(node_id[0], node_id[1]))
        else:
            tb_nodes.add(node_id)
    cluster = TFCluster()
    cluster.sc = sc
    cluster.meta = cluster_meta
    cluster.nodeRDD = nodeRDD
    cluster.cluster_info = cluster_info
    cluster.cluster_meta = cluster_meta
    cluster.input_mode = input_mode
    cluster.queues = queues
    cluster.server = server
    return cluster