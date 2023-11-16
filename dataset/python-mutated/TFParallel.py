from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function
import logging
from . import TFSparkNode
from . import util
logger = logging.getLogger(__name__)

def run(sc, map_fn, tf_args, num_executors, use_barrier=True):
    if False:
        print('Hello World!')
    'Runs the user map_fn as parallel, independent instances of TF on the Spark executors.\n\n  Args:\n    :sc: SparkContext\n    :map_fun: user-supplied TensorFlow "main" function\n    :tf_args: ``argparse`` args, or command-line ``ARGV``.  These will be passed to the ``map_fun``.\n    :num_executors: number of Spark executors.  This should match your Spark job\'s ``--num_executors``.\n    :use_barrier: Boolean indicating if TFParallel should use Spark barrier execution mode to wait for all executors.\n\n  Returns:\n    None\n  '
    defaultFS = sc._jsc.hadoopConfiguration().get('fs.defaultFS')
    if defaultFS.startswith('file://') and len(defaultFS) > 7 and defaultFS.endswith('/'):
        defaultFS = defaultFS[:-1]

    def _run(it):
        if False:
            print('Hello World!')
        from pyspark import BarrierTaskContext
        for i in it:
            worker_num = i
        if use_barrier:
            barrier_ctx = BarrierTaskContext.get()
            tasks = barrier_ctx.getTaskInfos()
            nodes = [t.address for t in tasks]
            num_workers = len(nodes)
        else:
            nodes = []
            num_workers = num_executors
        num_gpus = tf_args.num_gpus if 'num_gpus' in tf_args else 1
        util.single_node_env(num_gpus=num_gpus, worker_index=worker_num, nodes=nodes)
        ctx = TFSparkNode.TFNodeContext()
        ctx.defaultFS = defaultFS
        ctx.worker_num = worker_num
        ctx.executor_id = worker_num
        ctx.num_workers = num_workers
        map_fn(tf_args, ctx)
        return [0]
    nodeRDD = sc.parallelize(list(range(num_executors)), num_executors)
    if use_barrier:
        nodeRDD.barrier().mapPartitions(_run).collect()
    else:
        nodeRDD.mapPartitions(_run).collect()