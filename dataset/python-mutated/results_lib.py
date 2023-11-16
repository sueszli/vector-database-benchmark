from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
'Results object manages distributed reading and writing of results to disk.'
import ast
from collections import namedtuple
import os
import re
from six.moves import xrange
import tensorflow as tf
ShardStats = namedtuple('ShardStats', ['num_local_reps_completed', 'max_local_reps', 'finished'])

def ge_non_zero(a, b):
    if False:
        while True:
            i = 10
    return a >= b and b > 0

def get_shard_id(file_name):
    if False:
        i = 10
        return i + 15
    assert file_name[-4:].lower() == '.txt'
    return int(file_name[file_name.rfind('_') + 1:-4])

class Results(object):
    """Manages reading and writing training results to disk asynchronously.

  Each worker writes to its own file, so that there are no race conditions when
  writing happens. However any worker may read any file, as is the case for
  `read_all`. Writes are expected to be atomic so that workers will never
  read incomplete data, and this is likely to be the case on Unix systems.
  Reading out of date data is fine, as workers calling `read_all` will wait
  until data from every worker has been written before proceeding.
  """
    file_template = 'experiment_results_{0}.txt'
    search_regex = '^experiment_results_([0-9])+\\.txt$'

    def __init__(self, log_dir, shard_id=0):
        if False:
            for i in range(10):
                print('nop')
        'Construct `Results` instance.\n\n    Args:\n      log_dir: Where to write results files.\n      shard_id: Unique id for this file (i.e. shard). Each worker that will\n          be writing results should use a different shard id. If there are\n          N shards, each shard should be numbered 0 through N-1.\n    '
        assert 0 <= shard_id
        self.file_name = self.file_template.format(shard_id)
        self.log_dir = log_dir
        self.results_file = os.path.join(self.log_dir, self.file_name)

    def append(self, metrics):
        if False:
            return 10
        'Append results to results list on disk.'
        with tf.gfile.FastGFile(self.results_file, 'a') as writer:
            writer.write(str(metrics) + '\n')

    def read_this_shard(self):
        if False:
            i = 10
            return i + 15
        'Read only from this shard.'
        return self._read_shard(self.results_file)

    def _read_shard(self, results_file):
        if False:
            print('Hello World!')
        'Read only from the given shard file.'
        try:
            with tf.gfile.FastGFile(results_file, 'r') as reader:
                results = [ast.literal_eval(entry) for entry in reader]
        except tf.errors.NotFoundError:
            return []
        return results

    def _get_max_local_reps(self, shard_results):
        if False:
            i = 10
            return i + 15
        "Get maximum number of repetitions the given shard needs to complete.\n\n    Worker working on each shard needs to complete a certain number of runs\n    before it finishes. This method will return that number so that we can\n    determine which shards are still not done.\n\n    We assume that workers are including a 'max_local_repetitions' value in\n    their results, which should be the total number of repetitions it needs to\n    run.\n\n    Args:\n      shard_results: Dict mapping metric names to values. This should be read\n          from a shard on disk.\n\n    Returns:\n      Maximum number of repetitions the given shard needs to complete.\n    "
        mlrs = [r['max_local_repetitions'] for r in shard_results]
        if not mlrs:
            return 0
        for n in mlrs[1:]:
            assert n == mlrs[0], 'Some reps have different max rep.'
        return mlrs[0]

    def read_all(self, num_shards=None):
        if False:
            print('Hello World!')
        'Read results across all shards, i.e. get global results list.\n\n    Args:\n      num_shards: (optional) specifies total number of shards. If the caller\n          wants information about which shards are incomplete, provide this\n          argument (so that shards which have yet to be created are still\n          counted as incomplete shards). Otherwise, no information about\n          incomplete shards will be returned.\n\n    Returns:\n      aggregate: Global list of results (across all shards).\n      shard_stats: List of ShardStats instances, one for each shard. Or None if\n          `num_shards` is None.\n    '
        try:
            all_children = tf.gfile.ListDirectory(self.log_dir)
        except tf.errors.NotFoundError:
            if num_shards is None:
                return ([], None)
            return ([], [[] for _ in xrange(num_shards)])
        shard_ids = {get_shard_id(fname): fname for fname in all_children if re.search(self.search_regex, fname)}
        if num_shards is None:
            aggregate = []
            shard_stats = None
            for results_file in shard_ids.values():
                aggregate.extend(self._read_shard(os.path.join(self.log_dir, results_file)))
        else:
            results_per_shard = [None] * num_shards
            for shard_id in xrange(num_shards):
                if shard_id in shard_ids:
                    results_file = shard_ids[shard_id]
                    results_per_shard[shard_id] = self._read_shard(os.path.join(self.log_dir, results_file))
                else:
                    results_per_shard[shard_id] = []
            shard_stats = []
            for shard_results in results_per_shard:
                max_local_reps = self._get_max_local_reps(shard_results)
                shard_stats.append(ShardStats(num_local_reps_completed=len(shard_results), max_local_reps=max_local_reps, finished=ge_non_zero(len(shard_results), max_local_reps)))
            aggregate = [r for shard_results in results_per_shard for r in shard_results]
        return (aggregate, shard_stats)