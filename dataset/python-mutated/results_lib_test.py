from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
'Tests for results_lib.'
import contextlib
import os
import shutil
import tempfile
from six.moves import xrange
import tensorflow as tf
from single_task import results_lib

@contextlib.contextmanager
def temporary_directory(suffix='', prefix='tmp', base_path=None):
    if False:
        return 10
    'A context manager to create a temporary directory and clean up on exit.\n\n  The parameters are the same ones expected by tempfile.mkdtemp.\n  The directory will be securely and atomically created.\n  Everything under it will be removed when exiting the context.\n\n  Args:\n    suffix: optional suffix.\n    prefix: options prefix.\n    base_path: the base path under which to create the temporary directory.\n  Yields:\n    The absolute path of the new temporary directory.\n  '
    temp_dir_path = tempfile.mkdtemp(suffix, prefix, base_path)
    try:
        yield temp_dir_path
    finally:
        try:
            shutil.rmtree(temp_dir_path)
        except OSError as e:
            if e.message == 'Cannot call rmtree on a symbolic link':
                os.unlink(temp_dir_path)
            else:
                raise

def freeze(dictionary):
    if False:
        return 10
    'Convert dict to hashable frozenset.'
    return frozenset(dictionary.iteritems())

class ResultsLibTest(tf.test.TestCase):

    def testResults(self):
        if False:
            for i in range(10):
                print('nop')
        with temporary_directory() as logdir:
            results_obj = results_lib.Results(logdir)
            self.assertEqual(results_obj.read_this_shard(), [])
            results_obj.append({'foo': 1.5, 'bar': 2.5, 'baz': 0})
            results_obj.append({'foo': 5.5, 'bar': -1, 'baz': 2})
            self.assertEqual(results_obj.read_this_shard(), [{'foo': 1.5, 'bar': 2.5, 'baz': 0}, {'foo': 5.5, 'bar': -1, 'baz': 2}])

    def testShardedResults(self):
        if False:
            while True:
                i = 10
        with temporary_directory() as logdir:
            n = 4
            results_objs = [results_lib.Results(logdir, shard_id=i) for i in xrange(n)]
            for (i, robj) in enumerate(results_objs):
                robj.append({'foo': i, 'bar': 1 + i * 2})
            (results_list, _) = results_objs[0].read_all()
            self.assertEqual(set((freeze(r) for r in results_list)), set((freeze({'foo': i, 'bar': 1 + i * 2}) for i in xrange(n))))
if __name__ == '__main__':
    tf.test.main()