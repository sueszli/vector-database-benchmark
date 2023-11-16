"""Tests for fivo.nested_utils."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import tensorflow as tf
nest = tf.contrib.framework.nest
from fivo import nested_utils
ExampleTuple = collections.namedtuple('ExampleTuple', ['a', 'b'])

class NestedUtilsTest(tf.test.TestCase):

    def test_map_nested_works_on_nested_structures(self):
        if False:
            while True:
                i = 10
        'Check that map_nested works with nested structures.'
        original = [1, (2, 3.2, (4.0, ExampleTuple(5, 6)))]
        expected = [2, (3, 4.2, (5.0, ExampleTuple(6, 7)))]
        out = nested_utils.map_nested(lambda x: x + 1, original)
        self.assertEqual(expected, out)

    def test_map_nested_works_on_single_objects(self):
        if False:
            print('Hello World!')
        'Check that map_nested works with raw objects.'
        original = 1
        expected = 2
        out = nested_utils.map_nested(lambda x: x + 1, original)
        self.assertEqual(expected, out)

    def test_map_nested_works_on_flat_lists(self):
        if False:
            return 10
        'Check that map_nested works with a flat list.'
        original = [1, 2, 3]
        expected = [2, 3, 4]
        out = nested_utils.map_nested(lambda x: x + 1, original)
        self.assertEqual(expected, out)

    def test_tile_tensors(self):
        if False:
            return 10
        'Checks that tile_tensors correctly tiles tensors of different ranks.'
        a = tf.range(20)
        b = tf.reshape(a, [2, 10])
        c = tf.reshape(a, [2, 2, 5])
        a_tiled = tf.tile(a, [3])
        b_tiled = tf.tile(b, [3, 1])
        c_tiled = tf.tile(c, [3, 1, 1])
        tensors = [a, (b, ExampleTuple(c, c))]
        expected_tensors = [a_tiled, (b_tiled, ExampleTuple(c_tiled, c_tiled))]
        tiled = nested_utils.tile_tensors(tensors, [3])
        nest.assert_same_structure(expected_tensors, tiled)
        with self.test_session() as sess:
            (expected, out) = sess.run([expected_tensors, tiled])
            expected = nest.flatten(expected)
            out = nest.flatten(out)
            for (x, y) in zip(expected, out):
                self.assertAllClose(x, y)

    def test_gather_tensors(self):
        if False:
            for i in range(10):
                print('nop')
        a = tf.reshape(tf.range(20), [5, 4])
        inds = [0, 0, 1, 4]
        a_gathered = tf.gather(a, inds)
        tensors = [a, (a, ExampleTuple(a, a))]
        gt_gathered = [a_gathered, (a_gathered, ExampleTuple(a_gathered, a_gathered))]
        gathered = nested_utils.gather_tensors(tensors, inds)
        nest.assert_same_structure(gt_gathered, gathered)
        with self.test_session() as sess:
            (gt, out) = sess.run([gt_gathered, gathered])
            gt = nest.flatten(gt)
            out = nest.flatten(out)
            for (x, y) in zip(gt, out):
                self.assertAllClose(x, y)

    def test_tas_for_tensors(self):
        if False:
            for i in range(10):
                print('nop')
        a = tf.reshape(tf.range(20), [5, 4])
        tensors = [a, (a, ExampleTuple(a, a))]
        tas = nested_utils.tas_for_tensors(tensors, 5)
        nest.assert_same_structure(tensors, tas)
        stacked = nested_utils.map_nested(lambda x: x.stack(), tas)
        with self.test_session() as sess:
            (gt, out) = sess.run([tensors, stacked])
            gt = nest.flatten(gt)
            out = nest.flatten(out)
            for (x, y) in zip(gt, out):
                self.assertAllClose(x, y)

    def test_read_tas(self):
        if False:
            print('Hello World!')
        a = tf.reshape(tf.range(20), [5, 4])
        a_read = a[3, :]
        tensors = [a, (a, ExampleTuple(a, a))]
        gt_read = [a_read, (a_read, ExampleTuple(a_read, a_read))]
        tas = nested_utils.tas_for_tensors(tensors, 5)
        tas_read = nested_utils.read_tas(tas, 3)
        nest.assert_same_structure(tas, tas_read)
        with self.test_session() as sess:
            (gt, out) = sess.run([gt_read, tas_read])
            gt = nest.flatten(gt)
            out = nest.flatten(out)
            for (x, y) in zip(gt, out):
                self.assertAllClose(x, y)
if __name__ == '__main__':
    tf.test.main()