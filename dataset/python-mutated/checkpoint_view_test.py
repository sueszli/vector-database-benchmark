"""Tests for the checkpoint view."""
import os
from tensorflow.python.checkpoint import checkpoint as trackable_utils
from tensorflow.python.checkpoint import checkpoint_view
from tensorflow.python.eager import test
from tensorflow.python.trackable import autotrackable

class CheckpointViewTest(test.TestCase):

    def test_children(self):
        if False:
            print('Hello World!')
        root = autotrackable.AutoTrackable()
        root.leaf = autotrackable.AutoTrackable()
        root_ckpt = trackable_utils.Checkpoint(root=root)
        root_save_path = root_ckpt.save(os.path.join(self.get_temp_dir(), 'root_ckpt'))
        (current_name, node_id) = next(iter(checkpoint_view.CheckpointView(root_save_path).children(0).items()))
        self.assertEqual('leaf', current_name)
        self.assertEqual(1, node_id)

    def test_all_nodes(self):
        if False:
            print('Hello World!')
        root = autotrackable.AutoTrackable()
        root.leaf = autotrackable.AutoTrackable()
        root_ckpt = trackable_utils.Checkpoint(root=root)
        root_save_path = root_ckpt.save(os.path.join(self.get_temp_dir(), 'root_ckpt'))
        all_nodes = checkpoint_view.CheckpointView(root_save_path).descendants()
        self.assertEqual(3, len(all_nodes))
        self.assertEqual(0, all_nodes[0])
        self.assertEqual(1, all_nodes[1])

    def test_all_nodes_with_paths(self):
        if False:
            print('Hello World!')
        root = autotrackable.AutoTrackable()
        leaf1 = root.leaf1 = autotrackable.AutoTrackable()
        leaf2 = root.leaf2 = autotrackable.AutoTrackable()
        leaf1.leaf3 = autotrackable.AutoTrackable()
        leaf1.leaf4 = autotrackable.AutoTrackable()
        leaf2.leaf5 = autotrackable.AutoTrackable()
        root_ckpt = trackable_utils.Checkpoint(root=root)
        root_save_path = root_ckpt.save(os.path.join(self.get_temp_dir(), 'root_ckpt'))
        all_nodes_with_paths = checkpoint_view.CheckpointView(root_save_path)._descendants_with_paths()
        self.assertEqual({'root', 'root.leaf1', 'root.leaf2', 'root.save_counter', 'root.leaf1.leaf3', 'root.leaf1.leaf4', 'root.leaf2.leaf5'}, set(all_nodes_with_paths.values()))

    def test_match(self):
        if False:
            for i in range(10):
                print('nop')
        root1 = autotrackable.AutoTrackable()
        leaf1 = root1.leaf1 = autotrackable.AutoTrackable()
        leaf2 = root1.leaf2 = autotrackable.AutoTrackable()
        leaf1.leaf3 = autotrackable.AutoTrackable()
        leaf1.leaf4 = autotrackable.AutoTrackable()
        leaf2.leaf5 = autotrackable.AutoTrackable()
        root_ckpt = trackable_utils.Checkpoint(root=root1)
        root_save_path = root_ckpt.save(os.path.join(self.get_temp_dir(), 'root_ckpt'))
        root2 = autotrackable.AutoTrackable()
        leaf11 = root2.leaf1 = autotrackable.AutoTrackable()
        leaf12 = root2.leaf2 = autotrackable.AutoTrackable()
        leaf13 = leaf11.leaf3 = autotrackable.AutoTrackable()
        leaf15 = leaf12.leaf5 = autotrackable.AutoTrackable()
        matching_nodes = checkpoint_view.CheckpointView(root_save_path).match(root2)
        self.assertDictEqual(matching_nodes, {0: root2, 1: leaf11, 2: leaf12, 4: leaf13, 6: leaf15})

    def test_match_overlapping_nodes(self):
        if False:
            for i in range(10):
                print('nop')
        root1 = autotrackable.AutoTrackable()
        root1.a = root1.b = autotrackable.AutoTrackable()
        root_ckpt = trackable_utils.Checkpoint(root=root1)
        root_save_path = root_ckpt.save(os.path.join(self.get_temp_dir(), 'root_ckpt'))
        root2 = autotrackable.AutoTrackable()
        a1 = root2.a = autotrackable.AutoTrackable()
        root2.b = autotrackable.AutoTrackable()
        with self.assertLogs(level='WARNING') as logs:
            matching_nodes = checkpoint_view.CheckpointView(root_save_path).match(root2)
        self.assertDictEqual(matching_nodes, {0: root2, 1: a1})
        expected_message = 'Inconsistent references when matching the checkpoint into this object graph.'
        self.assertIn(expected_message, logs.output[0])

    def test_diff(self):
        if False:
            i = 10
            return i + 15
        root1 = autotrackable.AutoTrackable()
        leaf1 = root1.leaf1 = autotrackable.AutoTrackable()
        leaf2 = root1.leaf2 = autotrackable.AutoTrackable()
        leaf1.leaf3 = autotrackable.AutoTrackable()
        leaf1.leaf4 = autotrackable.AutoTrackable()
        leaf2.leaf5 = autotrackable.AutoTrackable()
        root_ckpt = trackable_utils.Checkpoint(root=root1)
        root_save_path = root_ckpt.save(os.path.join(self.get_temp_dir(), 'root_ckpt'))
        root2 = autotrackable.AutoTrackable()
        leaf11 = root2.leaf1 = autotrackable.AutoTrackable()
        leaf12 = root2.leaf2 = autotrackable.AutoTrackable()
        leaf13 = leaf11.leaf3 = autotrackable.AutoTrackable()
        leaf15 = leaf12.leaf5 = autotrackable.AutoTrackable()
        leaf16 = leaf12.leaf6 = autotrackable.AutoTrackable()
        diff = checkpoint_view.CheckpointView(root_save_path).diff(root2)
        self.assertEqual(len(diff), 3)
        self.assertDictEqual(diff[0], {0: root2, 1: leaf11, 2: leaf12, 4: leaf13, 6: leaf15})
        self.assertListEqual(diff[1], [3, 5])
        self.assertListEqual(diff[2], [leaf16])
if __name__ == '__main__':
    test.main()