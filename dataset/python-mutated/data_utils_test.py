"""Tests for data_utils."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from data import data_utils
data = data_utils

class SequenceWrapperTest(tf.test.TestCase):

    def testDefaultTimesteps(self):
        if False:
            print('Hello World!')
        seq = data.SequenceWrapper()
        t1 = seq.add_timestep()
        _ = seq.add_timestep()
        self.assertEqual(len(seq), 2)
        self.assertEqual(t1.weight, 0.0)
        self.assertEqual(t1.label, 0)
        self.assertEqual(t1.token, 0)

    def testSettersAndGetters(self):
        if False:
            i = 10
            return i + 15
        ts = data.SequenceWrapper().add_timestep()
        ts.set_token(3)
        ts.set_label(4)
        ts.set_weight(2.0)
        self.assertEqual(ts.token, 3)
        self.assertEqual(ts.label, 4)
        self.assertEqual(ts.weight, 2.0)

    def testTimestepIteration(self):
        if False:
            while True:
                i = 10
        seq = data.SequenceWrapper()
        seq.add_timestep().set_token(0)
        seq.add_timestep().set_token(1)
        seq.add_timestep().set_token(2)
        for (i, ts) in enumerate(seq):
            self.assertEqual(ts.token, i)

    def testFillsSequenceExampleCorrectly(self):
        if False:
            while True:
                i = 10
        seq = data.SequenceWrapper()
        seq.add_timestep().set_token(1).set_label(2).set_weight(3.0)
        seq.add_timestep().set_token(10).set_label(20).set_weight(30.0)
        seq_ex = seq.seq
        fl = seq_ex.feature_lists.feature_list
        fl_token = fl[data.SequenceWrapper.F_TOKEN_ID].feature
        fl_label = fl[data.SequenceWrapper.F_LABEL].feature
        fl_weight = fl[data.SequenceWrapper.F_WEIGHT].feature
        _ = [self.assertEqual(len(f), 2) for f in [fl_token, fl_label, fl_weight]]
        self.assertAllEqual([f.int64_list.value[0] for f in fl_token], [1, 10])
        self.assertAllEqual([f.int64_list.value[0] for f in fl_label], [2, 20])
        self.assertAllEqual([f.float_list.value[0] for f in fl_weight], [3.0, 30.0])

class DataUtilsTest(tf.test.TestCase):

    def testSplitByPunct(self):
        if False:
            while True:
                i = 10
        output = data.split_by_punct("hello! world, i've been\nwaiting\tfor\ryou for.a long time")
        expected = ['hello', 'world', 'i', 've', 'been', 'waiting', 'for', 'you', 'for', 'a', 'long', 'time']
        self.assertListEqual(output, expected)

    def _buildDummySequence(self):
        if False:
            for i in range(10):
                print('nop')
        seq = data.SequenceWrapper()
        for i in range(10):
            seq.add_timestep().set_token(i)
        return seq

    def testBuildLMSeq(self):
        if False:
            for i in range(10):
                print('nop')
        seq = self._buildDummySequence()
        lm_seq = data.build_lm_sequence(seq)
        for (i, ts) in enumerate(lm_seq):
            if i == len(lm_seq) - 1:
                self.assertEqual(ts.token, i)
                self.assertEqual(ts.label, i)
                self.assertEqual(ts.weight, 0.0)
            else:
                self.assertEqual(ts.token, i)
                self.assertEqual(ts.label, i + 1)
                self.assertEqual(ts.weight, 1.0)

    def testBuildSAESeq(self):
        if False:
            for i in range(10):
                print('nop')
        seq = self._buildDummySequence()
        sa_seq = data.build_seq_ae_sequence(seq)
        self.assertEqual(len(sa_seq), len(seq) * 2 - 1)
        for (i, ts) in enumerate(sa_seq):
            self.assertEqual(ts.token, seq[i % 10].token)
        for i in range(len(seq) - 1):
            self.assertEqual(sa_seq[i].weight, 0.0)
        for i in range(len(seq) - 1, len(sa_seq)):
            self.assertEqual(sa_seq[i].weight, 1.0)
        for i in range(len(seq) - 1):
            self.assertEqual(sa_seq[i].label, 0)
        for i in range(len(seq) - 1, len(sa_seq)):
            self.assertEqual(sa_seq[i].label, seq[i - (len(seq) - 1)].token)

    def testBuildLabelSeq(self):
        if False:
            return 10
        seq = self._buildDummySequence()
        eos_id = len(seq) - 1
        label_seq = data.build_labeled_sequence(seq, True)
        for (i, ts) in enumerate(label_seq[:-1]):
            self.assertEqual(ts.token, i)
            self.assertEqual(ts.label, 0)
            self.assertEqual(ts.weight, 0.0)
        final_timestep = label_seq[-1]
        self.assertEqual(final_timestep.token, eos_id)
        self.assertEqual(final_timestep.label, 1)
        self.assertEqual(final_timestep.weight, 1.0)

    def testBuildBidirLabelSeq(self):
        if False:
            while True:
                i = 10
        seq = self._buildDummySequence()
        reverse_seq = data.build_reverse_sequence(seq)
        bidir_seq = data.build_bidirectional_seq(seq, reverse_seq)
        label_seq = data.build_labeled_sequence(bidir_seq, True)
        for ((i, ts), j) in zip(enumerate(label_seq[:-1]), reversed(range(len(seq) - 1))):
            self.assertAllEqual(ts.tokens, [i, j])
            self.assertEqual(ts.label, 0)
            self.assertEqual(ts.weight, 0.0)
        final_timestep = label_seq[-1]
        eos_id = len(seq) - 1
        self.assertAllEqual(final_timestep.tokens, [eos_id, eos_id])
        self.assertEqual(final_timestep.label, 1)
        self.assertEqual(final_timestep.weight, 1.0)

    def testReverseSeq(self):
        if False:
            print('Hello World!')
        seq = self._buildDummySequence()
        reverse_seq = data.build_reverse_sequence(seq)
        for (i, ts) in enumerate(reversed(reverse_seq[:-1])):
            self.assertEqual(ts.token, i)
            self.assertEqual(ts.label, 0)
            self.assertEqual(ts.weight, 0.0)
        final_timestep = reverse_seq[-1]
        eos_id = len(seq) - 1
        self.assertEqual(final_timestep.token, eos_id)
        self.assertEqual(final_timestep.label, 0)
        self.assertEqual(final_timestep.weight, 0.0)

    def testBidirSeq(self):
        if False:
            for i in range(10):
                print('nop')
        seq = self._buildDummySequence()
        reverse_seq = data.build_reverse_sequence(seq)
        bidir_seq = data.build_bidirectional_seq(seq, reverse_seq)
        for ((i, ts), j) in zip(enumerate(bidir_seq[:-1]), reversed(range(len(seq) - 1))):
            self.assertAllEqual(ts.tokens, [i, j])
            self.assertEqual(ts.label, 0)
            self.assertEqual(ts.weight, 0.0)
        final_timestep = bidir_seq[-1]
        eos_id = len(seq) - 1
        self.assertAllEqual(final_timestep.tokens, [eos_id, eos_id])
        self.assertEqual(final_timestep.label, 0)
        self.assertEqual(final_timestep.weight, 0.0)

    def testLabelGain(self):
        if False:
            print('Hello World!')
        seq = self._buildDummySequence()
        label_seq = data.build_labeled_sequence(seq, True, label_gain=True)
        for (i, ts) in enumerate(label_seq):
            self.assertEqual(ts.token, i)
            self.assertEqual(ts.label, 1)
            self.assertNear(ts.weight, float(i) / (len(seq) - 1), 0.001)
if __name__ == '__main__':
    tf.test.main()