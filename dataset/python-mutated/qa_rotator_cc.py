from gnuradio import gr, gr_unittest
from gnuradio import blocks
import numpy as np
from numpy.random import uniform
import pmt

class qa_rotator_cc(gr_unittest.TestCase):

    def _setUp(self, n_samples=100, tag_inc_updates=True):
        if False:
            print('Hello World!')
        'Base fixture: set up flowgraph and parameters'
        self.n_samples = n_samples
        self.f_in = uniform(high=0.5)
        self.f_shift = uniform(high=0.5) - self.f_in
        in_angles = 2 * np.pi * np.arange(self.n_samples) * self.f_in
        in_samples = np.exp(1j * in_angles)
        phase_inc = 2 * np.pi * self.f_shift
        self.tb = gr.top_block()
        self.source = blocks.vector_source_c(in_samples)
        self.throttle = blocks.throttle(gr.sizeof_gr_complex, 2 ** 16)
        self.rotator_cc = blocks.rotator_cc(phase_inc, tag_inc_updates)
        self.sink = blocks.vector_sink_c()
        self.tag_sink = blocks.tag_debug(gr.sizeof_gr_complex, 'rot_phase_inc', 'rot_phase_inc')
        self.tag_sink.set_save_all(True)
        self.tb.connect(self.source, self.throttle, self.rotator_cc, self.sink)
        self.tb.connect(self.rotator_cc, self.tag_sink)

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self._setUp()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.tb = None

    def _post_phase_inc_cmd(self, new_phase_inc, offset=None):
        if False:
            return 10
        'Post phase increment update command to the rotator block'
        cmd = pmt.make_dict()
        cmd = pmt.dict_add(cmd, pmt.intern('inc'), pmt.from_double(new_phase_inc))
        if offset is not None:
            cmd = pmt.dict_add(cmd, pmt.intern('offset'), pmt.from_uint64(offset))
        self.rotator_cc.insert_tail(pmt.to_pmt('cmd'), cmd)

    def _assert_tags(self, expected_values, expected_offsets):
        if False:
            for i in range(10):
                print('nop')
        'Check the tags received by the tag debug block'
        tags = self.tag_sink.current_tags()
        expected_tags = list(zip(expected_values, expected_offsets))
        self.assertEqual(len(tags), len(expected_tags))
        for (idx, (val, offset)) in enumerate(expected_tags):
            self.assertAlmostEqual(pmt.to_double(tags[idx].value), val, places=5)
            self.assertEqual(tags[idx].offset, offset)

    def _compute_expected_samples(self, offsets, new_phase_incs):
        if False:
            print('Hello World!')
        'Compute the samples expected on the rotator output\n\n        Args:\n            offsets (list): Sample offsets where the updates are expected.\n            new_phase_incs (list): Rotator phase increments for each update.\n\n        Returns:\n            np.array: Array of expected IQ samples on the rotator output.\n\n        '
        f_out = self.f_in + self.f_shift
        expected_angles = 2 * np.pi * np.arange(offsets[0]) * f_out
        expected_samples = np.exp(1j * expected_angles)
        for (idx, (offset, rot_phase_inc)) in enumerate(zip(offsets, new_phase_incs)):
            prev_f_out = f_out
            new_f_shift = rot_phase_inc / (2.0 * np.pi)
            f_out = self.f_in + new_f_shift
            if idx == len(offsets) - 1:
                segment_len = self.n_samples - offset
            else:
                segment_len = offsets[idx + 1] - offset
            expected_angles = expected_angles[-1] + 2 * np.pi * prev_f_out + 2 * np.pi * np.arange(segment_len) * f_out
            expected_samples = np.concatenate((expected_samples, np.exp(1j * expected_angles)))
        return expected_samples

    def _post_random_phase_inc_updates(self, offsets):
        if False:
            while True:
                i = 10
        'Update the phase increment randomly at chosen offsets\n\n        Args:\n            offsets (list): Sample offsets where the updates are to be applied.\n\n        Returns:\n            list: New phase increments defined randomly (list).\n\n        '
        new_phase_incs = list()
        for offset in offsets:
            new_f_out = uniform(high=0.5)
            new_f_shift = new_f_out - self.f_in
            new_phase_inc = float(2 * np.pi * new_f_shift)
            new_phase_incs.append(new_phase_inc)
            self._post_phase_inc_cmd(new_phase_inc, offset)
        return new_phase_incs

    def test_freq_shift(self):
        if False:
            i = 10
            return i + 15
        'Complex sinusoid frequency shift'
        f_out = self.f_in + self.f_shift
        expected_angles = 2 * np.pi * np.arange(self.n_samples) * f_out
        expected_samples = np.exp(1j * expected_angles)
        self.tb.run()
        self.assertComplexTuplesAlmostEqual(self.sink.data(), expected_samples, places=4)

    def test_scheduled_phase_inc_update(self):
        if False:
            i = 10
            return i + 15
        'Update the phase increment at a chosen offset via command message'
        offset = int(self.n_samples / 2)
        offsets = [offset]
        new_phase_incs = self._post_random_phase_inc_updates(offsets)
        expected_samples = self._compute_expected_samples(offsets, new_phase_incs)
        self.tb.run()
        self._assert_tags(new_phase_incs, offsets)
        self.assertComplexTuplesAlmostEqual(self.sink.data(), expected_samples, places=4)

    def test_scheduled_phase_inc_update_with_tagging_disabled(self):
        if False:
            while True:
                i = 10
        'Test a scheduled phase increment update without tagging the update\n\n        Same as test_scheduled_phase_inc_update but with tagging disabled.\n\n        '
        self._setUp(tag_inc_updates=False)
        offset = int(self.n_samples / 2)
        offsets = [offset]
        new_phase_incs = self._post_random_phase_inc_updates(offsets)
        expected_samples = self._compute_expected_samples(offsets, new_phase_incs)
        self.tb.run()
        tags = self.tag_sink.current_tags()
        self.assertEqual(len(tags), 0)
        self.assertComplexTuplesAlmostEqual(self.sink.data(), expected_samples, places=4)

    def test_immediate_phase_inc_update(self):
        if False:
            i = 10
            return i + 15
        'Immediate phase increment update via command message\n\n        In this test, the command message does not include the offset\n        key. Hence, the rotator should update its phase increment immediately.\n\n        '
        new_f_shift = uniform(high=0.5) - self.f_in
        new_phase_inc = float(2 * np.pi * new_f_shift)
        f_out = self.f_in + new_f_shift
        self._post_phase_inc_cmd(new_phase_inc)
        expected_tag_offset = 0
        expected_angles = 2 * np.pi * np.arange(self.n_samples) * f_out
        expected_samples = np.exp(1j * expected_angles)
        self.tb.run()
        self._assert_tags([new_phase_inc], [expected_tag_offset])
        self.assertComplexTuplesAlmostEqual(self.sink.data(), expected_samples, places=4)

    def test_zero_change_phase_inc_update(self):
        if False:
            print('Hello World!')
        'Schedule a phase increment update that does not change anything\n\n        If the scheduled phase increment update sets the same phase increment\n        that is already active in the rotator block, there should be no effect\n        on the output signal. Nevertheless, the rotator should still tag the\n        update.\n\n        '
        new_phase_inc = 2 * np.pi * self.f_shift
        offset = int(self.n_samples / 2)
        f_out = self.f_in + self.f_shift
        self._post_phase_inc_cmd(new_phase_inc, offset)
        expected_angles = 2 * np.pi * np.arange(self.n_samples) * f_out
        expected_samples = np.exp(1j * expected_angles)
        self.tb.run()
        self._assert_tags([new_phase_inc], [offset])
        self.assertComplexTuplesAlmostEqual(self.sink.data(), expected_samples, places=4)

    def test_consecutive_phase_inc_updates(self):
        if False:
            return 10
        'Test tagging of a few consecutive phase increment updates'
        offsets = list(map(int, self.n_samples * np.arange(1, 4, 1) / 4))
        new_phase_incs = self._post_random_phase_inc_updates(offsets)
        expected_samples = self._compute_expected_samples(offsets, new_phase_incs)
        self.tb.run()
        self._assert_tags(new_phase_incs, offsets)
        self.assertComplexTuplesAlmostEqual(self.sink.data(), expected_samples, places=4)

    def test_out_of_order_phase_inc_updates(self):
        if False:
            for i in range(10):
                print('nop')
        'Test tagging of a few out-of-order phase increment updates\n\n        The rotator should sort the increment updates and apply them in order.\n\n        '
        n_updates = 3
        new_f_shifts = uniform(high=0.5, size=n_updates)
        new_phase_incs = 2 * np.pi * new_f_shifts
        offsets = self.n_samples * np.arange(1, 4, 1) / 4
        for i in [0, 2, 1]:
            self._post_phase_inc_cmd(new_phase_incs[i], int(offsets[i]))
        self.tb.run()
        self._assert_tags(new_phase_incs, offsets)
        expected_samples = self._compute_expected_samples(offsets, new_phase_incs)
        self.assertComplexTuplesAlmostEqual(self.sink.data(), expected_samples, places=4)

    def test_duplicate_phase_inc_updates(self):
        if False:
            for i in range(10):
                print('nop')
        'Test multiple phase increment updates scheduled for the same sample\n\n        The rotator block applies all updates scheduled for the same sample\n        offset. In the end, only the last update shall take effect.\n\n        '
        n_updates = 3
        offset = int(self.n_samples / 2)
        all_new_phase_incs = list()
        for i in range(n_updates):
            new_phase_incs = self._post_random_phase_inc_updates([offset])
            expected_samples = self._compute_expected_samples([offset], new_phase_incs)
            all_new_phase_incs.extend(new_phase_incs)
        self.tb.run()
        self._assert_tags(all_new_phase_incs, [offset] * n_updates)
        self.assertComplexTuplesAlmostEqual(self.sink.data(), expected_samples, places=4)

    def test_phase_inc_update_out_of_range(self):
        if False:
            i = 10
            return i + 15
        'Test phase increment update sent for an out-of-range offset'
        self._setUp(n_samples=2 ** 16)
        n_half_samples = int(self.n_samples / 2)
        new_phase_inc = 2 * np.pi * 0.1
        self._post_phase_inc_cmd(new_phase_inc, offset=n_half_samples)
        self.tb.start()
        while self.rotator_cc.nitems_written(0) == 0:
            pass
        self._assert_tags([], [])
        while self.rotator_cc.nitems_written(0) < n_half_samples:
            pass
        self._post_phase_inc_cmd(new_phase_inc, offset=0)
        self.tb.wait()
        self._assert_tags([new_phase_inc], [n_half_samples])
if __name__ == '__main__':
    gr_unittest.run(qa_rotator_cc)