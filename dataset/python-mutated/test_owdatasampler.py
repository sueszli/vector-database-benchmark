import unittest
from Orange.data import Table
from Orange.widgets.data.owdatasampler import OWDataSampler
from Orange.widgets.tests.base import WidgetTest

class TestOWDataSampler(WidgetTest):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        super().setUpClass()
        cls.iris = Table('iris')
        cls.zoo = Table('zoo')

    def setUp(self):
        if False:
            print('Hello World!')
        self.widget = self.create_widget(OWDataSampler)

    def test_error_message(self):
        if False:
            return 10
        ' Check if error message appears and then disappears when\n        data is removed from input'
        self.widget.controls.sampling_type.buttons[2].click()
        self.send_signal(self.iris)
        self.assertFalse(self.widget.Error.too_many_folds.is_shown())
        self.send_signal(self.iris[:5])
        self.assertTrue(self.widget.Error.too_many_folds.is_shown())
        self.send_signal(None)
        self.assertFalse(self.widget.Error.too_many_folds.is_shown())
        self.send_signal(Table.from_domain(self.iris.domain))
        self.assertTrue(self.widget.Error.no_data.is_shown())

    def test_stratified_on_unbalanced_data(self):
        if False:
            for i in range(10):
                print('nop')
        unbalanced_data = self.iris[:51]
        self.widget.controls.stratify.setChecked(True)
        self.send_signal(unbalanced_data)
        self.assertTrue(self.widget.Warning.could_not_stratify.is_shown())

    def test_bootstrap(self):
        if False:
            i = 10
            return i + 15
        self.select_sampling_type(self.widget.Bootstrap)
        self.send_signal(self.iris)
        in_input = set(self.iris.ids)
        sample = self.get_output(self.widget.Outputs.data_sample)
        in_sample = set(sample.ids)
        in_remaining = set(self.get_output(self.widget.Outputs.remaining_data).ids)
        self.assertEqual(len(sample), len(self.iris))
        self.assertEqual(len(in_sample | in_remaining), len(in_input))
        self.assertEqual(len(in_sample & in_remaining), 0)
        self.assertGreater(len(in_sample), 0)
        self.assertGreater(len(in_remaining), 0)

    def select_sampling_type(self, sampling_type):
        if False:
            i = 10
            return i + 15
        buttons = self.widget.controls.sampling_type.group.buttons()
        buttons[sampling_type].click()

    def test_no_intersection_in_outputs(self):
        if False:
            print('Hello World!')
        ' Check whether outputs intersect and whether length of outputs sums\n        to length of original data'
        self.send_signal(self.iris)
        w = self.widget
        sampling_types = [w.FixedProportion, w.FixedSize, w.CrossValidation]
        for replicable in [True, False]:
            for stratified in [True, False]:
                for sampling_type in sampling_types:
                    self.widget.cb_seed.setChecked(replicable)
                    self.widget.cb_stratify.setChecked(stratified)
                    self.select_sampling_type(sampling_type)
                    self.widget.commit()
                    sample = self.get_output(self.widget.Outputs.data_sample)
                    other = self.get_output(self.widget.Outputs.remaining_data)
                    self.assertEqual(len(self.iris), len(sample) + len(other))
                    self.assertNoIntersection(sample, other)

    def test_bigger_size_with_replacement(self):
        if False:
            while True:
                i = 10
        'Allow bigger output without replacement.'
        self.send_signal(self.iris[:2])
        sample_size = self.set_fixed_sample_size(3, with_replacement=True)
        self.assertEqual(3, sample_size, 'Should be able to set a bigger size with replacement')

    def test_bigger_size_without_replacement(self):
        if False:
            for i in range(10):
                print('nop')
        "Lower output samples to match input's without replacement."
        self.send_signal(self.iris[:2])
        sample_size = self.set_fixed_sample_size(3)
        self.assertEqual(2, sample_size)

    def test_bigger_output_warning(self):
        if False:
            i = 10
            return i + 15
        'Should warn when sample size is bigger than input.'
        self.send_signal(self.iris[:2])
        self.set_fixed_sample_size(3, with_replacement=True)
        self.assertTrue(self.widget.Warning.bigger_sample.is_shown())

    def test_shuffling(self):
        if False:
            i = 10
            return i + 15
        self.send_signal(self.iris)
        self.set_fixed_sample_size(150)
        self.assertFalse(self.widget.Warning.bigger_sample.is_shown())
        sample = self.get_output(self.widget.Outputs.data_sample)
        self.assertTrue((self.iris.ids != sample.ids).any())
        self.assertEqual(set(self.iris.ids), set(sample.ids))
        self.select_sampling_type(self.widget.FixedProportion)
        self.widget.sampleSizePercentage = 100
        self.widget.commit()
        sample = self.get_output(self.widget.Outputs.data_sample)
        self.assertTrue((self.iris.ids != sample.ids).any())
        self.assertEqual(set(self.iris.ids), set(sample.ids))

    def set_fixed_sample_size(self, sample_size, with_replacement=False):
        if False:
            while True:
                i = 10
        'Set fixed sample size and return the number of gui spin.\n\n        Return the actual number in gui so we can check whether it is different\n        from sample_size. The number can be changed depending on the spin\n        maximum value.\n        '
        self.select_sampling_type(self.widget.FixedSize)
        self.widget.controls.replacement.setChecked(with_replacement)
        self.widget.sampleSizeSpin.setValue(sample_size)
        self.widget.commit()
        return self.widget.sampleSizeSpin.value()

    def set_fixed_proportion(self, proportion):
        if False:
            while True:
                i = 10
        'Set fixed sample proportion.\n        '
        self.select_sampling_type(self.widget.FixedProportion)
        self.widget.sampleSizePercentageSlider.setValue(proportion)
        self.widget.commit()

    def assertNoIntersection(self, sample, other):
        if False:
            i = 10
            return i + 15
        self.assertFalse(bool(set(sample.ids) & set(other.ids)))

    def test_cv_outputs(self):
        if False:
            while True:
                i = 10
        w = self.widget
        self.send_signal(w.Inputs.data, self.iris)
        self.select_sampling_type(w.CrossValidation)
        self.widget.commit()
        self.assertEqual(len(self.get_output(w.Outputs.data_sample)), 135)
        self.assertEqual(len(self.get_output(w.Outputs.remaining_data)), 15)

    def test_cv_output_migration(self):
        if False:
            print('Hello World!')
        self.assertFalse(self.widget.compatibility_mode)
        settings = {'sampling_type': OWDataSampler.CrossValidation}
        OWDataSampler.migrate_settings(settings, version=2)
        self.assertFalse(settings.get('compatibility_mode', False))
        settings = {'sampling_type': OWDataSampler.FixedProportion}
        OWDataSampler.migrate_settings(settings, version=1)
        self.assertFalse(settings.get('compatibility_mode', False))
        settings = {'sampling_type': OWDataSampler.CrossValidation}
        OWDataSampler.migrate_settings(settings, version=1)
        self.assertTrue(settings['compatibility_mode'])
        w = self.create_widget(OWDataSampler, stored_settings={'sampling_type': OWDataSampler.CrossValidation, '__version__': 1})
        self.assertTrue(w.compatibility_mode)
        self.send_signal(w.Inputs.data, self.iris)
        self.select_sampling_type(w.CrossValidation)
        w.commit()
        self.assertEqual(len(self.get_output(w.Outputs.data_sample)), 15)
        self.assertEqual(len(self.get_output(w.Outputs.remaining_data)), 135)

    def test_empty_sample(self):
        if False:
            for i in range(10):
                print('nop')
        w = self.widget
        self.send_signal(w.Inputs.data, self.iris)
        self.set_fixed_sample_size(150)
        self.assertEqual(len(self.get_output(w.Outputs.data_sample)), 150)
        self.assertEqual(len(self.get_output(w.Outputs.remaining_data)), 0)
        self.set_fixed_sample_size(0)
        self.assertEqual(len(self.get_output(w.Outputs.data_sample)), 0)
        self.assertEqual(len(self.get_output(w.Outputs.remaining_data)), 150)
        self.set_fixed_proportion(100)
        self.assertEqual(len(self.get_output(w.Outputs.data_sample)), 150)
        self.assertEqual(len(self.get_output(w.Outputs.remaining_data)), 0)
        self.set_fixed_proportion(0)
        self.assertEqual(len(self.get_output(w.Outputs.data_sample)), 0)
        self.assertEqual(len(self.get_output(w.Outputs.remaining_data)), 150)

    def test_send_report(self):
        if False:
            return 10
        w = self.widget
        self.send_signal(w.Inputs.data, self.iris)
        w.stratify = True
        w.use_seed = True
        self.select_sampling_type(0)
        w.commit()
        w.send_report()
        self.select_sampling_type(1)
        w.sampleSizeNumber = 1
        w.commit()
        w.send_report()
        w.sampleSizeNumber = 10
        w.replacement = False
        w.commit()
        w.send_report()
        w.replacement = True
        w.commit()
        w.send_report()
        self.select_sampling_type(2)
        w.commit()
        w.send_report()
        self.select_sampling_type(3)
        w.commit()
        w.send_report()
if __name__ == '__main__':
    unittest.main()