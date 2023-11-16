import unittest
from unittest.mock import patch, Mock
import datetime
from collections import namedtuple
import numpy as np
from AnyQt.QtCore import Qt
from orangecanvas.scheme.signalmanager import LazyValue
from orangewidget.utils.signals import summarize
from Orange.data import Table, Domain, StringVariable, ContinuousVariable, DiscreteVariable, TimeVariable
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.utils.state_summary import format_summary_details, format_multiple_summaries
VarDataPair = namedtuple('VarDataPair', ['variable', 'data'])
continuous_full = VarDataPair(ContinuousVariable('continuous_full'), np.array([0, 1, 2, 3, 4], dtype=float))
continuous_missing = VarDataPair(ContinuousVariable('continuous_missing'), np.array([0, 1, 2, np.nan, 4], dtype=float))
rgb_full = VarDataPair(DiscreteVariable('rgb_full', values=('r', 'g', 'b')), np.array([0, 1, 1, 1, 2], dtype=float))
rgb_missing = VarDataPair(DiscreteVariable('rgb_missing', values=('r', 'g', 'b')), np.array([0, 1, 1, np.nan, 2], dtype=float))
ints_full = VarDataPair(DiscreteVariable('ints_full', values=('2', '3', '4')), np.array([0, 1, 1, 1, 2], dtype=float))
ints_missing = VarDataPair(DiscreteVariable('ints_missing', values=('2', '3', '4')), np.array([0, 1, 1, np.nan, 2], dtype=float))

def _to_timestamps(years):
    if False:
        while True:
            i = 10
    return [datetime.datetime(year, 1, 1).timestamp() if not np.isnan(year) else np.nan for year in years]
time_full = VarDataPair(TimeVariable('time_full'), np.array(_to_timestamps([2000, 2001, 2002, 2003, 2004]), dtype=float))
time_missing = VarDataPair(TimeVariable('time_missing'), np.array(_to_timestamps([2000, np.nan, 2001, 2003, 2004]), dtype=float))
string_full = VarDataPair(StringVariable('string_full'), np.array(['a', 'b', 'c', 'd', 'e'], dtype=object))
string_missing = VarDataPair(StringVariable('string_missing'), np.array(['a', 'b', 'c', StringVariable.Unknown, 'e'], dtype=object))

def make_table(attributes, target=None, metas=None):
    if False:
        i = 10
        return i + 15
    'Build an instance of a table given various variables.\n\n    Parameters\n    ----------\n    attributes : Iterable[Tuple[Variable, np.array]\n    target : Optional[Iterable[Tuple[Variable, np.array]]\n    metas : Optional[Iterable[Tuple[Variable, np.array]]\n\n    Returns\n    -------\n    Table\n\n    '
    (attribute_vars, attribute_vals) = list(zip(*attributes))
    attribute_vals = np.array(attribute_vals).T
    (target_vars, target_vals) = (None, None)
    if target is not None:
        (target_vars, target_vals) = list(zip(*target))
        target_vals = np.array(target_vals).T
    (meta_vars, meta_vals) = (None, None)
    if metas is not None:
        (meta_vars, meta_vals) = list(zip(*metas))
        meta_vals = np.array(meta_vals).T
    return Table.from_numpy(Domain(attribute_vars, class_vars=target_vars, metas=meta_vars), X=attribute_vals, Y=target_vals, metas=meta_vals)

class TestUtils(unittest.TestCase):

    @WidgetTest.skipNonEnglish
    def test_details(self):
        if False:
            print('Hello World!')
        'Check if details part of the summary is formatted correctly'
        data = Table('zoo')
        n_features = len(data.domain.variables) + len(data.domain.metas)
        details = f'zoo: {len(data)} instances, {n_features} variables\nFeatures: {len(data.domain.attributes)} categorical (no missing values)\nTarget: categorical\nMetas: string'
        self.assertEqual(details, format_summary_details(data))
        details = f'Table with {n_features} variables\nFeatures: {len(data.domain.attributes)} categorical\nTarget: categorical\nMetas: string'
        self.assertEqual(details, format_summary_details(data.domain))
        data = Table('housing')
        n_features = len(data.domain.variables) + len(data.domain.metas)
        details = f'housing: {len(data)} instances, {n_features} variables\nFeatures: {len(data.domain.attributes)} numeric (no missing values)\nTarget: numeric'
        self.assertEqual(details, format_summary_details(data))
        data = Table('heart_disease')
        n_features = len(data.domain.variables) + len(data.domain.metas)
        details = f'heart_disease: {len(data)} instances, {n_features} variables\nFeatures: {len(data.domain.attributes)} (7 categorical, 6 numeric) (0.2% missing values)\nTarget: categorical'
        self.assertEqual(details, format_summary_details(data))
        data = make_table([continuous_full, continuous_missing], target=[rgb_full, rgb_missing], metas=[ints_full, ints_missing])
        n_features = len(data.domain.variables) + len(data.domain.metas)
        details = f'Table with {len(data)} instances, {n_features} variables\nFeatures: {len(data.domain.attributes)} numeric (10.0% missing values)\nTarget: {len(data.domain.class_vars)} categorical\nMetas: {len(data.domain.metas)} categorical'
        self.assertEqual(details, format_summary_details(data))
        data = make_table([continuous_full, time_full, ints_full, rgb_missing], target=[rgb_full, continuous_missing], metas=[string_full, string_missing])
        n_features = len(data.domain.variables) + len(data.domain.metas)
        details = f'Table with {len(data)} instances, {n_features} variables\nFeatures: {len(data.domain.attributes)} (2 categorical, 1 numeric, 1 time) (5.0% missing values)\nTarget: {len(data.domain.class_vars)} (1 categorical, 1 numeric)\nMetas: {len(data.domain.metas)} string'
        self.assertEqual(details, format_summary_details(data))
        data = make_table([time_full, time_missing], target=[ints_missing], metas=None)
        details = f'Table with {len(data)} instances, {len(data.domain.variables)} variables\nFeatures: {len(data.domain.attributes)} time (10.0% missing values)\nTarget: categorical'
        self.assertEqual(details, format_summary_details(data))
        data = make_table([rgb_full, ints_full], target=None, metas=None)
        details = f'Table with {len(data)} instances, {len(data.domain.variables)} variables\nFeatures: {len(data.domain.variables)} categorical (no missing values)\nTarget: —'
        self.assertEqual(details, format_summary_details(data))
        data = make_table([rgb_full], target=None, metas=None)
        details = f'Table with {len(data)} instances, {len(data.domain.variables)} variable\nFeatures: categorical (no missing values)\nTarget: —'
        self.assertEqual(details, format_summary_details(data))
        data = Table.from_numpy(domain=None, X=np.random.random((10000, 1000)))
        details = f'Table with {len(data):n} instances, {len(data.domain.variables)} variables\nFeatures: {len(data.domain.variables)} numeric\nTarget: —'
        with patch.object(Table, 'get_nan_frequency_attribute') as mock:
            self.assertEqual(details, format_summary_details(data))
            mock.assert_not_called()
        data = None
        self.assertEqual('', format_summary_details(data))

    @WidgetTest.skipNonEnglish
    def test_multiple_summaries(self):
        if False:
            i = 10
            return i + 15
        data = Table('zoo')
        extra_data = Table('zoo')[20:]
        n_features_data = len(data.domain.variables) + len(data.domain.metas)
        n_features_extra_data = len(extra_data.domain.variables) + len(extra_data.domain.metas)
        details = f'Data:<br>zoo: {len(data)} instances, {n_features_data} variables<br>Features: {len(data.domain.attributes)} categorical (no missing values)<br>Target: categorical<br>Metas: string<hr>Extra Data:<br>zoo: {len(extra_data)} instances, {n_features_extra_data} variables<br>Features: {len(extra_data.domain.attributes)} categorical (no missing values)<br>Target: categorical<br>Metas: string'
        inputs = [('Data', data), ('Extra Data', extra_data)]
        self.assertEqual(details, format_multiple_summaries(inputs))
        details = f'zoo: {len(data)} instances, {n_features_data} variables<br>Features: {len(data.domain.attributes)} categorical (no missing values)<br>Target: categorical<br>Metas: string<hr>zoo: {len(extra_data)} instances, {n_features_extra_data} variables<br>Features: {len(extra_data.domain.attributes)} categorical (no missing values)<br>Target: categorical<br>Metas: string'
        inputs = [('', data), ('', extra_data)]
        self.assertEqual(details, format_multiple_summaries(inputs))
        details = f'No data on output.<hr>Extra data:<br>zoo: {len(extra_data)} instances, {n_features_extra_data} variables<br>Features: {len(extra_data.domain.attributes)} categorical (no missing values)<br>Target: categorical<br>Metas: string<hr>No data on output.'
        outputs = [('', None), ('Extra data', extra_data), ('', None)]
        self.assertEqual(details, format_multiple_summaries(outputs, type_io='output'))

class TestSummarize(unittest.TestCase):

    @patch('Orange.widgets.utils.state_summary._table_previewer')
    def test_summarize_table(self, previewer):
        if False:
            print('Hello World!')
        data = Table('zoo')
        summary = summarize(data)
        self.assertEqual(summary.summary, len(data))
        self.assertEqual(summary.details, format_summary_details(data, format=Qt.RichText))
        previewer.assert_not_called()
        summary.preview_func()
        previewer.assert_called_with(data)

    @patch('Orange.widgets.utils.state_summary._table_previewer')
    def test_summarize_lazy_table(self, previewer):
        if False:
            return 10
        data = Table('zoo')
        lazy_data = LazyValue[Table](lambda : data)
        lazy_data.get_value = Mock(return_value=data)
        summary = summarize(lazy_data)
        self.assertEqual(summary.summary, '?')
        self.assertIsInstance(summary.details, str)
        lazy_data.get_value.assert_not_called()
        previewer.assert_not_called()
        summary.preview_func()
        lazy_data.get_value.assert_called()
        previewer.assert_called_with(data)
        previewer.reset_mock()
        lazy_data = LazyValue[Table](lambda : data, length=123, domain=data.domain)
        lazy_data.get_value = Mock(return_value=data)
        summary = summarize(lazy_data)
        self.assertEqual(summary.summary, 123)
        self.assertEqual(summary.details, format_summary_details(data.domain, format=Qt.RichText))
        lazy_data.get_value.assert_not_called()
        previewer.assert_not_called()
        summary.preview_func()
        lazy_data.get_value.assert_called()
        previewer.assert_called_with(data)
        previewer.reset_mock()
        lazy_data = LazyValue[Table](lambda : data)
        lazy_data.get_value()
        summary = summarize(lazy_data)
        self.assertEqual(summary.summary, len(data))
        self.assertEqual(summary.details, format_summary_details(data, format=Qt.RichText))
        previewer.assert_not_called()
        summary.preview_func()
        previewer.assert_called_with(data)
if __name__ == '__main__':
    unittest.main()