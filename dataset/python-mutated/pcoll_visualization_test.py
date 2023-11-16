"""Tests for apache_beam.runners.interactive.display.pcoll_visualization."""
import unittest
from unittest.mock import ANY
from unittest.mock import PropertyMock
from unittest.mock import patch
import pytz
import apache_beam as beam
from apache_beam.runners import runner
from apache_beam.runners.interactive import interactive_beam as ib
from apache_beam.runners.interactive import interactive_environment as ie
from apache_beam.runners.interactive import interactive_runner as ir
from apache_beam.runners.interactive.display import pcoll_visualization as pv
from apache_beam.runners.interactive.recording_manager import RecordingManager
from apache_beam.runners.interactive.testing.mock_ipython import mock_get_ipython
from apache_beam.transforms.window import GlobalWindow
from apache_beam.transforms.window import IntervalWindow
from apache_beam.utils.windowed_value import PaneInfo
from apache_beam.utils.windowed_value import PaneInfoTiming
try:
    import timeloop
except ImportError:
    pass

@unittest.skipIf(not ie.current_env().is_interactive_ready, '[interactive] dependency is not installed.')
class PCollectionVisualizationTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        ie.new_env()
        pv._pcoll_visualization_ready = True
        ie.current_env()._is_in_notebook = True
        ib.options.display_timezone = pytz.timezone('US/Pacific')
        self._p = beam.Pipeline(ir.InteractiveRunner())
        self._pcoll = self._p | 'Create' >> beam.Create(range(5))
        ib.watch(self)
        ie.current_env().track_user_pipelines()
        recording_manager = RecordingManager(self._p)
        recording = recording_manager.record([self._pcoll], 5, 5)
        self._stream = recording.stream(self._pcoll)

    def test_pcoll_visualization_generate_unique_display_id(self):
        if False:
            i = 10
            return i + 15
        pv_1 = pv.PCollectionVisualization(self._stream)
        pv_2 = pv.PCollectionVisualization(self._stream)
        self.assertNotEqual(pv_1._dive_display_id, pv_2._dive_display_id)
        self.assertNotEqual(pv_1._overview_display_id, pv_2._overview_display_id)
        self.assertNotEqual(pv_1._df_display_id, pv_2._df_display_id)

    @patch('IPython.get_ipython', new_callable=mock_get_ipython)
    @patch('apache_beam.runners.interactive.interactive_environment.InteractiveEnvironment.is_in_notebook', new_callable=PropertyMock)
    def test_one_shot_visualization_not_return_handle(self, mocked_is_in_notebook, unused):
        if False:
            while True:
                i = 10
        mocked_is_in_notebook.return_value = True
        self.assertIsNone(pv.visualize(self._stream, display_facets=True))

    @patch('IPython.get_ipython', new_callable=mock_get_ipython)
    @patch('apache_beam.runners.interactive.interactive_environment.InteractiveEnvironment.is_in_notebook', new_callable=PropertyMock)
    def test_dynamic_plotting_return_handle(self, mocked_is_in_notebook, unused):
        if False:
            i = 10
            return i + 15
        mocked_is_in_notebook.return_value = True
        h = pv.visualize(self._stream, dynamic_plotting_interval=1, display_facets=True)
        self.assertIsInstance(h, timeloop.Timeloop)
        h.stop()

    @patch('IPython.get_ipython', new_callable=mock_get_ipython)
    @patch('apache_beam.runners.interactive.interactive_environment.InteractiveEnvironment.is_in_notebook', new_callable=PropertyMock)
    def test_no_dynamic_plotting_when_not_in_notebook(self, mocked_is_in_notebook, unused):
        if False:
            print('Hello World!')
        mocked_is_in_notebook.return_value = False
        h = pv.visualize(self._stream, dynamic_plotting_interval=1, display_facets=True)
        self.assertIsNone(h)

    @patch('apache_beam.runners.interactive.display.pcoll_visualization.PCollectionVisualization._display_dive')
    @patch('apache_beam.runners.interactive.display.pcoll_visualization.PCollectionVisualization._display_overview')
    @patch('apache_beam.runners.interactive.display.pcoll_visualization.PCollectionVisualization._display_dataframe')
    def test_dynamic_plotting_updates_same_display(self, mocked_display_dataframe, mocked_display_overview, mocked_display_dive):
        if False:
            return 10
        original_pcollection_visualization = pv.PCollectionVisualization(self._stream, display_facets=True)
        new_pcollection_visualization = pv.PCollectionVisualization(self._stream, display_facets=True)
        new_pcollection_visualization.display(updating_pv=original_pcollection_visualization)
        mocked_display_dataframe.assert_called_once_with(ANY, original_pcollection_visualization)
        mocked_display_overview.assert_called_once_with(ANY, original_pcollection_visualization)
        mocked_display_dive.assert_called_once_with(ANY, original_pcollection_visualization)

    def test_auto_stop_dynamic_plotting_when_job_is_terminated(self):
        if False:
            while True:
                i = 10
        fake_pipeline_result = runner.PipelineResult(runner.PipelineState.RUNNING)
        ie.current_env().set_pipeline_result(self._p, fake_pipeline_result)
        self.assertFalse(ie.current_env().is_terminated(self._p))
        fake_pipeline_result = runner.PipelineResult(runner.PipelineState.DONE)
        ie.current_env().set_pipeline_result(self._p, fake_pipeline_result)
        self.assertTrue(ie.current_env().is_terminated(self._p))

    @patch('pandas.DataFrame.head')
    def test_display_plain_text_when_kernel_has_no_frontend(self, _mocked_head):
        if False:
            return 10
        ie.current_env()._is_in_notebook = False
        self.assertIsNone(pv.visualize(self._stream, display_facets=True))
        _mocked_head.assert_called_once()

    def test_event_time_formatter(self):
        if False:
            for i in range(10):
                print('nop')
        event_time_us = 1583190894000000
        self.assertEqual('2020-03-02 15:14:54.000000-0800', pv.event_time_formatter(event_time_us))

    def test_event_time_formatter_overflow_lower_bound(self):
        if False:
            while True:
                i = 10
        event_time_us = -100000000000000000
        self.assertEqual('Min Timestamp', pv.event_time_formatter(event_time_us))

    def test_event_time_formatter_overflow_upper_bound(self):
        if False:
            while True:
                i = 10
        event_time_us = 253402300800000000
        self.assertEqual('Max Timestamp', pv.event_time_formatter(event_time_us))

    def test_windows_formatter_global(self):
        if False:
            while True:
                i = 10
        gw = GlobalWindow()
        self.assertEqual(str(gw), pv.windows_formatter([gw]))

    def test_windows_formatter_interval(self):
        if False:
            for i in range(10):
                print('nop')
        iw = IntervalWindow(start=1583190894, end=1583200000)
        self.assertEqual('2020-03-02 15:14:54.000000-0800 (2h 31m 46s)', pv.windows_formatter([iw]))

    def test_pane_info_formatter(self):
        if False:
            while True:
                i = 10
        self.assertEqual('Pane 0: Final Early', pv.pane_info_formatter(PaneInfo(is_first=False, is_last=True, timing=PaneInfoTiming.EARLY, index=0, nonspeculative_index=0)))
if __name__ == '__main__':
    unittest.main()