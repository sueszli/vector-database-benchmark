from tests.QtTestCase import QtTestCase
from tests.utils_testing import get_path_for_data_file
from urh.controller.dialogs.SignalDetailsDialog import SignalDetailsDialog
from urh.signalprocessing.Signal import Signal
from urh.util.Formatter import Formatter

class TestSignalDetailsGUI(QtTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.signal = Signal(get_path_for_data_file('esaver.complex16s'), 'test')
        self.signal.sample_rate = 2000000.0
        self.dialog = SignalDetailsDialog(self.signal)
        if self.SHOW:
            self.dialog.show()

    def test_set_sample_rate(self):
        if False:
            return 10
        self.assertEqual(Formatter.science_time(self.signal.num_samples / self.signal.sample_rate), self.dialog.ui.lDuration.text())
        self.dialog.ui.dsb_sample_rate.setValue(5000000.0)
        self.assertEqual(self.signal.sample_rate, 5000000.0)
        self.assertEqual(Formatter.science_time(self.signal.num_samples / self.signal.sample_rate), self.dialog.ui.lDuration.text())