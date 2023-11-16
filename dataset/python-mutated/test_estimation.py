__author__ = 'Gina Häußge <osd@foosel.net>'
__license__ = 'GNU Affero General Public License http://www.gnu.org/licenses/agpl.html'
__copyright__ = 'Copyright (C) 2014 The OctoPrint Project - Released under terms of the AGPLv3 License'
import unittest
from ddt import data, ddt, unpack
from octoprint.printer.estimation import TimeEstimationHelper

@ddt
class EstimationTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.estimation_helper = type(TimeEstimationHelper)(TimeEstimationHelper.__name__, (TimeEstimationHelper,), {'STABLE_THRESHOLD': 0.1, 'STABLE_ROLLING_WINDOW': 3, 'STABLE_COUNTDOWN': 1})()

    @data(((1.0, 2.0, 3.0, 4.0, 5.0), 3.0), ((1.0, 2.0, 0.0, 1.0, 2.0), 1.2), ((1.0, -2.0, -1.0, -2.0, 3.0), -0.2))
    @unpack
    def test_average_total(self, estimates, expected):
        if False:
            print('Hello World!')
        for estimate in estimates:
            self.estimation_helper.update(estimate)
        self.assertEqual(self.estimation_helper.average_total, expected)

    @data(((1.0, 2.0), -1), ((1.0, 2.0, 3.0), -1), ((1.0, 2.0, 3.0, 4.0), 0.5), ((1.0, 2.0, 3.0, 4.0, 5.0), 0.5), ((1.0, 2.0, 0.0, 1.0, 2.0), 0.7 / 3))
    @unpack
    def test_average_distance(self, estimates, expected):
        if False:
            while True:
                i = 10
        for estimate in estimates:
            self.estimation_helper.update(estimate)
        self.assertEqual(self.estimation_helper.average_distance, expected)

    @data(((1.0, 1.0), -1), ((1.0, 1.0, 1.0), 1.0), ((1.0, 2.0, 3.0, 4.0, 5.0), 4.0))
    @unpack
    def test_average_total_rolling(self, estimates, expected):
        if False:
            return 10
        for estimate in estimates:
            self.estimation_helper.update(estimate)
        self.assertEqual(self.estimation_helper.average_total_rolling, expected)

    @data(((1.0, 1.0, 1.0, 1.0), False), ((1.0, 1.0, 1.0, 1.0, 1.0), True), ((1.0, 2.0, 3.0, 4.0, 5.0), False), ((0.0, 0.09, 0.18, 0.27, 0.36), True))
    @unpack
    def test_is_stable(self, estimates, expected):
        if False:
            i = 10
            return i + 15
        for estimate in estimates:
            self.estimation_helper.update(estimate)
        self.assertEqual(self.estimation_helper.is_stable(), expected)