__license__ = 'GNU Affero General Public License http://www.gnu.org/licenses/agpl.html'
__copyright__ = 'Copyright (C) 2021 The OctoPrint Project - Released under terms of the AGPLv3 License'
import unittest
import octoprint.util

class FastDeepcopyTest(unittest.TestCase):

    def test_clean(self):
        if False:
            i = 10
            return i + 15
        data = {'a': 1, 'b': 2, 'c': 3}
        self.assertEqual(data, octoprint.util.fast_deepcopy(data))

    def test_function(self):
        if False:
            for i in range(10):
                print('nop')
        data = {'a': 1, 'b': 2, 'c': 3, 'f': lambda x: x + 1}
        self.assertEqual(data, octoprint.util.fast_deepcopy(data))