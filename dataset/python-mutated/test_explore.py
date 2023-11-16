from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import os
import signal
import time
import unittest
import uuid
import six
import turicreate as tc
from turicreate.toolkits._internal_utils import _mac_ver

class ExploreTest(unittest.TestCase):

    @unittest.skipIf(_mac_ver() < (10, 12), "macOS-only test; UISoup doesn't work on Linux")
    @unittest.skipIf(_mac_ver() > (10, 13), 'macOS 10.14 appears to have broken the UX flow to prompt for accessibility access')
    @unittest.skipIf(not six.PY2, "Python 2.7-only test; UISoup doesn't work on 3.x")
    def test_sanity_on_macOS(self):
        if False:
            while True:
                i = 10
        '\n        Create a simple SFrame, containing a very unique string.\n        Then, using uisoup, look for this string within a window\n        and assert that it appears.\n        '
        from uisoup import uisoup
        unique_str = repr(uuid.uuid4())
        sf = tc.SFrame({'a': [1, 2, 3], 'b': ['hello', 'world', unique_str]})
        sf.explore()
        time.sleep(2)
        window = None
        try:
            window = uisoup.get_window('Turi*Create*Visualization')
            result = window.findall(value=unique_str)
            self.assertEqual(len(result), 1, 'Expected to find exactly one element containing the uniquestring %s.' % unique_str)
            first = result[0]
            self.assertEqual(first.acc_name, unique_str, 'Expected to find the unique string %s as the name of the foundelement. Instead, got %s.' % (unique_str, first.acc_name))
        finally:
            if window is not None:
                os.kill(window.proc_id, signal.SIGTERM)