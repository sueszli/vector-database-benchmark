"""Tests for `geemap` package."""
import unittest
import geemap
import ipyleaflet

class TestGeemap(unittest.TestCase):
    """Tests for `geemap` package."""

    def setUp(self):
        if False:
            return 10
        'Set up test fixtures, if any.'

    def tearDown(self):
        if False:
            while True:
                i = 10
        'Tear down test fixtures, if any.'

    def test_map(self):
        if False:
            i = 10
            return i + 15
        m = geemap.Map(ee_initialize=False)
        self.assertIsInstance(m, ipyleaflet.Map)