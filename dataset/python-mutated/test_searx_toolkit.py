import unittest
from superagi.tools.searx.searx import SearxSearchTool
from superagi.tools.searx.searx_toolkit import SearxSearchToolkit

class TestSearxSearchToolkit(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        '\n        Set up the test fixture.\n\n        This method is called before each test method is executed to prepare the test environment.\n\n        Returns:\n            None\n        '
        self.toolkit = SearxSearchToolkit()

    def test_get_tools(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test the `get_tools` method of the `SearxSearchToolkit` class.\n\n        It should return a list of tools, containing one instance of `SearxSearchTool`.\n\n        Returns:\n            None\n        '
        tools = self.toolkit.get_tools()
        self.assertEqual(1, len(tools))
        self.assertIsInstance(tools[0], SearxSearchTool)

    def test_get_env_keys(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test the `get_env_keys` method of the `SearxSearchToolkit` class.\n\n        It should return an empty list of environment keys.\n\n        Returns:\n            None\n        '
        env_keys = self.toolkit.get_env_keys()
        self.assertEqual(0, len(env_keys))