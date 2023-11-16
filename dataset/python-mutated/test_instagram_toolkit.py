import pytest
from superagi.tools.instagram_tool.instagram import InstagramTool
from superagi.tools.instagram_tool.instagram_toolkit import InstagramToolkit

class TestInstagramToolKit:

    def setup_method(self):
        if False:
            print('Hello World!')
        '\n        Set up the test fixture.\n\n        This method is called before each test method is executed to prepare the test environment.\n\n        Returns:\n            None\n        '
        self.toolkit = InstagramToolkit()

    def test_get_tools(self):
        if False:
            while True:
                i = 10
        '\n        Test the `get_tools` method of the `DuckDuckGoToolkit` class.\n\n        It should return a list of tools, containing one instance of `DuckDuckGoSearchTool`.\n\n        Returns:\n            None\n        '
        tools = self.toolkit.get_tools()
        assert len(tools) == 1
        assert isinstance(tools[0], InstagramTool)

    def test_get_env_keys(self):
        if False:
            while True:
                i = 10
        '\n        Test the `get_env_keys` method of the `DuckDuckGoToolkit` class.\n\n        It should return an empty list of environment keys.\n\n        Returns:\n            None\n        '
        env_keys = self.toolkit.get_env_keys()
        assert len(env_keys) == 2