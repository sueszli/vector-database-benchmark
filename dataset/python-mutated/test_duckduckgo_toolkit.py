import pytest
from superagi.tools.duck_duck_go.duck_duck_go_search_toolkit import DuckDuckGoToolkit
from superagi.tools.duck_duck_go.duck_duck_go_search import DuckDuckGoSearchTool

class TestDuckDuckGoSearchToolKit:

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set up the test fixture.\n\n        This method is called before each test method is executed to prepare the test environment.\n\n        Returns:\n            None\n        '
        self.toolkit = DuckDuckGoToolkit()

    def test_get_tools(self):
        if False:
            i = 10
            return i + 15
        '\n        Test the `get_tools` method of the `DuckDuckGoToolkit` class.\n\n        It should return a list of tools, containing one instance of `DuckDuckGoSearchTool`.\n\n        Returns:\n            None\n        '
        tools = self.toolkit.get_tools()
        assert len(tools) == 1
        assert isinstance(tools[0], DuckDuckGoSearchTool)

    def test_get_env_keys(self):
        if False:
            while True:
                i = 10
        '\n        Test the `get_env_keys` method of the `DuckDuckGoToolkit` class.\n\n        It should return an empty list of environment keys.\n\n        Returns:\n            None\n        '
        env_keys = self.toolkit.get_env_keys()
        assert len(env_keys) == 0