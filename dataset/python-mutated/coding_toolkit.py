from abc import ABC
from typing import List
from superagi.tools.base_tool import BaseToolkit, BaseTool, ToolConfiguration
from superagi.tools.code.improve_code import ImproveCodeTool
from superagi.tools.code.write_code import CodingTool
from superagi.tools.code.write_spec import WriteSpecTool
from superagi.tools.code.write_test import WriteTestTool

class CodingToolkit(BaseToolkit, ABC):
    name: str = 'CodingToolkit'
    description: str = 'Coding Tool kit contains all tools related to coding tasks'

    def get_tools(self) -> List[BaseTool]:
        if False:
            for i in range(10):
                print('nop')
        return [CodingTool(), WriteSpecTool(), WriteTestTool(), ImproveCodeTool()]

    def get_env_keys(self) -> List[ToolConfiguration]:
        if False:
            i = 10
            return i + 15
        return []