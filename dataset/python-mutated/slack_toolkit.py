from abc import ABC
from typing import List
from superagi.tools.base_tool import BaseTool, BaseToolkit, ToolConfiguration
from superagi.tools.slack.send_message import SlackMessageTool
from superagi.types.key_type import ToolConfigKeyType

class SlackToolkit(BaseToolkit, ABC):
    name: str = 'Slack Toolkit'
    description: str = 'Toolkit containing tools for Slack integration'

    def get_tools(self) -> List[BaseTool]:
        if False:
            print('Hello World!')
        return [SlackMessageTool()]

    def get_env_keys(self) -> List[ToolConfiguration]:
        if False:
            while True:
                i = 10
        return [ToolConfiguration(key='SLACK_BOT_TOKEN', key_type=ToolConfigKeyType.STRING, is_required=True, is_secret=True)]