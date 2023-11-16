from abc import ABC
from typing import List
from superagi.tools.base_tool import BaseTool, BaseToolkit, ToolConfiguration
from superagi.tools.jira.create_issue import CreateIssueTool
from superagi.tools.jira.edit_issue import EditIssueTool
from superagi.tools.jira.get_projects import GetProjectsTool
from superagi.tools.jira.search_issues import SearchJiraTool
from superagi.types.key_type import ToolConfigKeyType
from superagi.models.tool_config import ToolConfig

class JiraToolkit(BaseToolkit, ABC):
    name: str = 'Jira Toolkit'
    description: str = 'Toolkit containing tools for Jira integration'

    def get_tools(self) -> List[BaseTool]:
        if False:
            for i in range(10):
                print('nop')
        return [CreateIssueTool(), EditIssueTool(), GetProjectsTool(), SearchJiraTool()]

    def get_env_keys(self) -> List[ToolConfiguration]:
        if False:
            for i in range(10):
                print('nop')
        return [ToolConfiguration(key='JIRA_INSTANCE_URL', key_type=ToolConfigKeyType.STRING, is_required=True, is_secret=False), ToolConfiguration(key='JIRA_USERNAME', key_type=ToolConfigKeyType.STRING, is_required=True, is_secret=False), ToolConfiguration(key='JIRA_API_TOKEN', key_type=ToolConfigKeyType.STRING, is_required=True, is_secret=True)]