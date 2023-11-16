from typing import Type
from pydantic import BaseModel, Field
from superagi.helper.github_helper import GithubHelper
from superagi.tools.base_tool import BaseTool

class GithubSearchRepoSchema(BaseModel):
    repository_name: str = Field(..., description='Repository name in which we have to search')
    repository_owner: str = Field(..., description='Owner of the github repository')
    file_name: str = Field(..., description='Name of the file we need to fetch from the repository')
    folder_path: str = Field(..., description='folder path in which file is present')

class GithubRepoSearchTool(BaseTool):
    """
    Search File tool

    Attributes:
        name : The name.
        description : The description.
        args_schema : The args schema.
    """
    name = 'GithubRepo Search'
    description = 'Search for a file inside a Github repository'
    args_schema: Type[GithubSearchRepoSchema] = GithubSearchRepoSchema

    def _execute(self, repository_owner: str, repository_name: str, file_name: str, folder_path=None) -> str:
        if False:
            while True:
                i = 10
        '\n        Execute the search file tool.\n\n        Args:\n            repository_owner : The owner of the repository to search file in.\n            repository_name : The name of the repository to search file in.\n            file_name : The name of the file to search.\n            folder_path : The path of the folder to search the file in.\n\n        Returns:\n            The content of the github file.\n        '
        github_access_token = self.get_tool_config('GITHUB_ACCESS_TOKEN')
        github_username = self.get_tool_config('GITHUB_USERNAME')
        github_repo_search = GithubHelper(github_access_token, github_username)
        try:
            content = github_repo_search.get_content_in_file(repository_owner, repository_name, file_name, folder_path)
            return content
        except:
            return 'File not found'