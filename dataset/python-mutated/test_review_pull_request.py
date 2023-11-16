import pytest
from unittest.mock import patch, Mock
import pytest_mock
from pydantic import ValidationError
from superagi.tools.github.review_pull_request import GithubReviewPullRequest

class MockLLM:

    def get_model(self):
        if False:
            for i in range(10):
                print('nop')
        return 'some_model'

class MockTokenCounter:

    @staticmethod
    def count_message_tokens(message, model):
        if False:
            while True:
                i = 10
        return len(message[0]['content'])

def test_split_pull_request_content_into_multiple_parts():
    if False:
        print('Hello World!')
    tool = GithubReviewPullRequest()
    tool.llm = MockLLM()
    pull_request_arr = ['part1', 'part2', 'part3']
    result = tool.split_pull_request_content_into_multiple_parts(4000, pull_request_arr)
    expected = ['diff --gitpart1diff --gitpart2diff --gitpart3']
    assert result == expected

@pytest.mark.parametrize('diff_content, file_path, line_number, expected', [('file_path_1\n@@ -1,3 +1,4 @@\n+ line1\n+ line2\n+ line3', 'file_path_1', 3, 4), ('file_path_2\n@@ -1,3 +1,3 @@\n+ line1\n- line2', 'file_path_2', 1, 2), ('file_path_3\n@@ -1,3 +1,4 @@\n+ line1\n+ line2\n- line3', 'file_path_3', 2, 3)])
def test_get_exact_line_number(diff_content, file_path, line_number, expected):
    if False:
        return 10
    tool = GithubReviewPullRequest()
    result = tool.get_exact_line_number(diff_content, file_path, line_number)
    assert result == expected

class MockGithubHelper:

    def __init__(self, access_token, username):
        if False:
            print('Hello World!')
        pass

    def get_pull_request_content(self, owner, repo, pr_number):
        if False:
            while True:
                i = 10
        return 'mock_content'

    def get_latest_commit_id_of_pull_request(self, owner, repo, pr_number):
        if False:
            while True:
                i = 10
        return 'mock_commit_id'

    def add_line_comment_to_pull_request(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return True

def test_execute():
    if False:
        while True:
            i = 10
    with patch('superagi.tools.github.review_pull_request.GithubHelper', MockGithubHelper), patch('superagi.tools.github.review_pull_request.TokenCounter.count_message_tokens', return_value=3000), patch('superagi.tools.github.review_pull_request.Agent.find_org_by_agent_id', return_value=Mock()), patch.object(GithubReviewPullRequest, 'get_tool_config', return_value='mock_value'), patch.object(GithubReviewPullRequest, 'run_code_review', return_value=None):
        tool = GithubReviewPullRequest()
        tool.llm = Mock()
        tool.llm.get_model = Mock(return_value='mock_model')
        tool.toolkit_config = Mock()
        tool.toolkit_config.session = 'mock_session'
        result = tool._execute('mock_repo', 'mock_owner', 42)
        assert result == 'Added comments to the pull request:42'