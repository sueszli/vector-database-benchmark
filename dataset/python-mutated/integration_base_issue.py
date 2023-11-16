from abc import ABC, abstractmethod

class RequestException(Exception):
    pass

def proxy_issues_handler(e):
    if False:
        return 10
    print('=======__proxy_issues_handler=======')
    print(str(e))
    return {'errors': [str(e)]}

class BaseIntegrationIssue(ABC):

    def __init__(self, provider, integration_token):
        if False:
            i = 10
            return i + 15
        self.provider = provider
        self.integration_token = integration_token

    @abstractmethod
    def create_new_assignment(self, integration_project_id, title, description, assignee, issue_type):
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def get_by_ids(self, saved_issues):
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def get(self, integration_project_id, assignment_id):
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def comment(self, integration_project_id, assignment_id, comment):
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def get_metas(self, integration_project_id):
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def get_projects(self):
        if False:
            print('Hello World!')
        pass