from chalicelib.core.integration_base_issue import BaseIntegrationIssue
from chalicelib.utils import github_client_v3
from chalicelib.utils.github_client_v3 import github_formatters as formatter

class GithubIntegrationIssue(BaseIntegrationIssue):

    def __init__(self, integration_token):
        if False:
            print('Hello World!')
        self.__client = github_client_v3.githubV3Request(integration_token)
        super(GithubIntegrationIssue, self).__init__('GITHUB', integration_token)

    def get_current_user(self):
        if False:
            i = 10
            return i + 15
        return formatter.user(self.__client.get('/user'))

    def get_meta(self, repoId):
        if False:
            i = 10
            return i + 15
        current_user = self.get_current_user()
        try:
            users = self.__client.get(f'/repositories/{repoId}/collaborators')
        except Exception as e:
            users = []
        users = [formatter.user(u) for u in users]
        if current_user not in users:
            users.insert(0, current_user)
        meta = {'users': users, 'issueTypes': [formatter.label(l) for l in self.__client.get(f'/repositories/{repoId}/labels')]}
        return meta

    def create_new_assignment(self, integration_project_id, title, description, assignee, issue_type):
        if False:
            while True:
                i = 10
        repoId = integration_project_id
        assignees = [assignee]
        labels = [str(issue_type)]
        metas = self.get_meta(repoId)
        real_assignees = []
        for a in assignees:
            for u in metas['users']:
                if a == str(u['id']):
                    real_assignees.append(u['name'])
                    break
        real_labels = ['OpenReplay']
        for l in labels:
            found = False
            for ll in metas['issueTypes']:
                if l == str(ll['id']):
                    real_labels.append(ll['name'])
                    found = True
                    break
            if not found:
                real_labels.append(l)
        issue = self.__client.post(f'/repositories/{repoId}/issues', body={'title': title, 'body': description, 'assignees': real_assignees, 'labels': real_labels})
        return formatter.issue(issue)

    def get_by_ids(self, saved_issues):
        if False:
            while True:
                i = 10
        results = []
        for i in saved_issues:
            results.append(self.get(integration_project_id=i['integrationProjectId'], assignment_id=i['id']))
        return {'issues': results}

    def get(self, integration_project_id, assignment_id):
        if False:
            return 10
        repoId = integration_project_id
        issueNumber = assignment_id
        issue = self.__client.get(f'/repositories/{repoId}/issues/{issueNumber}')
        issue = formatter.issue(issue)
        if issue['commentsCount'] > 0:
            issue['comments'] = [formatter.comment(c) for c in self.__client.get(f'/repositories/{repoId}/issues/{issueNumber}/comments')]
        return issue

    def comment(self, integration_project_id, assignment_id, comment):
        if False:
            for i in range(10):
                print('nop')
        repoId = integration_project_id
        issueNumber = assignment_id
        commentCreated = self.__client.post(f'/repositories/{repoId}/issues/{issueNumber}/comments', body={'body': comment})
        return formatter.comment(commentCreated)

    def get_metas(self, integration_project_id):
        if False:
            i = 10
            return i + 15
        current_user = self.get_current_user()
        try:
            users = self.__client.get(f'/repositories/{integration_project_id}/collaborators')
        except Exception as e:
            users = []
        users = [formatter.user(u) for u in users]
        if current_user not in users:
            users.insert(0, current_user)
        return {'provider': self.provider.lower(), 'users': users, 'issueTypes': [formatter.label(l) for l in self.__client.get(f'/repositories/{integration_project_id}/labels')]}

    def get_projects(self):
        if False:
            print('Hello World!')
        repos = self.__client.get('/user/repos')
        return [formatter.repo(r) for r in repos]