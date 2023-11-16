import pytest
from azure.ai.language.questionanswering.authoring import AuthoringClient
from azure.core.credentials import AzureKeyCredential
from helpers import QnaAuthoringHelper
from testcase import QuestionAnsweringTestCase

class TestCreateAndDeploy(QuestionAnsweringTestCase):

    def test_polling_interval(self, qna_creds):
        if False:
            for i in range(10):
                print('nop')
        client = AuthoringClient(qna_creds['qna_endpoint'], AzureKeyCredential(qna_creds['qna_key']))
        assert client._config.polling_interval == 5
        client = AuthoringClient(qna_creds['qna_endpoint'], AzureKeyCredential(qna_creds['qna_key']), polling_interval=1)
        assert client._config.polling_interval == 1

    def test_create_project_aad(self, recorded_test, qna_creds):
        if False:
            i = 10
            return i + 15
        token = self.get_credential(AuthoringClient)
        client = AuthoringClient(qna_creds['qna_endpoint'], token)
        project_name = 'IssacNewton'
        client.create_project(project_name=project_name, options={'description': 'biography of Sir Issac Newton', 'language': 'en', 'multilingualResource': True, 'settings': {'defaultAnswer': 'no answer'}})
        qna_projects = client.list_projects()
        found = False
        for p in qna_projects:
            if 'projectName' in p and p['projectName'] == project_name:
                found = True
        assert found

    def test_create_project(self, recorded_test, qna_creds):
        if False:
            print('Hello World!')
        client = AuthoringClient(qna_creds['qna_endpoint'], AzureKeyCredential(qna_creds['qna_key']))
        project_name = 'IssacNewton'
        client.create_project(project_name=project_name, options={'description': 'biography of Sir Issac Newton', 'language': 'en', 'multilingualResource': True, 'settings': {'defaultAnswer': 'no answer'}})
        qna_projects = client.list_projects()
        found = False
        for p in qna_projects:
            if 'projectName' in p and p['projectName'] == project_name:
                found = True
        assert found

    def test_deploy_project(self, recorded_test, qna_creds):
        if False:
            print('Hello World!')
        client = AuthoringClient(qna_creds['qna_endpoint'], AzureKeyCredential(qna_creds['qna_key']))
        project_name = 'IssacNewton'
        QnaAuthoringHelper.create_test_project(client, project_name=project_name, is_deployable=True, **self.kwargs_for_polling)
        deployment_name = 'production'
        deployment_poller = client.begin_deploy_project(project_name=project_name, deployment_name=deployment_name, **self.kwargs_for_polling)
        project = deployment_poller.result()
        assert project['lastDeployedDateTime']
        assert project['deploymentName'] == 'production'
        deployments = client.list_deployments(project_name=project_name)
        deployment_found = False
        for d in deployments:
            if 'deploymentName' in d and d['deploymentName'] == deployment_name:
                deployment_found = True
        assert deployment_found