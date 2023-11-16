from azure.ai.language.questionanswering.authoring import AuthoringClient
from azure.core.credentials import AzureKeyCredential
from helpers import QnaAuthoringHelper
from testcase import QuestionAnsweringTestCase

class TestSourcesQnasSynonyms(QuestionAnsweringTestCase):

    def test_add_source(self, recorded_test, qna_creds):
        if False:
            while True:
                i = 10
        client = AuthoringClient(qna_creds['qna_endpoint'], AzureKeyCredential(qna_creds['qna_key']))
        project_name = 'IssacNewton'
        QnaAuthoringHelper.create_test_project(client, project_name=project_name, **self.kwargs_for_polling)
        source_display_name = 'MicrosoftFAQ'
        sources_poller = client.begin_update_sources(project_name=project_name, sources=[{'op': 'add', 'value': {'displayName': source_display_name, 'source': 'https://www.microsoft.com/en-in/software-download/faq', 'sourceUri': 'https://www.microsoft.com/en-in/software-download/faq', 'sourceKind': 'url', 'contentStructureKind': 'unstructured', 'refresh': False}}], **self.kwargs_for_polling)
        sources = sources_poller.result()
        for source in sources:
            assert source['sourceKind']
        sources = client.list_sources(project_name=project_name)
        source_added = False
        for s in sources:
            if 'displayName' in s and s['displayName'] == source_display_name:
                source_added = True
        assert source_added

    def test_add_qna(self, recorded_test, qna_creds):
        if False:
            print('Hello World!')
        client = AuthoringClient(qna_creds['qna_endpoint'], AzureKeyCredential(qna_creds['qna_key']))
        project_name = 'IssacNewton'
        QnaAuthoringHelper.create_test_project(client, project_name=project_name, **self.kwargs_for_polling)
        question = 'What is the easiest way to use azure services in my .NET project?'
        answer = "Using Microsoft's Azure SDKs"
        qna_poller = client.begin_update_qnas(project_name=project_name, qnas=[{'op': 'add', 'value': {'questions': [question], 'answer': answer}}], **self.kwargs_for_polling)
        qnas = qna_poller.result()
        for qna in qnas:
            assert qna['questions']
            assert qna['answer']
        qnas = client.list_qnas(project_name=project_name)
        qna_added = False
        for qna in qnas:
            if ('answer' in qna and 'questions' in qna) and (qna['answer'] == answer and question in qna['questions']):
                qna_added = True
        assert qna_added

    def test_add_synonym(self, recorded_test, qna_creds):
        if False:
            for i in range(10):
                print('nop')
        client = AuthoringClient(qna_creds['qna_endpoint'], AzureKeyCredential(qna_creds['qna_key']))
        project_name = 'IssacNewton'
        QnaAuthoringHelper.create_test_project(client, project_name=project_name, **self.kwargs_for_polling)
        client.update_synonyms(project_name=project_name, synonyms={'value': [{'alterations': ['qnamaker', 'qna maker']}]})
        synonym_added = False
        synonyms = client.list_synonyms(project_name=project_name)
        for s in synonyms:
            if 'alterations' in s and ('qnamaker' in s['alterations'] and 'qna maker' in s['alterations']):
                synonym_added = True
        assert synonym_added