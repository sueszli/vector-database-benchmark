from azure.ai.language.conversations import ConversationAnalysisClient
from azure.core.credentials import AzureKeyCredential
from devtools_testutils import AzureRecordedTestCase

class TestOrchestrationAppQnaResponse(AzureRecordedTestCase):

    def test_orchestration_app_qna_response(self, recorded_test, conversation_creds):
        if False:
            i = 10
            return i + 15
        client = ConversationAnalysisClient(conversation_creds['endpoint'], AzureKeyCredential(conversation_creds['key']))
        with client:
            query = 'How are you?'
            result = client.analyze_conversation(task={'kind': 'Conversation', 'analysisInput': {'conversationItem': {'participantId': '1', 'id': '1', 'modality': 'text', 'language': 'en', 'text': query}, 'isLoggingEnabled': False}, 'parameters': {'projectName': conversation_creds['orch_project_name'], 'deploymentName': conversation_creds['orch_deployment_name'], 'verbose': True}})
            top_project = 'ChitChat-QnA'
            assert not result is None
            assert result['kind'] == 'ConversationResult'
            assert result['result']['query'] == query
            assert result['result']['prediction']['projectKind'] == 'Orchestration'
            assert result['result']['prediction']['topIntent'] == top_project
            top_intent_object = result['result']['prediction']['intents'][top_project]
            assert top_intent_object['targetProjectKind'] == 'QuestionAnswering'
            qna_result = top_intent_object['result']
            answer = qna_result['answers'][0]['answer']
            assert not answer is None
            assert qna_result['answers'][0]['confidenceScore'] >= 0