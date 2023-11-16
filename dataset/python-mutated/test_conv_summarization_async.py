import pytest
from azure.ai.language.conversations.aio import ConversationAnalysisClient
from azure.core.credentials import AzureKeyCredential
from devtools_testutils import AzureRecordedTestCase

class TestConversationalSummarizationAsync(AzureRecordedTestCase):

    def test_polling_interval(self, conversation_creds):
        if False:
            print('Hello World!')
        client = ConversationAnalysisClient(conversation_creds['endpoint'], AzureKeyCredential(conversation_creds['key']))
        assert client._config.polling_interval == 5
        client = ConversationAnalysisClient(conversation_creds['endpoint'], AzureKeyCredential(conversation_creds['key']), polling_interval=1)
        assert client._config.polling_interval == 1

    @pytest.mark.asyncio
    async def test_conversational_summarization(self, recorded_test, conversation_creds):
        client = ConversationAnalysisClient(conversation_creds['endpoint'], AzureKeyCredential(conversation_creds['key']))
        async with client:
            poller = await client.begin_conversation_analysis(task={'displayName': 'Analyze conversations from xxx', 'analysisInput': {'conversations': [{'conversationItems': [{'text': 'Hello, how can I help you?', 'modality': 'text', 'id': '1', 'role': 'Agent', 'participantId': 'Agent'}, {'text': 'How to upgrade Office? I am getting error messages the whole day.', 'modality': 'text', 'id': '2', 'role': 'Customer', 'participantId': 'Customer'}, {'text': 'Press the upgrade button please. Then sign in and follow the instructions.', 'modality': 'text', 'id': '3', 'role': 'Agent', 'participantId': 'Agent'}], 'modality': 'text', 'id': 'conversation1', 'language': 'en'}]}, 'tasks': [{'taskName': 'Issue task', 'kind': 'ConversationalSummarizationTask', 'parameters': {'summaryAspects': ['issue']}}, {'taskName': 'Resolution task', 'kind': 'ConversationalSummarizationTask', 'parameters': {'summaryAspects': ['resolution']}}]})
            result = await poller.result()
            assert not result is None
            assert result['status'] == 'succeeded'
            task_result = result['tasks']['items'][0]
            assert task_result['status'] == 'succeeded'
            assert task_result['kind'] == 'conversationalSummarizationResults'
            conversation_result = task_result['results']['conversations'][0]
            summaries = conversation_result['summaries']
            assert summaries
            for summary in summaries:
                assert summary['aspect'] in ['issue', 'resolution']
                assert summary['text']

    @pytest.mark.asyncio
    async def test_conv_summ_chapter_narrative(self, recorded_test, conversation_creds):
        client = ConversationAnalysisClient(conversation_creds['endpoint'], AzureKeyCredential(conversation_creds['key']))
        async with client:
            poller = await client.begin_conversation_analysis(task={'displayName': 'Conversation Summarization Example', 'analysisInput': {'conversations': [{'id': '1', 'language': 'en', 'modality': 'transcript', 'conversationItems': [{'participantId': 'speaker 1', 'id': '1', 'text': "Let's get started.", 'lexical': '', 'itn': '', 'maskedItn': '', 'conversationItemLevelTiming': {'offset': 0, 'duration': 20000000}}, {'participantId': 'speaker 2', 'id': '2', 'text': 'OK. How many remaining bugs do we have now?', 'lexical': '', 'itn': '', 'maskedItn': '', 'conversationItemLevelTiming': {'offset': 20000000, 'duration': 50000000}}, {'participantId': 'speaker 3', 'id': '3', 'text': 'Only 3.', 'lexical': '', 'itn': '', 'maskedItn': '', 'conversationItemLevelTiming': {'offset': 50000000, 'duration': 60000000}}]}]}, 'tasks': [{'taskName': 'Conversation Summarization Task 1', 'kind': 'ConversationalSummarizationTask', 'parameters': {'summaryAspects': ['chapterTitle', 'narrative']}}]})
            result = await poller.result()
            assert result is not None
            assert result['status'] == 'succeeded'
            task_result = result['tasks']['items'][0]
            assert task_result['status'] == 'succeeded'
            assert task_result['kind'] == 'conversationalSummarizationResults'
            conversation_result = task_result['results']['conversations'][0]
            summaries = conversation_result['summaries']
            assert summaries
            for summary in summaries:
                assert summary['aspect'] in ['chapterTitle', 'narrative']
                assert summary['text']
                assert summary['contexts']