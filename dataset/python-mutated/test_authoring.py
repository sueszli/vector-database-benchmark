from azure.core.credentials import AzureKeyCredential
from azure.ai.language.conversations.authoring import ConversationAuthoringClient
from devtools_testutils import AzureRecordedTestCase

class TestConversationAuthoring(AzureRecordedTestCase):

    def test_polling_interval(self, conversation_creds):
        if False:
            return 10
        client = ConversationAuthoringClient(conversation_creds['endpoint'], AzureKeyCredential(conversation_creds['key']))
        assert client._config.polling_interval == 5
        client = ConversationAuthoringClient(conversation_creds['endpoint'], AzureKeyCredential(conversation_creds['key']), polling_interval=1)
        assert client._config.polling_interval == 1

    def test_authoring_aad(self, recorded_test, conversation_creds):
        if False:
            while True:
                i = 10
        token = self.get_credential(ConversationAuthoringClient)
        client = ConversationAuthoringClient(conversation_creds['endpoint'], token)
        entities = client.list_supported_prebuilt_entities(language='en')
        for entity in entities:
            assert entity