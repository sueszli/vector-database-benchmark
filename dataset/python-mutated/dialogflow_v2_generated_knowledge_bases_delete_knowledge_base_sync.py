from google.cloud import dialogflow_v2

def sample_delete_knowledge_base():
    if False:
        return 10
    client = dialogflow_v2.KnowledgeBasesClient()
    request = dialogflow_v2.DeleteKnowledgeBaseRequest(name='name_value')
    client.delete_knowledge_base(request=request)