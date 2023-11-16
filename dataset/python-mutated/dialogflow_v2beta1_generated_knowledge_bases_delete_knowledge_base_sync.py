from google.cloud import dialogflow_v2beta1

def sample_delete_knowledge_base():
    if False:
        print('Hello World!')
    client = dialogflow_v2beta1.KnowledgeBasesClient()
    request = dialogflow_v2beta1.DeleteKnowledgeBaseRequest(name='name_value')
    client.delete_knowledge_base(request=request)