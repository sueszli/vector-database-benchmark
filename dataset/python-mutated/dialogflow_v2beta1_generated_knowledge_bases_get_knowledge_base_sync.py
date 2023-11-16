from google.cloud import dialogflow_v2beta1

def sample_get_knowledge_base():
    if False:
        return 10
    client = dialogflow_v2beta1.KnowledgeBasesClient()
    request = dialogflow_v2beta1.GetKnowledgeBaseRequest(name='name_value')
    response = client.get_knowledge_base(request=request)
    print(response)