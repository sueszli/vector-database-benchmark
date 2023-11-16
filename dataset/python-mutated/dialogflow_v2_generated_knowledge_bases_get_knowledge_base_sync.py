from google.cloud import dialogflow_v2

def sample_get_knowledge_base():
    if False:
        print('Hello World!')
    client = dialogflow_v2.KnowledgeBasesClient()
    request = dialogflow_v2.GetKnowledgeBaseRequest(name='name_value')
    response = client.get_knowledge_base(request=request)
    print(response)