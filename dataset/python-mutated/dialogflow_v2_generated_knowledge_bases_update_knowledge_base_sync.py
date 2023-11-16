from google.cloud import dialogflow_v2

def sample_update_knowledge_base():
    if False:
        print('Hello World!')
    client = dialogflow_v2.KnowledgeBasesClient()
    knowledge_base = dialogflow_v2.KnowledgeBase()
    knowledge_base.display_name = 'display_name_value'
    request = dialogflow_v2.UpdateKnowledgeBaseRequest(knowledge_base=knowledge_base)
    response = client.update_knowledge_base(request=request)
    print(response)