from google.cloud import dialogflow_v2

def sample_create_knowledge_base():
    if False:
        i = 10
        return i + 15
    client = dialogflow_v2.KnowledgeBasesClient()
    knowledge_base = dialogflow_v2.KnowledgeBase()
    knowledge_base.display_name = 'display_name_value'
    request = dialogflow_v2.CreateKnowledgeBaseRequest(parent='parent_value', knowledge_base=knowledge_base)
    response = client.create_knowledge_base(request=request)
    print(response)