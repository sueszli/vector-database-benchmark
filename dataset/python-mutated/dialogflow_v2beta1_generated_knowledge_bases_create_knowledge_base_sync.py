from google.cloud import dialogflow_v2beta1

def sample_create_knowledge_base():
    if False:
        while True:
            i = 10
    client = dialogflow_v2beta1.KnowledgeBasesClient()
    knowledge_base = dialogflow_v2beta1.KnowledgeBase()
    knowledge_base.display_name = 'display_name_value'
    request = dialogflow_v2beta1.CreateKnowledgeBaseRequest(parent='parent_value', knowledge_base=knowledge_base)
    response = client.create_knowledge_base(request=request)
    print(response)