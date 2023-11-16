from google.cloud import dialogflow_v2beta1

def sample_list_knowledge_bases():
    if False:
        print('Hello World!')
    client = dialogflow_v2beta1.KnowledgeBasesClient()
    request = dialogflow_v2beta1.ListKnowledgeBasesRequest(parent='parent_value')
    page_result = client.list_knowledge_bases(request=request)
    for response in page_result:
        print(response)