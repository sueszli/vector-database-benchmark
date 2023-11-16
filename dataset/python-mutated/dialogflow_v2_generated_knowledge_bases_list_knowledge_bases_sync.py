from google.cloud import dialogflow_v2

def sample_list_knowledge_bases():
    if False:
        return 10
    client = dialogflow_v2.KnowledgeBasesClient()
    request = dialogflow_v2.ListKnowledgeBasesRequest(parent='parent_value')
    page_result = client.list_knowledge_bases(request=request)
    for response in page_result:
        print(response)