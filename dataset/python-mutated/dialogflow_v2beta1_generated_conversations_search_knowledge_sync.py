from google.cloud import dialogflow_v2beta1

def sample_search_knowledge():
    if False:
        print('Hello World!')
    client = dialogflow_v2beta1.ConversationsClient()
    request = dialogflow_v2beta1.SearchKnowledgeRequest(conversation_profile='conversation_profile_value')
    response = client.search_knowledge(request=request)
    print(response)