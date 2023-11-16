from google.cloud import dialogflow_v2

def sample_search_knowledge():
    if False:
        i = 10
        return i + 15
    client = dialogflow_v2.ConversationsClient()
    query = dialogflow_v2.TextInput()
    query.text = 'text_value'
    query.language_code = 'language_code_value'
    request = dialogflow_v2.SearchKnowledgeRequest(query=query, conversation_profile='conversation_profile_value')
    response = client.search_knowledge(request=request)
    print(response)