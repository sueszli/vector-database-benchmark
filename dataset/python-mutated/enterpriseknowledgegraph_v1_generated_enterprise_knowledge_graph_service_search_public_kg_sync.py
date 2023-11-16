from google.cloud import enterpriseknowledgegraph_v1

def sample_search_public_kg():
    if False:
        i = 10
        return i + 15
    client = enterpriseknowledgegraph_v1.EnterpriseKnowledgeGraphServiceClient()
    request = enterpriseknowledgegraph_v1.SearchPublicKgRequest(parent='parent_value', query='query_value')
    response = client.search_public_kg(request=request)
    print(response)