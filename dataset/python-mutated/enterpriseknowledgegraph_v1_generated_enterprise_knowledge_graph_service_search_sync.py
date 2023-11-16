from google.cloud import enterpriseknowledgegraph_v1

def sample_search():
    if False:
        while True:
            i = 10
    client = enterpriseknowledgegraph_v1.EnterpriseKnowledgeGraphServiceClient()
    request = enterpriseknowledgegraph_v1.SearchRequest(parent='parent_value', query='query_value')
    response = client.search(request=request)
    print(response)