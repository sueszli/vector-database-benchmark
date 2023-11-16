from google.cloud import enterpriseknowledgegraph_v1

def sample_lookup():
    if False:
        for i in range(10):
            print('nop')
    client = enterpriseknowledgegraph_v1.EnterpriseKnowledgeGraphServiceClient()
    request = enterpriseknowledgegraph_v1.LookupRequest(parent='parent_value', ids=['ids_value1', 'ids_value2'])
    response = client.lookup(request=request)
    print(response)