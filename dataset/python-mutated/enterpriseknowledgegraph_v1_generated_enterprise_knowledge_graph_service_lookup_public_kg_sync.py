from google.cloud import enterpriseknowledgegraph_v1

def sample_lookup_public_kg():
    if False:
        i = 10
        return i + 15
    client = enterpriseknowledgegraph_v1.EnterpriseKnowledgeGraphServiceClient()
    request = enterpriseknowledgegraph_v1.LookupPublicKgRequest(parent='parent_value', ids=['ids_value1', 'ids_value2'])
    response = client.lookup_public_kg(request=request)
    print(response)