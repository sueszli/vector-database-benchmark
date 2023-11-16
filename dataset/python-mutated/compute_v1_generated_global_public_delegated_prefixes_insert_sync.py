from google.cloud import compute_v1

def sample_insert():
    if False:
        return 10
    client = compute_v1.GlobalPublicDelegatedPrefixesClient()
    request = compute_v1.InsertGlobalPublicDelegatedPrefixeRequest(project='project_value')
    response = client.insert(request=request)
    print(response)