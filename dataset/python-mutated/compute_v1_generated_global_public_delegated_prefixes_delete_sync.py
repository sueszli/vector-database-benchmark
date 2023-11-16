from google.cloud import compute_v1

def sample_delete():
    if False:
        i = 10
        return i + 15
    client = compute_v1.GlobalPublicDelegatedPrefixesClient()
    request = compute_v1.DeleteGlobalPublicDelegatedPrefixeRequest(project='project_value', public_delegated_prefix='public_delegated_prefix_value')
    response = client.delete(request=request)
    print(response)