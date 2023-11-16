from google.cloud import compute_v1

def sample_get():
    if False:
        print('Hello World!')
    client = compute_v1.GlobalPublicDelegatedPrefixesClient()
    request = compute_v1.GetGlobalPublicDelegatedPrefixeRequest(project='project_value', public_delegated_prefix='public_delegated_prefix_value')
    response = client.get(request=request)
    print(response)