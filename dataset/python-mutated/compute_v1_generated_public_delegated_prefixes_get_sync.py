from google.cloud import compute_v1

def sample_get():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.PublicDelegatedPrefixesClient()
    request = compute_v1.GetPublicDelegatedPrefixeRequest(project='project_value', public_delegated_prefix='public_delegated_prefix_value', region='region_value')
    response = client.get(request=request)
    print(response)