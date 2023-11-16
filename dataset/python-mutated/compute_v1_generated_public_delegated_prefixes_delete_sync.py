from google.cloud import compute_v1

def sample_delete():
    if False:
        while True:
            i = 10
    client = compute_v1.PublicDelegatedPrefixesClient()
    request = compute_v1.DeletePublicDelegatedPrefixeRequest(project='project_value', public_delegated_prefix='public_delegated_prefix_value', region='region_value')
    response = client.delete(request=request)
    print(response)