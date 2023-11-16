from google.cloud import compute_v1

def sample_patch():
    if False:
        print('Hello World!')
    client = compute_v1.PublicDelegatedPrefixesClient()
    request = compute_v1.PatchPublicDelegatedPrefixeRequest(project='project_value', public_delegated_prefix='public_delegated_prefix_value', region='region_value')
    response = client.patch(request=request)
    print(response)