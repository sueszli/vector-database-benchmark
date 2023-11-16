from google.cloud import compute_v1

def sample_patch():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.GlobalPublicDelegatedPrefixesClient()
    request = compute_v1.PatchGlobalPublicDelegatedPrefixeRequest(project='project_value', public_delegated_prefix='public_delegated_prefix_value')
    response = client.patch(request=request)
    print(response)