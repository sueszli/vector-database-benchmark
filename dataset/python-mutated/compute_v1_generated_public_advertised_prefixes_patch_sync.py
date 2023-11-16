from google.cloud import compute_v1

def sample_patch():
    if False:
        print('Hello World!')
    client = compute_v1.PublicAdvertisedPrefixesClient()
    request = compute_v1.PatchPublicAdvertisedPrefixeRequest(project='project_value', public_advertised_prefix='public_advertised_prefix_value')
    response = client.patch(request=request)
    print(response)