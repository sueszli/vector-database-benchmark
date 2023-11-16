from google.cloud import compute_v1

def sample_delete():
    if False:
        i = 10
        return i + 15
    client = compute_v1.PublicAdvertisedPrefixesClient()
    request = compute_v1.DeletePublicAdvertisedPrefixeRequest(project='project_value', public_advertised_prefix='public_advertised_prefix_value')
    response = client.delete(request=request)
    print(response)