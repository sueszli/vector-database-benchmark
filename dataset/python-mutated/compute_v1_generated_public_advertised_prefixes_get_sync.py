from google.cloud import compute_v1

def sample_get():
    if False:
        while True:
            i = 10
    client = compute_v1.PublicAdvertisedPrefixesClient()
    request = compute_v1.GetPublicAdvertisedPrefixeRequest(project='project_value', public_advertised_prefix='public_advertised_prefix_value')
    response = client.get(request=request)
    print(response)