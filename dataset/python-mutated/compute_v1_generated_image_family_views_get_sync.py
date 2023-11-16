from google.cloud import compute_v1

def sample_get():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.ImageFamilyViewsClient()
    request = compute_v1.GetImageFamilyViewRequest(family='family_value', project='project_value', zone='zone_value')
    response = client.get(request=request)
    print(response)