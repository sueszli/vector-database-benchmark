from google.cloud import compute_v1

def sample_update():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.UrlMapsClient()
    request = compute_v1.UpdateUrlMapRequest(project='project_value', url_map='url_map_value')
    response = client.update(request=request)
    print(response)