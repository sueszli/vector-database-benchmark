from google.cloud import compute_v1

def sample_delete():
    if False:
        print('Hello World!')
    client = compute_v1.UrlMapsClient()
    request = compute_v1.DeleteUrlMapRequest(project='project_value', url_map='url_map_value')
    response = client.delete(request=request)
    print(response)